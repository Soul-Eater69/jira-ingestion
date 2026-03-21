"""
Four-layer attachment triage funnel.

Layer 0: Extension + size filter  (free — API metadata only)
Layer 1: Filename heuristic scoring  (free — string parsing)
Layer 2: Cheap metadata peek  (cheap — 100-200ms, no full parsing)
Layer 3: Full extraction + confirmation  (moderate — only the winner)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer 0 constants
# ---------------------------------------------------------------------------

KILL_EXTENSIONS = {
    "png", "jpg", "jpeg", "gif", "svg", "bmp", "ico", "tiff", "webp",
    "mp4", "mov", "avi", "wmv", "mp3", "wav", "m4a", "flac",
    "zip", "rar", "7z", "tar", "gz",
    "exe", "dmg", "msi", "sh", "bat",
}

EXTRACTABLE_EXTENSIONS = {"pptx", "ppt", "pdf", "docx", "doc", "xlsx", "csv", "xls"}

# ---------------------------------------------------------------------------
# Layer 1 scoring signals
# ---------------------------------------------------------------------------

_L1_POSITIVE: list[tuple[str | re.Pattern, int]] = [
    (re.compile(r"\bidea[\s_-]?card\b", re.IGNORECASE), 50),
    (re.compile(r"\bidea\b", re.IGNORECASE), 40),
    (re.compile(r"\bproposal\b|\binitiative\b", re.IGNORECASE), 40),
    (re.compile(r"\bpitch\b|\bconcept\b", re.IGNORECASE), 35),
    (re.compile(r"\bbusiness[\s_-]?case\b", re.IGNORECASE), 30),
    (re.compile(r"\bv\d+\b|final|latest", re.IGNORECASE), 15),
]

_L1_NEGATIVE: list[tuple[str | re.Pattern, int]] = [
    (re.compile(r"\blogo\b|\bbrand\b|\bicon\b", re.IGNORECASE), -50),
    (re.compile(r"\btemplate\b|\bblank\b", re.IGNORECASE), -40),
    (re.compile(r"\bscreenshot\b|\bcapture\b|\bscreen\b", re.IGNORECASE), -30),
    (re.compile(r"\bold\b|\barchive\b|\bbackup\b", re.IGNORECASE), -25),
    (re.compile(r"^\d+$"), -20),                           # purely numeric stem
    (re.compile(r"\bbudget\b|\bfinance\b|\bcost\b", re.IGNORECASE), -20),
    (re.compile(r"\bappendix\b|\bsupplement\b", re.IGNORECASE), -15),
]

_EXT_BONUS: dict[str, int] = {
    "pptx": 20, "ppt": 20,
    "pdf": 5,
    "docx": 10, "doc": 10,
    "xlsx": -15, "xls": -15, "csv": -15,
}

_TEMPLATE_PHRASES = [
    "insert here", "[placeholder]", "lorem ipsum",
    "click to add", "type here", "[your text]",
    "add title", "add text",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def triage_attachments(
    attachments: list[dict],
    ticket_summary: str = "",
    download_fn: Optional[Callable[[dict], bytes]] = None,
) -> tuple[Optional[dict], list[dict], str]:
    """
    Run the four-layer triage funnel.

    Args:
        attachments:   Raw attachment dicts from get_ticket_data()
        ticket_summary: Ticket title (used for filename matching in Layer 1)
        download_fn:   Callable(attachment_dict) -> bytes for Layers 2 & 3.
                       If None, triage stops after Layer 1.

    Returns:
        (primary, supplementary, att_quality)
        - primary:        Winning attachment dict (enriched with triage metadata) or None
        - supplementary:  Up to 2 runner-up attachments
        - att_quality:    'good' | 'fallback' | 'none'
    """
    if not attachments:
        return None, [], "none"

    # Layer 0
    survivors = layer0_filter(attachments)
    if not survivors:
        return None, [], "none"

    # Layer 1
    scored = layer1_score(survivors, ticket_summary=ticket_summary)
    if not scored:
        return None, [], "none"

    scored.sort(key=lambda x: x["triage_score"], reverse=True)
    top_score = scored[0]["triage_score"]
    gap = top_score - (scored[1]["triage_score"] if len(scored) > 1 else 0)

    # Determine how many need a Layer 2 peek
    if top_score >= 60 and gap >= 20:
        peek_candidates = [scored[0]]
        skip_peek = True
    elif top_score >= 30:
        peek_candidates = scored[:2]
        skip_peek = False
    else:
        peek_candidates = scored[:3]
        skip_peek = False

    if download_fn is None:
        # Can't do Layers 2/3 — return Layer 1 winner
        winner = scored[0]
        supp = [s for s in scored[1:3] if s["triage_score"] >= 15]
        return winner, supp, "fallback"

    # Layer 2
    if not skip_peek:
        peek_candidates = layer2_peek(peek_candidates, download_fn)
        peek_candidates.sort(key=lambda x: x["triage_score"], reverse=True)

    # Layer 3 — full extraction + confirmation
    for candidate in peek_candidates:
        try:
            file_bytes = download_fn(candidate)
            extracted = _full_extract_text(file_bytes, candidate["ext"])
            if confirm_is_idea_card(extracted):
                candidate["file_bytes"] = file_bytes
                candidate["confirmed"] = True
                supp = [
                    s for s in scored
                    if s is not candidate and s["triage_score"] >= 15
                ][:2]
                return candidate, supp, "good"
        except Exception as exc:
            logger.warning("Layer 3 failed for %s: %s", candidate.get("filename"), exc)
            continue

    # All candidates failed confirmation — fall back to highest scorer
    winner = scored[0]
    try:
        if "file_bytes" not in winner:
            winner["file_bytes"] = download_fn(winner)
        winner["confirmed"] = False
    except Exception:
        pass

    supp = [s for s in scored[1:3] if s["triage_score"] >= 15]
    return winner, supp, "fallback"


def layer0_filter(attachments: list[dict]) -> list[dict]:
    """Remove obviously unsuitable attachments based on extension and size."""
    survivors: list[dict] = []
    for att in attachments:
        filename = att.get("filename", "")
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        size = att.get("size", 0)

        if ext in KILL_EXTENSIONS:
            continue
        if ext not in EXTRACTABLE_EXTENSIONS:
            continue
        if size < 15_000:
            continue
        if ext == "pdf" and size < 50_000:
            continue
        if size > 100_000_000:
            continue

        survivors.append({**att, "ext": ext})

    return survivors


def layer1_score(
    survivors: list[dict],
    ticket_summary: str = "",
) -> list[dict]:
    """Score each surviving attachment based on filename heuristics."""
    # Summary words for filename matching
    summary_words = set(re.sub(r"[^a-z0-9\s]", "", ticket_summary.lower()).split())
    summary_words -= {"the", "a", "an", "in", "of", "for", "and", "or", "to", "is"}

    # Stem set for duplicate detection
    stems: dict[str, list[str]] = {}
    for att in survivors:
        stem = att["filename"].lower().rsplit(".", 1)[0]
        stems.setdefault(stem, []).append(att["filename"])

    sorted_by_date = sorted(survivors, key=lambda a: a.get("created", ""), reverse=True)

    scored: list[dict] = []
    for rank, att in enumerate(sorted_by_date):
        filename = att.get("filename", "")
        ext = att.get("ext", "")
        stem = filename.lower().rsplit(".", 1)[0]
        name_lower = stem.lower()
        score = 0
        reasons: list[str] = []

        for pattern, pts in _L1_POSITIVE:
            if isinstance(pattern, re.Pattern) and pattern.search(name_lower):
                score += pts
                reasons.append(f"+{pts} ({pattern.pattern[:30]})")

        for pattern, pts in _L1_NEGATIVE:
            if isinstance(pattern, re.Pattern) and pattern.search(name_lower):
                score += pts  # pts is negative
                reasons.append(f"{pts} ({pattern.pattern[:30]})")

        score += _EXT_BONUS.get(ext, 0)
        if _EXT_BONUS.get(ext, 0) != 0:
            reasons.append(f"+{_EXT_BONUS.get(ext, 0)} (ext)")

        # Summary word overlap
        filename_words = set(re.sub(r"[^a-z0-9\s]", "", name_lower).split())
        overlap = filename_words & summary_words
        if len(overlap) >= 2:
            score += 25
            reasons.append(f"+25 (summary words: {', '.join(list(overlap)[:3])})")

        # Uploaded by reporter
        if att.get("is_reporter_upload"):
            score += 10
            reasons.append("+10 (reporter upload)")

        # Most recent
        if rank == 0:
            score += 5
            reasons.append("+5 (most recent)")

        # Duplicate stem penalty
        if len(stems.get(stem, [])) > 1:
            score -= 10
            reasons.append("-10 (duplicate stem)")

        # Descriptive filename (2-6 words)
        word_count = len(re.findall(r"[a-z]+", name_lower))
        if 2 <= word_count <= 6:
            score += 10
            reasons.append("+10 (descriptive name)")

        scored.append({**att, "triage_score": score, "triage_reasons": reasons})

    return scored


def layer2_peek(candidates: list[dict], download_fn: Callable[[dict], bytes]) -> list[dict]:
    """Download candidates and inspect metadata without full extraction."""
    from .extraction.pptx import cheap_peek_pptx
    from .extraction.pdf import cheap_peek_pdf
    from .extraction.docx import cheap_peek_docx

    updated: list[dict] = []
    for att in candidates:
        ext = att.get("ext", "")
        try:
            file_bytes = download_fn(att)
            att = {**att, "_peek_bytes": file_bytes}  # cache for Layer 3

            if ext in ("pptx", "ppt"):
                peek = cheap_peek_pptx(file_bytes)
            elif ext == "pdf":
                peek = cheap_peek_pdf(file_bytes)
            elif ext in ("docx", "doc"):
                peek = cheap_peek_docx(file_bytes)
            else:
                peek = {}

            # Adjust score based on peek results
            score_adj = 0
            if peek.get("is_likely_idea_card"):
                score_adj += 20
            if peek.get("is_likely_template"):
                score_adj -= 30
            if peek.get("slide_count") and peek["slide_count"] > 60:
                score_adj -= 10

            att["triage_score"] = att["triage_score"] + score_adj
            att["peek_metadata"] = peek
        except Exception as exc:
            logger.debug("Peek failed for %s: %s", att.get("filename"), exc)

        updated.append(att)
    return updated


def confirm_is_idea_card(extracted: Optional[dict]) -> bool:
    """Validate that a fully extracted document is indeed an idea card."""
    if not extracted or not extracted.get("text"):
        return False
    text = extracted["text"]
    if len(text.split()) < 100:
        return False
    text_lower = text.lower()
    template_hits = sum(1 for p in _TEMPLATE_PHRASES if p in text_lower)
    if template_hits >= 2:
        return False
    non_bp_count = extracted.get("non_boilerplate_count", 0)
    if non_bp_count < 2:
        return False
    return True


# ---------------------------------------------------------------------------
# Layer 3 helper: lightweight full-text extraction for confirmation
# ---------------------------------------------------------------------------

def _full_extract_text(file_bytes: bytes, ext: str) -> dict:
    """Run just enough extraction to confirm the document is an idea card."""
    try:
        if ext in ("pptx", "ppt"):
            from .extraction.pptx import extract_pptx
            result = extract_pptx(file_bytes, max_slides=60)
            non_bp = sum(1 for c in result["chunks"] if not c.get("is_boilerplate"))
            text = " ".join(c["text"] for c in result["chunks"] if not c.get("is_boilerplate"))
            return {"text": text, "non_boilerplate_count": non_bp}

        elif ext == "pdf":
            from .extraction.pdf import extract_pdf
            result = extract_pdf(file_bytes, max_pages=30)
            text = " ".join(c["text"] for c in result["chunks"])
            return {"text": text, "non_boilerplate_count": len(result["chunks"])}

        elif ext in ("docx", "doc"):
            from .extraction.docx import extract_docx
            result = extract_docx(file_bytes)
            text = " ".join(c["text"] for c in result["chunks"])
            return {"text": text, "non_boilerplate_count": len(result["chunks"])}
    except Exception as exc:
        logger.debug("Full extraction failed for %s: %s", ext, exc)

    return {}
