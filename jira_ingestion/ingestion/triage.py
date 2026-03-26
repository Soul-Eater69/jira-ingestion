"""
Four-layer attachment triage funnel + multi-score triage artifact.

Layer 0: Extension + size filter  (free — API metadata only)
Layer 1: Filename heuristic scoring  (free — string parsing)
Layer 2: Cheap metadata peek  (cheap — 100-200ms, no full parsing)
Layer 3: Full extraction + confirmation  (moderate — only the winner)

build_triage_artifact() assembles the first-class triage output with:
  - primary / supplementary decision
  - quality tier
  - selection reason
  - multi-dimensional scores (extraction_quality, semantic_density,
    idea_card_likeness, retrieval_readiness)
  - per-attachment score breakdown
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

# ---------------------------------------------------------------------------
# Multi-score triage artifact (first-class stage output)
# ---------------------------------------------------------------------------

def build_triage_artifact(
    primary: Optional[dict],
    supplementary: list[dict],
    att_quality: str,
    all_scored: list[dict],
    total_attachment_count: int = 0,
) -> dict:
    """
    Build a structured triage artifact from the triage funnel result.

    This is the persisted output of the triage stage (03_triage_output.json).
    All downstream layers consume this artifact instead of raw triage state.

    Args:
        primary:                 Winning attachment dict (enriched by triage) or None.
        supplementary:           Runner-up attachment dicts.
        att_quality:             'good' | 'fallback' | 'none'.
        all_scored:              All Layer-1 scored candidates.
        total_attachment_count:  Total attachments on the ticket (pre-filter).

    Returns:
        TriageArtifact-compatible dict.
    """
    viable_count = len(all_scored)

    # Build per-attachment score breakdown
    per_att: list[dict] = []
    for att in all_scored:
        scores = _compute_att_scores(att)
        per_att.append({
            "filename": att.get("filename", ""),
            "attachment_id": str(att.get("id", "")),
            "ext": att.get("ext", ""),
            "size": att.get("size", 0),
            "triage_score": att.get("triage_score", 0),
            "triage_reasons": att.get("triage_reasons", []),
            "confirmed": att.get("confirmed", False),
            "scores": scores,
        })

    # Aggregate scores for the primary attachment
    if primary:
        primary_scores = _compute_att_scores(primary)
    else:
        primary_scores = {
            "extraction_quality": 0.0,
            "semantic_density": 0.0,
            "idea_card_likeness": 0.0,
            "retrieval_readiness": 0.0,
        }

    # Derive quality tier from att_quality + primary scores
    quality_tier = _derive_quality_tier(att_quality, primary, primary_scores)

    # Build human-readable selection reason
    selection_reason = _build_selection_reason(primary, att_quality, primary_scores)

    return {
        "primary_attachment": primary["filename"] if primary else None,
        "primary_attachment_id": str(primary.get("id", "")) if primary else None,
        "supplementary_attachments": [s["filename"] for s in supplementary],
        "att_quality": att_quality,
        "quality_tier": quality_tier,
        "selection_reason": selection_reason,
        "scores": primary_scores,
        "per_attachment_scores": per_att,
        "attachment_count_total": total_attachment_count,
        "attachment_count_viable": viable_count,
        # Backward compat with v2.0 schema
        "triage_score": primary.get("triage_score") if primary else None,
        "triage_reasons": primary.get("triage_reasons", []) if primary else [],
    }


def _compute_att_scores(att: dict) -> dict:
    """
    Compute multi-dimensional scores for one attachment using available signals.

    These are proxy scores based on triage metadata (no full text analysis).
    They capture what we know at triage time about each file.

    Scores:
        extraction_quality   — how cleanly can text be extracted?
        semantic_density     — how likely is it to contain business signal?
        idea_card_likeness   — how likely is it the primary idea card?
        retrieval_readiness  — how useful for value-stream retrieval?
    """
    ext = att.get("ext", "").lower()
    triage_score = att.get("triage_score", 0)
    confirmed = att.get("confirmed", False)
    peek = att.get("peek_metadata", {})
    size = att.get("size", 0)

    # --- extraction_quality ---
    # pptx/docx have native XML text = high quality
    # pdf can be native or scanned (treat as moderate without OCR info)
    ext_quality = {"pptx": 0.90, "ppt": 0.85, "docx": 0.90, "doc": 0.85, "pdf": 0.70, "xlsx": 0.50, "csv": 0.40}
    extraction_quality = ext_quality.get(ext, 0.50)
    if peek.get("is_likely_template"):
        extraction_quality *= 0.6
    if ext == "pdf" and size > 500_000:
        # Large PDFs often have native text
        extraction_quality = min(0.85, extraction_quality + 0.10)

    # --- semantic_density ---
    # Normalize triage score to 0-1; min=0, reasonable max=100
    semantic_density = max(0.0, min(1.0, triage_score / 100.0))
    if peek.get("is_likely_idea_card"):
        semantic_density = min(1.0, semantic_density + 0.20)
    slide_count = peek.get("slide_count") or 0
    if slide_count and slide_count > 60:
        semantic_density = max(0.0, semantic_density - 0.10)

    # --- idea_card_likeness ---
    if confirmed:
        idea_card_likeness = 0.90
    elif peek.get("is_likely_idea_card"):
        idea_card_likeness = 0.70
    elif triage_score >= 40:
        idea_card_likeness = 0.55
    elif triage_score >= 20:
        idea_card_likeness = 0.35
    else:
        idea_card_likeness = 0.15

    # Boost pptx — most idea cards are decks
    if ext in ("pptx", "ppt"):
        idea_card_likeness = min(1.0, idea_card_likeness + 0.05)

    # --- retrieval_readiness ---
    # Combination of the above — an attachment is retrieval-ready if it has
    # good extraction + semantic content + is likely the idea card
    retrieval_readiness = round(
        0.30 * extraction_quality
        + 0.30 * semantic_density
        + 0.40 * idea_card_likeness,
        4,
    )

    return {
        "extraction_quality": round(extraction_quality, 4),
        "semantic_density": round(semantic_density, 4),
        "idea_card_likeness": round(idea_card_likeness, 4),
        "retrieval_readiness": round(retrieval_readiness, 4),
    }


def _derive_quality_tier(att_quality: str, primary: Optional[dict], scores: dict) -> str:
    """Map triage result to a quality tier label for the triage artifact."""
    if not primary or att_quality == "none":
        return "none"
    if att_quality == "good":
        return "A"
    # fallback — differentiate on score
    if scores.get("retrieval_readiness", 0) >= 0.55:
        return "B"
    return "C"


def _build_selection_reason(primary: Optional[dict], att_quality: str, scores: dict) -> str:
    """Build a human-readable explanation of why the primary was selected."""
    if not primary:
        return "No viable attachments found after extension and size filtering."
    filename = primary.get("filename", "")
    reasons = primary.get("triage_reasons", [])
    top_reasons = "; ".join(reasons[:3]) if reasons else "filename heuristic scoring"
    confirmed = primary.get("confirmed", False)

    if att_quality == "good" and confirmed:
        return (
            f"'{filename}' selected as primary — confirmed idea card structure. "
            f"Scoring signals: {top_reasons}."
        )
    if att_quality == "good":
        return (
            f"'{filename}' selected as primary — passed all triage layers. "
            f"Scoring signals: {top_reasons}."
        )
    rr = scores.get("retrieval_readiness", 0)
    return (
        f"'{filename}' selected as fallback primary (quality tier={_derive_quality_tier(att_quality, primary, scores)}, "
        f"retrieval_readiness={rr:.2f}). "
        f"Scoring signals: {top_reasons}."
    )


# ---------------------------------------------------------------------------
# Layer 3 helper: lightweight full-text extraction for confirmation
# ---------------------------------------------------------------------------

def _full_extract_text(file_bytes: bytes, ext: str) -> dict:
    """
    Run just enough extraction to confirm the document is an idea card.
    All formats go through MarkItDown via their respective module.
    """
    try:
        if ext in ("pptx", "ppt"):
            from .extraction.pptx import extract_pptx
            result = extract_pptx(file_bytes, max_slides=60)
        elif ext == "pdf":
            from .extraction.pdf import extract_pdf
            result = extract_pdf(file_bytes)
        elif ext in ("docx", "doc"):
            from .extraction.docx import extract_docx
            result = extract_docx(file_bytes)
        else:
            return {}

        non_bp = sum(1 for c in result["chunks"] if not c.get("is_boilerplate"))
        text = " ".join(c["text"] for c in result["chunks"] if not c.get("is_boilerplate"))
        return {"text": text, "non_boilerplate_count": non_bp}

    except Exception as exc:
        logger.debug("Full extraction failed for %s: %s", ext, exc)

    return {}
