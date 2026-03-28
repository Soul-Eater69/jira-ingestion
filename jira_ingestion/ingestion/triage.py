"""
Multi-document attachment triage for Jira idea-card ingestion.

Why this version exists
-----------------------
The old triage implementation selected a single primary attachment and the
pipeline typically chunked only that file. That is too lossy for tickets where
supporting PDFs / DOCX / XLSX files contain value-stream, product, or business
context that does not appear in the main deck.

This version keeps triage as a *routing and weighting* layer rather than a hard
single-document gate. It still chooses a primary attachment, but it also:

- identifies multiple viable supporting documents
- returns an explicit processing plan for extraction/chunking
- preserves excluded/noise docs separately
- exposes per-attachment multi-dimensional scores
- makes downstream chunking use `chunk_candidates`, not only the primary doc

Expected caller pattern
-----------------------
    primary, supporting, att_quality, triage_artifact = triage_attachments(
        attachments=ticket_attachments,
        ticket_summary=ticket_title,
        download_fn=download_attachment_bytes,
    )

    chunk_candidates = get_chunking_candidates(triage_artifact)
    # chunk all docs in chunk_candidates, not just `primary`

Notes
-----
- This file assumes the existing extraction modules live beside it, e.g.
  `.extraction.pptx`, `.extraction.pdf`, `.extraction.docx`
- XLS/XLSX/CSV can be scored as supporting docs, but full extraction is only
  implemented here for PPT/PDF/DOCX because those are the current strong
  extractors shown in your repo screenshots.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

KILL_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "gif",
    "svg",
    "bmp",
    "ico",
    "tiff",
    "webp",
    "mp3",
    "wav",
    "m4a",
    "flac",
    "zip",
    "rar",
    "7z",
    "tar",
    "gz",
    "exe",
    "dmg",
    "msi",
    "sh",
    "bat",
}

EXTRACTABLE_EXTENSIONS = {"pptx", "ppt", "pdf", "docx", "doc", "xlsx", "xls", "csv"}

L1_POSITIVE: list[tuple[re.Pattern, int]] = [
    (re.compile(r"\b(idea|initiative|proposal|pitch|concept)\b", re.IGNORECASE), 45),
    (re.compile(r"\b(card|deck|slides?)\b", re.IGNORECASE), 25),
    (re.compile(r"\b(business[\s_-]?case|value[\s_-]?stream)\b", re.IGNORECASE), 20),
    (re.compile(r"\b(final|latest|v\d+)\b", re.IGNORECASE), 10),
    (re.compile(r"\b(roadmap|strategy|opportunity|proposal)\b", re.IGNORECASE), 12),
]

L1_NEGATIVE: list[tuple[re.Pattern, int]] = [
    (re.compile(r"\b(template|sample|placeholder|draft\s*template)\b", re.IGNORECASE), -40),
    (re.compile(r"\b(logo|icon|banner|screenshot|screen\s*shot)\b", re.IGNORECASE), -35),
    (re.compile(r"\b(backup|old|archive|duplicate|copy)\b", re.IGNORECASE), -20),
    (re.compile(r"\b(budget|finance|costing|invoice)\b", re.IGNORECASE), -10),
]

EXT_BONUS = {
    "pptx": 25,
    "ppt": 18,
    "pdf": 15,
    "docx": 12,
    "doc": 8,
    "xlsx": 5,
    "xls": 3,
    "csv": 0,
}

TEMPLATE_PHRASES = [
    "insert here",
    "[placeholder]",
    "lorem ipsum",
    "click to add",
    "type here",
    "your text",
    "add title",
    "add text",
]

BUSINESS_SIGNAL_TERMS = {
    "customer",
    "member",
    "provider",
    "workflow",
    "process",
    "journey",
    "value",
    "stream",
    "benefit",
    "impact",
    "problem",
    "opportunity",
    "solution",
    "initiative",
    "business",
    "product",
    "feature",
    "integration",
    "platform",
    "experience",
    "automation",
    "claims",
    "authorization",
    "clinical",
    "operations",
    "analytics",
    "data",
    "member",
}

MIN_TEXT_WORDS_SUPPORT = 40
MIN_TEXT_WORDS_IDEA = 80
MAX_FULL_EXTRACT_DOCS = 5
MAX_SUPPORTING_DOCS = 4
MAX_PEEK_DOCS = 5


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def triage_attachments(
    attachments: list[dict],
    ticket_summary: str = "",
    download_fn: Optional[Callable[[dict], bytes]] = None,
) -> tuple[Optional[dict], list[dict], str, dict]:
    """
    Multi-document triage.

    Returns:
        (primary, supporting, att_quality, triage_artifact)

    `supporting` are the viable chunk-worthy non-primary docs.
    `triage_artifact` includes the full processing plan and all scored attachments.
    """
    if not attachments:
        artifact = build_triage_artifact(
            primary=None,
            supporting=[],
            excluded=[],
            all_scored=[],
            att_quality="none",
            total_attachment_count=0,
        )
        return None, [], "none", artifact

    # Layer 0: coarse filter
    survivors, excluded = _layer0_filter(attachments)
    if not survivors:
        artifact = build_triage_artifact(
            primary=None,
            supporting=[],
            excluded=excluded,
            all_scored=[],
            att_quality="none",
            total_attachment_count=len(attachments),
        )
        return None, [], "none", artifact

    # Layer 1: lightweight filename/meta scoring
    scored = _layer1_score(survivors, ticket_summary=ticket_summary)
    scored.sort(key=_sort_key, reverse=True)

    # Layer 2: cheap peek on the most promising candidates
    if download_fn:
        peek_candidates = scored[:MAX_PEEK_DOCS]
        _layer2_peek(peek_candidates, download_fn)
        scored.sort(key=_sort_key, reverse=True)

        # Layer 3: full extraction on more than one doc, not first-win-stop
        full_extract_candidates = _select_full_extract_candidates(scored)
        _layer3_extract_many(full_extract_candidates, download_fn)
        scored.sort(key=_sort_key, reverse=True)

    primary = _select_primary(scored)
    supporting = _select_supporting(scored, primary)
    excluded.extend(_select_additional_excluded(scored, primary, supporting))
    att_quality = _derive_att_quality(primary, supporting)

    artifact = build_triage_artifact(
        primary=primary,
        supporting=supporting,
        excluded=excluded,
        all_scored=scored,
        att_quality=att_quality,
        total_attachment_count=len(attachments),
    )
    return primary, supporting, att_quality, artifact


def get_chunking_candidates(triage_artifact: dict) -> list[dict]:
    """Return the attachment dicts that should be chunked downstream."""
    return list(triage_artifact.get("chunk_candidates", []))


# -----------------------------------------------------------------------------
# Layer 0: filter
# -----------------------------------------------------------------------------


def _layer0_filter(attachments: list[dict]) -> tuple[list[dict], list[dict]]:
    survivors: list[dict] = []
    excluded: list[dict] = []

    for raw in attachments:
        att = dict(raw)
        ext = _ext_of(att)
        size = int(att.get("size", 0) or 0)
        att["ext"] = ext
        att["triage_stage"] = "layer0"

        reason: Optional[str] = None
        if ext in KILL_EXTENSIONS:
            reason = f"excluded extension: {ext}"
        elif ext not in EXTRACTABLE_EXTENSIONS:
            reason = f"non-extractable extension: {ext or 'unknown'}"
        elif size and size < 15_000:
            reason = "file too small"
        elif ext == "pdf" and size > 80_000_000:
            reason = "pdf too large"
        elif size > 100_000_000:
            reason = "file too large"

        if reason:
            att["excluded_reason"] = reason
            att["doc_role"] = "excluded_noise"
            excluded.append(att)
            continue

        survivors.append(att)

    return survivors, excluded


# -----------------------------------------------------------------------------
# Layer 1: score using metadata / filename
# -----------------------------------------------------------------------------


def _layer1_score(survivors: list[dict], ticket_summary: str = "") -> list[dict]:
    summary_words = {w for w in _simple_words(ticket_summary) if len(w) >= 3}
    stems: dict[str, list[str]] = {}

    for att in survivors:
        stems.setdefault(_stem(att), []).append(att.get("filename", ""))

    sorted_by_date = sorted(survivors, key=lambda a: a.get("created", ""), reverse=True)

    scored: list[dict] = []
    for rank, att in enumerate(sorted_by_date):
        filename = att.get("filename", "")
        ext = att.get("ext", _ext_of(att))
        name_lower = filename.lower()

        score = 0
        reasons: list[str] = []

        for pattern, pts in L1_POSITIVE:
            if pattern.search(name_lower):
                score += pts
                reasons.append(f"+{pts} filename:{pattern.pattern[:28]}")

        for pattern, pts in L1_NEGATIVE:
            if pattern.search(name_lower):
                score += pts
                reasons.append(f"{pts} filename:{pattern.pattern[:28]}")

        ext_pts = EXT_BONUS.get(ext, 0)
        if ext_pts:
            score += ext_pts
            reasons.append(f"+{ext_pts} ext:{ext}")

        overlap = summary_words.intersection(_simple_words(name_lower))
        if len(overlap) >= 2:
            pts = min(20, 5 * len(overlap))
            score += pts
            reasons.append(f"+{pts} summary-overlap:{', '.join(sorted(list(overlap))[:4])}")

        if att.get("is_reporter_upload"):
            score += 8
            reasons.append("+8 reporter-upload")

        if rank == 0:
            score += 5
            reasons.append("+5 most-recent")

        if len(stems.get(_stem(att), [])) > 1:
            score -= 10
            reasons.append("-10 duplicate-stem")

        wc = len(_simple_words(name_lower))
        if 2 <= wc <= 6:
            score += 8
            reasons.append("+8 descriptive-filename")

        att["triage_score"] = score
        att["triage_reasons"] = reasons
        att["triage_stage"] = "layer1"
        scored.append(att)

    return scored


# -----------------------------------------------------------------------------
# Layer 2: cheap peek
# -----------------------------------------------------------------------------


def _layer2_peek(candidates: list[dict], download_fn: Callable[[dict], bytes]) -> None:
    for att in candidates:
        ext = att.get("ext", "")
        try:
            file_bytes = download_fn(att)
            att["peek_bytes"] = file_bytes
            peek = _cheap_peek(file_bytes, ext)
            att["peek_metadata"] = peek

            score_adj = 0
            if peek.get("is_likely_idea_card"):
                score_adj += 20
            if peek.get("is_likely_template"):
                score_adj -= 30
            if (peek.get("slide_count") or 0) >= 5:
                score_adj += 8
            if (peek.get("word_count") or 0) >= 80:
                score_adj += 8

            att["triage_score"] = att.get("triage_score", 0) + score_adj
            if score_adj:
                att.setdefault("triage_reasons", []).append(f"{score_adj:+d} cheap-peek")
            att["triage_stage"] = "layer2"
        except Exception as exc:
            logger.warning("Cheap peek failed for %s: %s", att.get("filename"), exc)


# -----------------------------------------------------------------------------
# Layer 3: full extraction on multiple docs
# -----------------------------------------------------------------------------


def _select_full_extract_candidates(scored: list[dict]) -> list[dict]:
    """
    Extract more than one doc.

    Strategy:
    - always include the current top-ranked doc
    - include other docs with decent score / likely usefulness
    - cap total docs to keep ingestion bounded
    """
    out: list[dict] = []
    seen: set[str] = set()

    for idx, att in enumerate(scored):
        att_id = _attachment_id(att)
        if att_id in seen:
            continue

        triage_score = att.get("triage_score", 0)
        ext = att.get("ext", "")
        peek = att.get("peek_metadata", {}) or {}

        include = False
        if idx == 0:
            include = True
        elif triage_score >= 40:
            include = True
        elif peek.get("is_likely_idea_card"):
            include = True
        elif ext in {"pdf", "docx", "doc"} and triage_score >= 25:
            include = True

        if not include:
            continue

        seen.add(att_id)
        out.append(att)
        if len(out) >= MAX_FULL_EXTRACT_DOCS:
            break

    return out



def _layer3_extract_many(candidates: list[dict], download_fn: Callable[[dict], bytes]) -> None:
    for att in candidates:
        ext = att.get("ext", "")
        try:
            file_bytes = att.get("peek_bytes") or download_fn(att)
            att["file_bytes"] = file_bytes
            extracted = _full_extract_text(file_bytes, ext)
            att["extracted"] = extracted

            extracted_text = extracted.get("text", "") if extracted else ""
            word_count = len(_simple_words(extracted_text))
            business_signal_count = _business_signal_count(extracted_text)
            confirmed = _confirm_is_idea_card(extracted)
            semantic_density = _semantic_density(extracted_text)
            likely_template = extracted.get("template_phrase_hits", 0) >= 2

            att["confirmed"] = confirmed
            att["word_count"] = word_count
            att["business_signal_count"] = business_signal_count
            att["semantic_density"] = semantic_density
            att["likely_template"] = likely_template

            score_adj = 0
            if confirmed:
                score_adj += 25
            if business_signal_count >= 4:
                score_adj += 10
            if semantic_density >= 0.18:
                score_adj += 8
            if likely_template:
                score_adj -= 25

            att["triage_score"] = att.get("triage_score", 0) + score_adj
            if score_adj:
                att.setdefault("triage_reasons", []).append(f"{score_adj:+d} full-extract")
            att["att_scores"] = _compute_att_scores(att)
            att["triage_stage"] = "layer3"
        except Exception as exc:
            logger.warning("Full extraction failed for %s: %s", att.get("filename"), exc)
            att.setdefault("att_scores", _compute_att_scores(att))

    for att in candidates:
        att.setdefault("att_scores", _compute_att_scores(att))


# -----------------------------------------------------------------------------
# Selection
# -----------------------------------------------------------------------------


def _select_primary(scored: list[dict]) -> Optional[dict]:
    if not scored:
        return None

    confirmed = [a for a in scored if a.get("confirmed")]
    if confirmed:
        primary = max(confirmed, key=_primary_key)
        primary["doc_role"] = "primary_idea_card"
        return primary

    viable = [a for a in scored if _is_supporting_worthy(a)]
    if viable:
        primary = max(viable, key=_primary_key)
        primary["doc_role"] = "primary_fallback"
        return primary

    primary = max(scored, key=_primary_key)
    primary["doc_role"] = "primary_fallback"
    return primary



def _select_supporting(scored: list[dict], primary: Optional[dict]) -> list[dict]:
    if not scored:
        return []

    primary_id = _attachment_id(primary) if primary else None
    supporting: list[dict] = []
    seen_stems: set[str] = set()

    for att in scored:
        att_id = _attachment_id(att)
        if primary_id and att_id == primary_id:
            continue
        if not _is_supporting_worthy(att):
            continue

        stem = _stem(att)
        if stem in seen_stems:
            # Skip obvious duplicates among support docs.
            continue

        att["doc_role"] = "supporting_doc"
        supporting.append(att)
        seen_stems.add(stem)
        if len(supporting) >= MAX_SUPPORTING_DOCS:
            break

    return supporting



def _select_additional_excluded(scored: list[dict], primary: Optional[dict], supporting: list[dict]) -> list[dict]:
    keep_ids = {_attachment_id(a) for a in ([primary] if primary else []) + supporting}
    excluded: list[dict] = []
    for att in scored:
        if _attachment_id(att) in keep_ids:
            continue
        if att.get("triage_score", 0) <= 0 or att.get("likely_template"):
            att["doc_role"] = "excluded_noise"
            if "excluded_reason" not in att:
                att["excluded_reason"] = "low score or template-like"
            excluded.append(att)
    return excluded


# -----------------------------------------------------------------------------
# Artifact / plan
# -----------------------------------------------------------------------------


def build_triage_artifact(
    primary: Optional[dict],
    supporting: list[dict],
    excluded: list[dict],
    all_scored: list[dict],
    att_quality: str,
    total_attachment_count: int = 0,
) -> dict:
    chunk_candidates: list[dict] = []
    if primary:
        chunk_candidates.append(primary)
    for att in supporting:
        if _attachment_id(att) not in {_attachment_id(a) for a in chunk_candidates}:
            chunk_candidates.append(att)

    full_extract = [a for a in chunk_candidates if a.get("extracted")]
    light_extract = [
        a
        for a in all_scored
        if _attachment_id(a) not in {_attachment_id(x) for x in full_extract}
        and _attachment_id(a) not in {_attachment_id(x) for x in excluded}
        and a.get("triage_score", 0) > 10
    ]

    artifact = {
        "primary_attachment": _public_attachment_summary(primary),
        "supporting_attachments": [_public_attachment_summary(a) for a in supporting],
        "excluded_attachments": [_public_attachment_summary(a) for a in excluded],
        "chunk_candidates": chunk_candidates,
        "att_quality": att_quality,
        "quality_tier": _derive_quality_tier(att_quality, primary),
        "selection_reason": _build_selection_reason(primary, supporting),
        "processing_plan": {
            "full_extract": [_attachment_id(a) for a in full_extract],
            "light_extract": [_attachment_id(a) for a in light_extract],
            "skip": [_attachment_id(a) for a in excluded],
            "chunk_candidates": [_attachment_id(a) for a in chunk_candidates],
        },
        "per_attachment_scores": [_public_attachment_summary(a) for a in all_scored],
        "attachment_count_total": total_attachment_count,
        "attachment_count_viable": len(chunk_candidates),
        # backwards-friendly fields
        "primary_attachment_id": _attachment_id(primary) if primary else None,
        "triage_score": primary.get("triage_score") if primary else None,
        "triage_reasons": list(primary.get("triage_reasons", [])) if primary else [],
    }
    return artifact


# -----------------------------------------------------------------------------
# Helpers: scoring / heuristics
# -----------------------------------------------------------------------------


def _compute_att_scores(att: dict) -> dict:
    ext = att.get("ext", "")
    triage_score = float(att.get("triage_score", 0) or 0)
    confirmed = bool(att.get("confirmed", False))
    peek = att.get("peek_metadata", {}) or {}
    size = int(att.get("size", 0) or 0)

    extraction_quality = {
        "pptx": 0.90,
        "ppt": 0.85,
        "pdf": 0.78,
        "docx": 0.85,
        "doc": 0.72,
        "xlsx": 0.45,
        "xls": 0.35,
        "csv": 0.40,
    }.get(ext, 0.50)

    if ext == "pdf" and size >= 500_000:
        extraction_quality = min(0.86, extraction_quality + 0.06)
    if att.get("extracted") and att.get("word_count", 0) >= 80:
        extraction_quality = min(0.95, extraction_quality + 0.05)

    semantic_density = max(0.0, min(1.0, att.get("semantic_density", 0.0)))
    if semantic_density == 0.0:
        semantic_density = max(0.0, min(1.0, triage_score / 100.0))
        if peek.get("is_likely_idea_card"):
            semantic_density = min(1.0, semantic_density + 0.15)

    if confirmed:
        idea_card_likeness = 0.92
    elif peek.get("is_likely_idea_card"):
        idea_card_likeness = 0.70
    elif triage_score >= 40:
        idea_card_likeness = 0.55
    elif triage_score >= 20:
        idea_card_likeness = 0.35
    else:
        idea_card_likeness = 0.15

    if ext in {"pptx", "ppt"}:
        idea_card_likeness = min(1.0, idea_card_likeness + 0.05)

    retrieval_readiness = round(
        0.30 * extraction_quality + 0.30 * semantic_density + 0.40 * idea_card_likeness,
        4,
    )

    return {
        "extraction_quality": round(extraction_quality, 4),
        "semantic_density": round(semantic_density, 4),
        "idea_card_likeness": round(idea_card_likeness, 4),
        "retrieval_readiness": retrieval_readiness,
    }



def _derive_att_quality(primary: Optional[dict], supporting: list[dict]) -> str:
    if not primary:
        return "none"

    scores = primary.get("att_scores") or _compute_att_scores(primary)
    rr = scores.get("retrieval_readiness", 0.0)
    eq = scores.get("extraction_quality", 0.0)

    if primary.get("confirmed") and rr >= 0.65 and eq >= 0.75:
        return "good"
    if rr >= 0.45 or supporting:
        return "fallback"
    return "none"



def _derive_quality_tier(att_quality: str, primary: Optional[dict]) -> str:
    if att_quality == "none" or not primary:
        return "C"
    if primary.get("confirmed") and att_quality == "good":
        return "A"
    if att_quality == "good":
        return "B"
    return "C"



def _build_selection_reason(primary: Optional[dict], supporting: list[dict]) -> str:
    if not primary:
        return "No viable extractable attachments found after filtering."

    filename = primary.get("filename", "attachment")
    reasons = primary.get("triage_reasons", [])[:3]
    top_reasons = ", ".join(reasons) if reasons else "filename/meta heuristics"
    confirmed = primary.get("confirmed", False)
    support_count = len(supporting)

    if confirmed:
        return (
            f"{filename} selected as primary because it was confirmed as the best idea-card-like "
            f"document. Signals: {top_reasons}. Additional supporting docs kept for chunking: {support_count}."
        )
    return (
        f"{filename} selected as primary fallback based on ranking and extraction readiness. "
        f"Signals: {top_reasons}. Additional supporting docs kept for chunking: {support_count}."
    )



def _public_attachment_summary(att: Optional[dict]) -> Optional[dict]:
    if not att:
        return None
    scores = att.get("att_scores") or _compute_att_scores(att)
    return {
        "attachment_id": _attachment_id(att),
        "filename": att.get("filename"),
        "ext": att.get("ext"),
        "size": att.get("size"),
        "doc_role": att.get("doc_role"),
        "triage_score": att.get("triage_score"),
        "triage_reasons": att.get("triage_reasons", []),
        "confirmed": att.get("confirmed", False),
        "excluded_reason": att.get("excluded_reason"),
        "word_count": att.get("word_count"),
        "business_signal_count": att.get("business_signal_count"),
        "scores": scores,
    }



def _cheap_peek(file_bytes: bytes, ext: str) -> dict:
    if ext in {"pptx", "ppt"}:
        from .extraction.pptx import cheap_peek_pptx

        return cheap_peek_pptx(file_bytes)
    if ext == "pdf":
        from .extraction.pdf import cheap_peek_pdf

        return cheap_peek_pdf(file_bytes)
    if ext in {"docx", "doc"}:
        from .extraction.docx import cheap_peek_docx

        return cheap_peek_docx(file_bytes)
    return {}



def _full_extract_text(file_bytes: bytes, ext: str) -> dict:
    """
    Extract enough text to support triage and downstream chunk routing.
    """
    try:
        if ext in {"pptx", "ppt"}:
            from .extraction.pptx import extract_pptx

            result = extract_pptx(file_bytes, max_slides=60)
        elif ext == "pdf":
            from .extraction.pdf import extract_pdf

            result = extract_pdf(file_bytes)
        elif ext in {"docx", "doc"}:
            from .extraction.docx import extract_docx

            result = extract_docx(file_bytes)
        else:
            return {}

        text_chunks: list[str] = []
        non_bp = 0
        template_hits = 0

        for chunk in result.get("chunks", []):
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            text_chunks.append(text)
            if not chunk.get("is_boilerplate"):
                non_bp += 1
            lowered = text.lower()
            template_hits += sum(1 for p in TEMPLATE_PHRASES if p in lowered)

        combined = "\n".join(text_chunks).strip()
        return {
            **result,
            "text": combined,
            "non_boilerplate_count": non_bp,
            "template_phrase_hits": template_hits,
        }
    except Exception as exc:
        logger.debug("Full extraction failed for ext=%s: %s", ext, exc)
        return {}



def _confirm_is_idea_card(extracted: Optional[dict]) -> bool:
    if not extracted:
        return False

    text = (extracted.get("text") or "").strip()
    if not text:
        return False

    words = _simple_words(text)
    if len(words) < MIN_TEXT_WORDS_IDEA:
        return False

    if extracted.get("template_phrase_hits", 0) >= 2:
        return False

    if extracted.get("non_boilerplate_count", 0) < 2:
        return False

    business_signals = _business_signal_count(text)
    if business_signals < 3:
        return False

    return True



def _is_supporting_worthy(att: dict) -> bool:
    scores = att.get("att_scores") or _compute_att_scores(att)
    rr = scores.get("retrieval_readiness", 0.0)

    if att.get("likely_template"):
        return False
    if att.get("confirmed"):
        return True
    if att.get("word_count", 0) >= MIN_TEXT_WORDS_SUPPORT and rr >= 0.35:
        return True
    if att.get("triage_score", 0) >= 35 and att.get("ext") in {"pptx", "ppt", "pdf", "docx", "doc"}:
        return True
    return False



def _primary_key(att: dict) -> tuple[float, float, float, float]:
    scores = att.get("att_scores") or _compute_att_scores(att)
    return (
        1.0 if att.get("confirmed") else 0.0,
        float(scores.get("idea_card_likeness", 0.0)),
        float(scores.get("retrieval_readiness", 0.0)),
        float(att.get("triage_score", 0.0)),
    )



def _sort_key(att: dict) -> tuple[float, float, float, float]:
    scores = att.get("att_scores") or _compute_att_scores(att)
    return (
        float(att.get("triage_score", 0.0)),
        1.0 if att.get("confirmed") else 0.0,
        float(scores.get("retrieval_readiness", 0.0)),
        float(scores.get("idea_card_likeness", 0.0)),
    )


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def _attachment_id(att: Optional[dict]) -> str:
    if not att:
        return ""
    return str(att.get("id") or att.get("attachment_id") or att.get("filename") or "")



def _ext_of(att: dict) -> str:
    filename = (att.get("filename") or "").lower()
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1]



def _stem(att: dict) -> str:
    filename = (att.get("filename") or "").lower().rsplit(".", 1)[0]
    return re.sub(r"[^a-z0-9]+", " ", filename).strip()



def _simple_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())



def _business_signal_count(text: str) -> int:
    words = set(_simple_words(text))
    return sum(1 for term in BUSINESS_SIGNAL_TERMS if term in words)



def _semantic_density(text: str) -> float:
    words = _simple_words(text)
    if not words:
        return 0.0
    unique_ratio = len(set(words)) / max(len(words), 1)
    business_ratio = _business_signal_count(text) / 10.0
    return round(min(1.0, 0.65 * unique_ratio + 0.35 * min(1.0, business_ratio)), 4)


