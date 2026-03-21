"""
Chunking strategy — Section 8 of the architecture spec.

Implements:
  - Section-level chunk assembly (8.2)  [v2]
  - Slide/page chunks inherit from extraction modules
  - Supplementary content chunks (8.6)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# A section boundary is detected when a slide title looks like a header:
# short, no sentence punctuation, starts with a capital letter.
_SECTION_HEADER_RE = re.compile(r"^[A-Z][^.?!,;]{0,49}$")


# ---------------------------------------------------------------------------
# Section chunk assembly
# ---------------------------------------------------------------------------

def build_section_chunks(slide_chunks: list[dict]) -> list[dict]:
    """
    Group consecutive slide/page chunks into logical section chunks.
    Requires at least SECTION_MIN_SLIDES non-boilerplate slides.

    Returns a list of section-level chunk dicts.
    """
    # Filter to only non-boilerplate slides
    content_slides = [c for c in slide_chunks if not c.get("is_boilerplate")]
    if not content_slides:
        return []

    sections: list[dict] = []
    current: dict = {"title": None, "chunks": [], "start_slide": None}

    for chunk in content_slides:
        title = chunk.get("slide_title") or chunk.get("section_title") or ""
        layout = chunk.get("layout_name", "")

        is_new_section = (
            (title and _SECTION_HEADER_RE.match(title) and len(title.split()) <= 6)
            or layout == "Section Header"
            or _is_agenda_boundary(title)
        )

        if is_new_section and current["chunks"]:
            sections.append(_finalize_section(current))
            current = {
                "title": title or None,
                "chunks": [],
                "start_slide": _get_slide_num(chunk),
            }

        if current["title"] is None and title:
            current["title"] = title
        if current["start_slide"] is None:
            current["start_slide"] = _get_slide_num(chunk)

        current["chunks"].append(chunk)

    if current["chunks"]:
        sections.append(_finalize_section(current))

    # If section detection produced just one giant section, use the fallback grouper
    if len(sections) <= 1 and len(content_slides) >= 8:
        logger.debug("Section detection found only 1 section — using fallback grouper")
        sections = _fallback_sections(content_slides)

    return sections


def _finalize_section(section: dict) -> dict:
    chunks = section["chunks"]
    title = section["title"] or "Untitled Section"
    merged_lines: list[str] = [f"## Section: {title}"]

    for c in chunks:
        slide_text = c.get("text", "").strip()
        if slide_text:
            merged_lines.append(slide_text)

    merged_text = "\n\n".join(merged_lines)
    last_slide_num = _get_slide_num(chunks[-1])
    start = section["start_slide"] or _get_slide_num(chunks[0])

    return {
        "chunk_id": f"section-{start}",
        "source": "section",
        "text": merged_text,
        "section_title": title,
        "slide_range": [start, last_slide_num],
        "child_slide_count": len(chunks),
        "word_count": len(merged_text.split()),
        "is_boilerplate": False,
        "weight_multiplier": 1.0,
        "extraction_confidence": min(
            c.get("extraction_confidence", 1.0) for c in chunks
        ),
        "extraction_method": "section_grouping",
    }


def _fallback_sections(slides: list[dict], group_size: int = 4) -> list[dict]:
    """Group slides into equal-sized sections when structure detection fails."""
    sections: list[dict] = []
    for i in range(0, len(slides), group_size):
        group = slides[i : i + group_size]
        title = group[0].get("slide_title") or f"Section {i // group_size + 1}"
        section = {
            "title": title,
            "chunks": group,
            "start_slide": _get_slide_num(group[0]),
        }
        sections.append(_finalize_section(section))
    return sections


def _is_agenda_boundary(title: str) -> bool:
    """Detect agenda-style section titles."""
    agenda_re = re.compile(
        r"^(agenda|outline|overview|introduction|background|problem|solution|"
        r"approach|implementation|timeline|next steps|conclusion|summary|appendix)",
        re.IGNORECASE,
    )
    return bool(title and agenda_re.match(title.strip()))


def _get_slide_num(chunk: dict) -> Optional[int]:
    return chunk.get("slide_num") or chunk.get("page_num")


# ---------------------------------------------------------------------------
# Supplementary chunk builder
# ---------------------------------------------------------------------------

def build_supplementary_chunk(
    file_bytes: bytes,
    ext: str,
    attachment_id: str,
) -> Optional[dict]:
    """
    Extract text from a supplementary attachment and return a single chunk.
    Used for secondary attachments (not the primary idea card).
    """
    try:
        if ext in ("pptx", "ppt"):
            from .extraction.pptx import extract_pptx
            result = extract_pptx(file_bytes)
            text = " ".join(c["text"] for c in result["chunks"] if not c.get("is_boilerplate"))
        elif ext == "pdf":
            from .extraction.pdf import extract_pdf
            result = extract_pdf(file_bytes)
            text = " ".join(c["text"] for c in result["chunks"])
        elif ext in ("docx", "doc"):
            from .extraction.docx import extract_docx
            result = extract_docx(file_bytes)
            text = " ".join(c["text"] for c in result["chunks"])
        else:
            return None

        text = text.strip()[:3000]
        if not text or len(text.split()) < 30:
            return None

        return {
            "chunk_id": f"supp-{ext}-{attachment_id}",
            "source": f"supplementary_{ext}",
            "text": text,
            "word_count": len(text.split()),
            "is_boilerplate": False,
            "weight_multiplier": 0.7,
            "extraction_confidence": 0.8,
            "extraction_method": f"{ext}_native",
        }
    except Exception as exc:
        logger.warning("Supplementary extraction failed for %s: %s", attachment_id, exc)
        return None
