"""
PPTX content extraction via MarkItDown.

MarkItDown converts a PPTX to Markdown with slide boundaries marked as:
    <!-- Slide number: N -->

Each slide becomes one chunk. Speaker notes are included inline by MarkItDown.
The cheap_peek function still uses zipfile (stdlib) for fast triage metadata.
"""

from __future__ import annotations

import io
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Optional

from .markitdown import extract_markdown, word_count

logger = logging.getLogger(__name__)

# MarkItDown slide boundary marker
_SLIDE_MARKER = re.compile(r"<!--\s*Slide number:\s*(\d+)\s*-->")

# Boilerplate slide title patterns
_BOILERPLATE_TITLE_RE = re.compile(
    r"^(agenda|outline|table of contents|thank|questions|q\s*&\s*a|appendix|disclaimer|legal|confidential)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pptx(file_bytes: bytes, max_slides: int = 60) -> dict:
    """
    Extract structured slide chunks from a PPTX file via MarkItDown.

    Returns:
        {
            "chunks":      [slide-chunk-dict, ...],
            "slide_count": int,
            "has_tables":  bool,
            "has_notes":   bool,
        }
    """
    md_text = extract_markdown(file_bytes, "presentation.pptx")
    slides = _parse_slides(md_text, max_slides=max_slides)

    chunks: list[dict] = []
    has_tables = False
    has_notes = False

    for slide_num, slide_text in slides:
        title = _extract_title(slide_text)
        is_bp = _is_boilerplate(slide_num, title, slide_text)
        has_tbl = "|" in slide_text  # markdown table indicator
        has_note = "Notes:" in slide_text or "Speaker Notes" in slide_text.replace("\n", " ")

        if has_tbl:
            has_tables = True
        if has_note:
            has_notes = True

        wc = word_count(slide_text)
        chunks.append(
            {
                "chunk_id": f"slide-{slide_num}",
                "source": "pptx_slide",
                "text": slide_text,
                "slide_num": slide_num,
                "slide_title": title or "",
                "has_table": has_tbl,
                "has_notes": has_note,
                "word_count": wc,
                "is_boilerplate": is_bp,
                "weight_multiplier": 0.0 if is_bp else 1.0,
                "extraction_confidence": 0.95,
                "extraction_method": "markitdown",
            }
        )

    return {
        "chunks": chunks,
        "slide_count": len(slides),
        "has_tables": has_tables,
        "has_notes": has_notes,
    }


# ---------------------------------------------------------------------------
# Cheap peek (Layer 2 triage — no full parsing, no MarkItDown call)
# ---------------------------------------------------------------------------

def cheap_peek_pptx(file_bytes: bytes) -> dict:
    """
    Read structural metadata from a PPTX without any extraction.
    Uses only the ZIP directory + slide1.xml (stdlib only).
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(file_bytes))
    except Exception as exc:
        return {"error": str(exc)}

    file_list = zf.namelist()
    slide_files = [
        f for f in file_list
        if re.match(r"ppt/slides/slide\d+\.xml$", f)
    ]
    slide_count = len(slide_files)

    first_title: Optional[str] = None
    if "ppt/slides/slide1.xml" in file_list:
        try:
            root = ET.fromstring(zf.read("ppt/slides/slide1.xml"))
            ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
            texts = root.findall(".//a:t", ns)
            if texts:
                first_title = " ".join((t.text or "") for t in texts[:5]).strip() or None
        except ET.ParseError:
            pass

    has_tables = any(
        b"<a:tbl" in zf.read(f)
        for f in slide_files[:3]
        if f in {i.filename for i in zf.infolist()}
    )
    note_files = [f for f in file_list if f.startswith("ppt/notesSlides/")]
    text_volume = sum(
        zf.getinfo(f).file_size
        for f in slide_files
        if f in {i.filename for i in zf.infolist()}
    )
    zf.close()

    return {
        "slide_count": slide_count,
        "first_title": first_title,
        "has_tables": has_tables,
        "has_notes": len(note_files) > 0,
        "text_volume_bytes": text_volume,
        "is_likely_idea_card": 3 <= slide_count <= 60 and text_volume > 5_000,
        "is_likely_template": slide_count <= 2 and text_volume < 3_000,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_slides(md_text: str, max_slides: int) -> list[tuple[int, str]]:
    """Split MarkItDown output on slide markers and return (slide_num, text) pairs."""
    parts = _SLIDE_MARKER.split(md_text)
    # parts = ['preamble', '1', 'slide1 text', '2', 'slide2 text', ...]
    slides: list[tuple[int, str]] = []

    for i in range(1, len(parts), 2):
        slide_num = int(parts[i])
        if slide_num > max_slides:
            break
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if text:
            slides.append((slide_num, text))

    # Fallback: MarkItDown didn't emit markers — treat whole text as one slide
    if not slides and md_text.strip():
        slides = [(1, md_text.strip())]

    return slides


def _extract_title(slide_text: str) -> Optional[str]:
    """Return the first heading line from a slide's Markdown text."""
    for line in slide_text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line
    return None


def _is_boilerplate(slide_num: int, title: Optional[str], text: str) -> bool:
    if title and _BOILERPLATE_TITLE_RE.match(title.strip()):
        return True
    if slide_num == 1 and len(text.split()) < 20:
        return True
    if len(text.strip()) < 20:
        return True
    return False
