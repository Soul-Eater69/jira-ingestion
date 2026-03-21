"""
PPTX content extraction using python-pptx.

Returns a list of slide-level chunk dicts ready for the chunking layer.
"""

from __future__ import annotations

import io
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Optional

from .tables import serialize_pptx_table

logger = logging.getLogger(__name__)

# Slide layouts that are boilerplate by name
_BOILERPLATE_LAYOUTS = {"Title Slide", "Title Only", "Blank"}

# Title patterns that indicate boilerplate slides
_BOILERPLATE_TITLE_RE = re.compile(
    r"^(agenda|outline|table of contents|thank|questions|q\s*&\s*a|appendix|disclaimer|legal|confidential)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pptx(file_bytes: bytes, max_slides: int = 60) -> dict:
    """
    Extract structured content from a PPTX file.

    Returns:
        {
            "chunks": [slide-chunk-dict, ...],
            "slide_count": int,
            "has_tables": bool,
            "has_notes": bool,
        }
    """
    try:
        from pptx import Presentation  # type: ignore
        from pptx.util import Pt  # noqa: F401
    except ImportError as exc:
        raise ImportError("python-pptx is required for PPTX extraction") from exc

    prs = Presentation(io.BytesIO(file_bytes))
    slides = prs.slides
    total = len(slides)
    chunks: list[dict] = []

    for idx, slide in enumerate(slides):
        if idx >= max_slides:
            logger.info("Capped PPTX at %d slides (total=%d)", max_slides, total)
            break

        slide_num = idx + 1
        layout_name = _get_layout_name(slide)
        title = _get_slide_title(slide)

        # Check boilerplate
        if _is_boilerplate_slide(slide_num, title, layout_name):
            chunks.append(_make_boilerplate_chunk(slide_num, title, layout_name))
            continue

        body_parts: list[str] = []
        table_chunks: list[dict] = []

        for shape in slide.shapes:
            if not shape.has_text_frame and not shape.has_table:
                continue

            if shape.has_table:
                tbl = serialize_pptx_table(shape.table, context=f"Slide {slide_num}: {title}")
                if tbl["full_text"]:
                    body_parts.append(tbl["full_text"])
                    table_chunks.append(
                        _make_table_chunk(slide_num, title, tbl, idx_within_slide=len(table_chunks))
                    )
                continue

            # Text frame
            for para in shape.text_frame.paragraphs:
                text = para.text.strip()
                if text and text != title:
                    body_parts.append(text)

        # Speaker notes
        notes_text = _get_notes_text(slide)

        # Assemble slide text
        text_parts: list[str] = []
        if title:
            text_parts.append(f"## {title}")
        if body_parts:
            text_parts.append("\n".join(body_parts))
        if notes_text:
            text_parts.append(f"\n--- Speaker Notes ---\n{notes_text}")

        full_text = "\n\n".join(text_parts).strip()
        if not full_text:
            continue

        word_count = len(full_text.split())
        chunk: dict = {
            "chunk_id": f"slide-{slide_num}",
            "source": "pptx_slide",
            "text": full_text,
            "slide_num": slide_num,
            "slide_title": title or "",
            "layout_name": layout_name or "",
            "has_table": bool(table_chunks),
            "has_notes": bool(notes_text),
            "word_count": word_count,
            "is_boilerplate": False,
            "weight_multiplier": 1.0,
            "extraction_confidence": 1.0,
            "extraction_method": "pptx_native",
        }
        chunks.append(chunk)

        # Add individual table chunks immediately after their parent slide chunk
        chunks.extend(table_chunks)

        # If notes are long enough, also create a standalone notes chunk
        if notes_text and len(notes_text.split()) > 100:
            chunks.append(
                {
                    "chunk_id": f"slide-{slide_num}-notes",
                    "source": "pptx_notes",
                    "text": f"## Notes for slide {slide_num}: {title}\n\n{notes_text}",
                    "slide_num": slide_num,
                    "slide_title": title or "",
                    "has_table": False,
                    "has_notes": True,
                    "word_count": len(notes_text.split()),
                    "is_boilerplate": False,
                    "weight_multiplier": 0.8,
                    "extraction_confidence": 1.0,
                    "extraction_method": "pptx_native",
                }
            )

    has_tables = any(c.get("has_table") for c in chunks)
    has_notes = any(c.get("has_notes") for c in chunks)

    return {
        "chunks": chunks,
        "slide_count": total,
        "has_tables": has_tables,
        "has_notes": has_notes,
    }


# ---------------------------------------------------------------------------
# Cheap peek (Layer 2 triage — no full parsing)
# ---------------------------------------------------------------------------

def cheap_peek_pptx(file_bytes: bytes) -> dict:
    """
    Read structural metadata from a PPTX without fully parsing it.
    Uses only the ZIP directory and slide1.xml.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(file_bytes))
    except Exception as exc:
        return {"error": str(exc)}

    file_list = zf.namelist()
    slide_files = [
        f for f in file_list
        if f.startswith("ppt/slides/slide") and f.endswith(".xml")
        and not "slideLayout" in f
    ]
    slide_count = len(slide_files)

    first_title: Optional[str] = None
    if "ppt/slides/slide1.xml" in file_list:
        try:
            root = ET.fromstring(zf.read("ppt/slides/slide1.xml"))
            ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
            texts = root.findall(".//a:t", ns)
            if texts:
                first_title = " ".join((t.text or "") for t in texts[:5]).strip()
        except ET.ParseError:
            pass

    has_tables = False
    for f in slide_files[:3]:
        try:
            if b"<a:tbl" in zf.read(f):
                has_tables = True
                break
        except Exception:
            pass

    note_files = [f for f in file_list if f.startswith("ppt/notesSlides/")]
    text_volume = sum(
        zf.getinfo(f).file_size for f in slide_files if f in {i.filename for i in zf.infolist()}
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

def _get_layout_name(slide: Any) -> Optional[str]:
    try:
        return slide.slide_layout.name
    except Exception:
        return None


def _get_slide_title(slide: Any) -> Optional[str]:
    try:
        if slide.shapes.title and slide.shapes.title.has_text_frame:
            return slide.shapes.title.text.strip() or None
    except Exception:
        pass
    return None


def _get_notes_text(slide: Any) -> str:
    try:
        if slide.has_notes_slide:
            tf = slide.notes_slide.notes_text_frame
            lines = [p.text.strip() for p in tf.paragraphs if p.text.strip()]
            return "\n".join(lines)
    except Exception:
        pass
    return ""


def _is_boilerplate_slide(slide_num: int, title: Optional[str], layout_name: Optional[str]) -> bool:
    if layout_name in _BOILERPLATE_LAYOUTS:
        return True
    if title and _BOILERPLATE_TITLE_RE.match(title.strip()):
        return True
    # Slide 1 with cover layout
    if slide_num == 1 and layout_name and "title" in layout_name.lower():
        return True
    return False


def _make_boilerplate_chunk(slide_num: int, title: Optional[str], layout_name: Optional[str]) -> dict:
    return {
        "chunk_id": f"slide-{slide_num}",
        "source": "pptx_slide",
        "text": title or f"Slide {slide_num}",
        "slide_num": slide_num,
        "slide_title": title or "",
        "layout_name": layout_name or "",
        "has_table": False,
        "has_notes": False,
        "word_count": len((title or "").split()),
        "is_boilerplate": True,
        "weight_multiplier": 0.0,
        "extraction_confidence": 1.0,
        "extraction_method": "pptx_native",
    }


def _make_table_chunk(
    slide_num: int,
    slide_title: Optional[str],
    tbl: dict,
    idx_within_slide: int,
) -> dict:
    return {
        "chunk_id": f"slide-{slide_num}-table-{idx_within_slide}",
        "source": "pptx_table",
        "text": tbl["full_text"],
        "slide_num": slide_num,
        "slide_title": slide_title or "",
        "has_table": True,
        "has_notes": False,
        "table_summary": tbl["summary"],
        "row_count": tbl["row_count"],
        "col_count": tbl["col_count"],
        "word_count": len(tbl["full_text"].split()),
        "is_boilerplate": False,
        "weight_multiplier": 0.9,
        "extraction_confidence": 0.9,
        "extraction_method": "pptx_native",
    }
