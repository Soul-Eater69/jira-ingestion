"""
DOCX content extraction using python-docx.
Chunks by heading-delimited sections. Tables extracted separately.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

from .tables import serialize_docx_table

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_docx(file_bytes: bytes) -> dict:
    """
    Extract text and tables from a DOCX file, chunked by headings.

    Returns:
        {
            "chunks": [section-chunk-dict, ...],
            "section_count": int,
            "has_tables": bool,
        }
    """
    try:
        from docx import Document  # type: ignore
        from docx.oxml.ns import qn  # type: ignore
    except ImportError as exc:
        raise ImportError("python-docx is required for DOCX extraction") from exc

    doc = Document(io.BytesIO(file_bytes))
    chunks: list[dict] = []
    table_chunks: list[dict] = []

    current_heading: Optional[str] = None
    current_paragraphs: list[str] = []
    section_idx = 0

    def flush_section() -> None:
        nonlocal section_idx, current_heading, current_paragraphs
        if not current_paragraphs:
            return
        text_parts: list[str] = []
        if current_heading:
            text_parts.append(f"## {current_heading}")
        text_parts.extend(current_paragraphs)
        full_text = "\n\n".join(text_parts).strip()
        if full_text and len(full_text.split()) >= 5:
            chunks.append(
                {
                    "chunk_id": f"docx-section-{section_idx}",
                    "source": "docx_section",
                    "text": full_text,
                    "section_title": current_heading or "",
                    "word_count": len(full_text.split()),
                    "is_boilerplate": False,
                    "weight_multiplier": 1.0,
                    "extraction_confidence": 1.0,
                    "extraction_method": "docx_native",
                }
            )
        section_idx += 1
        current_heading = None
        current_paragraphs = []

    for block in _iter_block_items(doc):
        kind = block.get("type")
        if kind == "heading":
            flush_section()
            current_heading = block["text"]
        elif kind == "paragraph":
            text = block["text"].strip()
            if text:
                current_paragraphs.append(text)
        elif kind == "table":
            tbl = block["table"]
            serialized = serialize_docx_table(tbl, context=current_heading)
            if serialized["full_text"]:
                table_chunks.append(
                    {
                        "chunk_id": f"docx-table-{len(table_chunks)}",
                        "source": "docx_table",
                        "text": serialized["full_text"],
                        "table_summary": serialized["summary"],
                        "row_count": serialized["row_count"],
                        "col_count": serialized["col_count"],
                        "word_count": len(serialized["full_text"].split()),
                        "is_boilerplate": False,
                        "weight_multiplier": 0.9,
                        "extraction_confidence": 1.0,
                        "extraction_method": "docx_native",
                    }
                )

    flush_section()
    all_chunks = chunks + table_chunks

    return {
        "chunks": all_chunks,
        "section_count": len(chunks),
        "has_tables": bool(table_chunks),
    }


def cheap_peek_docx(file_bytes: bytes) -> dict:
    """
    Read structural metadata from a DOCX without full extraction.
    Uses the document.xml size as a proxy for word count.
    """
    import zipfile as zf_mod

    try:
        with zf_mod.ZipFile(io.BytesIO(file_bytes)) as zf:
            names = zf.namelist()
            doc_size = 0
            first_heading: Optional[str] = None

            if "word/document.xml" in names:
                doc_size = zf.getinfo("word/document.xml").file_size
                # Cheap heading scan
                import xml.etree.ElementTree as ET
                root = ET.fromstring(zf.read("word/document.xml"))
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                for para in root.findall(".//w:p", ns):
                    style = para.find(".//w:pStyle", ns)
                    if style is not None and "Heading" in (style.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val") or ""):
                        texts = para.findall(".//w:t", ns)
                        first_heading = "".join(t.text or "" for t in texts).strip()
                        if first_heading:
                            break

            has_tables = "word/document.xml" in names and b"<w:tbl" in zf.read("word/document.xml")

        # Rough word count estimate: ~6 bytes per word in XML
        estimated_words = doc_size // 6

        return {
            "estimated_word_count": estimated_words,
            "first_heading": first_heading,
            "has_tables": has_tables,
            "is_likely_idea_card": estimated_words > 200,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_block_items(doc: Any):
    """
    Yield paragraphs and tables in document order as dicts.
    Handles the fact that python-docx exposes paragraphs and tables separately
    unless you walk the XML body directly.
    """
    try:
        from docx.oxml.ns import qn  # type: ignore
        from docx.table import Table  # type: ignore
        from docx.text.paragraph import Paragraph  # type: ignore

        body = doc.element.body
        for child in body:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag == "p":
                para = Paragraph(child, doc)
                style_name = para.style.name if para.style else ""
                if "Heading" in style_name:
                    yield {"type": "heading", "text": para.text}
                else:
                    yield {"type": "paragraph", "text": para.text}
            elif tag == "tbl":
                yield {"type": "table", "table": Table(child, doc)}
    except Exception as exc:
        logger.debug("Error iterating DOCX blocks: %s", exc)
        # Fallback: just yield paragraphs
        for para in doc.paragraphs:
            yield {"type": "paragraph", "text": para.text}
