"""
DOCX content extraction via MarkItDown.

MarkItDown converts DOCX to Markdown with headings preserved as # markers.
We split on heading boundaries to produce one chunk per logical section.
Tables in the output are kept inline (MarkItDown renders them as Markdown tables).
"""

from __future__ import annotations

import io
import logging
import re
from typing import Optional

from .markitdown import extract_markdown, word_count

logger = logging.getLogger(__name__)

# Match any Markdown heading (# through ######)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

_CHUNK_MIN_WORDS = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_docx(file_bytes: bytes) -> dict:
    """
    Extract section chunks from a DOCX file via MarkItDown.

    Returns:
        {
            "chunks":        [section-chunk-dict, ...],
            "section_count": int,
            "has_tables":    bool,
        }
    """
    md_text = extract_markdown(file_bytes, "document.docx")
    chunks = _split_by_headings(md_text)
    has_tables = any("|" in c["text"] for c in chunks)

    return {
        "chunks": chunks,
        "section_count": len(chunks),
        "has_tables": has_tables,
    }


def cheap_peek_docx(file_bytes: bytes) -> dict:
    """
    Read structural metadata from a DOCX without running MarkItDown.
    Uses only stdlib zipfile to inspect document.xml size and first heading.
    """
    try:
        import zipfile
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            names = zf.namelist()
            if "word/document.xml" not in names:
                return {"error": "Not a valid DOCX"}

            doc_size = zf.getinfo("word/document.xml").file_size
            raw = zf.read("word/document.xml")

            has_tables = b"<w:tbl" in raw
            first_heading: Optional[str] = None

            import xml.etree.ElementTree as ET
            root = ET.fromstring(raw)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            for para in root.findall(".//w:p", ns):
                style = para.find(".//w:pStyle", ns)
                val = (style.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val") or "") if style is not None else ""
                if "Heading" in val:
                    texts = para.findall(".//w:t", ns)
                    first_heading = "".join(t.text or "" for t in texts).strip() or None
                    if first_heading:
                        break

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

def _split_by_headings(md_text: str) -> list[dict]:
    """
    Split Markdown text on heading boundaries.
    Each section = the heading + all content until the next heading.
    """
    # Find all heading positions
    matches = list(_HEADING_RE.finditer(md_text))

    if not matches:
        # No headings — treat entire document as one chunk
        text = md_text.strip()
        if text and word_count(text) >= _CHUNK_MIN_WORDS:
            return [_make_chunk(0, None, text)]
        return []

    chunks: list[dict] = []
    for i, match in enumerate(matches):
        heading_text = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        section_text = md_text[start:end].strip()

        if word_count(section_text) >= _CHUNK_MIN_WORDS:
            chunks.append(_make_chunk(i, heading_text, section_text))

    # Capture any content before the first heading
    preamble = md_text[: matches[0].start()].strip()
    if preamble and word_count(preamble) >= _CHUNK_MIN_WORDS:
        chunks.insert(0, _make_chunk(-1, None, preamble))

    return chunks


def _make_chunk(idx: int, title: Optional[str], text: str) -> dict:
    return {
        "chunk_id": f"docx-section-{idx}",
        "source": "docx_section",
        "text": text,
        "section_title": title or "",
        "word_count": word_count(text),
        "is_boilerplate": False,
        "weight_multiplier": 1.0,
        "extraction_confidence": 0.95,
        "extraction_method": "markitdown",
    }
