"""
PDF content extraction via MarkItDown.

MarkItDown converts a PDF to plain/Markdown text. Since PDFs have no
reliable page markers in the output, we split the result into fixed-size
paragraph-based chunks. A cheap_peek remains available for triage.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Optional

from .markitdown import extract_markdown, word_count

logger = logging.getLogger(__name__)

# Target word count per chunk when splitting PDF text
_CHUNK_TARGET_WORDS = 400
_CHUNK_MIN_WORDS = 40


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pdf(
    file_bytes: bytes,
    ocr_enabled: bool = True,  # kept for API compatibility; MarkItDown handles OCR internally
    max_pages: int = 60,
) -> dict:
    """
    Extract text chunks from a PDF via MarkItDown.

    Returns:
        {
            "chunks":     [page-chunk-dict, ...],
            "page_count": int,          # estimated from chunk count
            "has_tables": bool,
            "ocr_used":   bool,         # MarkItDown may trigger OCR internally
        }
    """
    md_text = extract_markdown(file_bytes, "document.pdf")
    chunks = _split_into_chunks(md_text)
    has_tables = any("|" in c["text"] for c in chunks)

    return {
        "chunks": chunks,
        "page_count": len(chunks),   # best estimate without page markers
        "has_tables": has_tables,
        "ocr_used": False,           # MarkItDown handles OCR transparently
    }


def cheap_peek_pdf(file_bytes: bytes) -> dict:
    """
    Read basic PDF metadata without running MarkItDown.
    Uses PyMuPDF if available, otherwise falls back to a byte scan.
    """
    try:
        import fitz  # PyMuPDF  # type: ignore
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = len(doc)
        has_text = False
        is_landscape = False
        if page_count > 0:
            page = doc[0]
            has_text = len(page.get_text("text").strip()) > 50
            rect = page.rect
            is_landscape = rect.width > rect.height
        doc.close()
        return {
            "page_count": page_count,
            "has_text_layer": has_text,
            "is_landscape": is_landscape,
            "is_likely_idea_card": has_text and 2 <= page_count <= 80,
        }
    except Exception:
        pass

    # Ultra-cheap fallback: count PDF page objects from raw bytes
    page_count = len(re.findall(rb"/Type\s*/Page\b", file_bytes))
    return {
        "page_count": page_count,
        "has_text_layer": None,
        "is_landscape": None,
        "is_likely_idea_card": 2 <= page_count <= 80,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_into_chunks(md_text: str) -> list[dict]:
    """
    Split PDF Markdown text into paragraph-based chunks.

    Strategy: accumulate paragraphs until we reach _CHUNK_TARGET_WORDS,
    then flush. This produces semantically coherent chunks without
    requiring page-boundary information.
    """
    # Split on double newlines (paragraph boundaries)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", md_text) if p.strip()]

    chunks: list[dict] = []
    current_parts: list[str] = []
    current_words = 0
    chunk_idx = 0

    def flush() -> None:
        nonlocal chunk_idx, current_parts, current_words
        if not current_parts:
            return
        text = "\n\n".join(current_parts)
        wc = word_count(text)
        if wc >= _CHUNK_MIN_WORDS:
            chunks.append(
                {
                    "chunk_id": f"page-{chunk_idx + 1}",
                    "source": "pdf_page",
                    "text": text,
                    "page_num": chunk_idx + 1,
                    "word_count": wc,
                    "is_boilerplate": False,
                    "weight_multiplier": 1.0,
                    "extraction_confidence": 0.90,
                    "extraction_method": "markitdown",
                }
            )
            chunk_idx += 1
        current_parts = []
        current_words = 0

    for para in paragraphs:
        para_words = word_count(para)
        if current_words + para_words > _CHUNK_TARGET_WORDS and current_parts:
            flush()
        current_parts.append(para)
        current_words += para_words

    flush()
    return chunks
