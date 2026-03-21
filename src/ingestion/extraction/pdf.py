"""
PDF content extraction using PyMuPDF (fitz), camelot/tabula for tables,
and Tesseract OCR as a fallback for scanned pages.
"""

from __future__ import annotations

import io
import logging
import tempfile
import os
from typing import Optional

from .tables import serialize_table

logger = logging.getLogger(__name__)

_OCR_MIN_TEXT_CHARS = 50  # Pages with fewer chars than this are considered scanned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pdf(file_bytes: bytes, ocr_enabled: bool = True, max_pages: int = 60) -> dict:
    """
    Extract text and tables from a PDF.

    Returns:
        {
            "chunks": [page-chunk-dict, ...],
            "page_count": int,
            "has_tables": bool,
            "ocr_used": bool,
        }
    """
    try:
        import fitz  # PyMuPDF  # type: ignore
    except ImportError as exc:
        raise ImportError("PyMuPDF (fitz) is required for PDF extraction") from exc

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page_count = len(doc)
    chunks: list[dict] = []
    ocr_used = False

    for page_num in range(min(page_count, max_pages)):
        page = doc[page_num]
        text = page.get_text("text").strip()  # type: ignore[attr-defined]
        method = "pdf_native"
        confidence = 0.95

        # Scanned page — try OCR
        if len(text) < _OCR_MIN_TEXT_CHARS and ocr_enabled:
            ocr_text = _ocr_page(page)
            if ocr_text:
                text = ocr_text
                method = "tesseract_ocr"
                confidence = 0.60
                ocr_used = True

        if not text:
            continue

        word_count = len(text.split())
        if word_count < 5:
            continue

        chunk: dict = {
            "chunk_id": f"page-{page_num + 1}",
            "source": "pdf_page",
            "text": text,
            "page_num": page_num + 1,
            "word_count": word_count,
            "is_boilerplate": False,
            "weight_multiplier": 1.0,
            "extraction_confidence": confidence,
            "extraction_method": method,
        }
        chunks.append(chunk)

    doc.close()

    # Extract tables from the PDF
    table_chunks = _extract_pdf_tables(file_bytes, page_count)
    chunks.extend(table_chunks)

    has_tables = bool(table_chunks)

    return {
        "chunks": chunks,
        "page_count": page_count,
        "has_tables": has_tables,
        "ocr_used": ocr_used,
    }


def cheap_peek_pdf(file_bytes: bytes) -> dict:
    """
    Read structural metadata from a PDF without full extraction.
    """
    try:
        import fitz  # type: ignore
    except ImportError:
        return {"error": "PyMuPDF not available"}

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = len(doc)
        has_text_layer = False
        is_landscape = False

        if page_count > 0:
            page = doc[0]
            text = page.get_text("text").strip()
            has_text_layer = len(text) > _OCR_MIN_TEXT_CHARS
            rect = page.rect
            is_landscape = rect.width > rect.height

        doc.close()
        return {
            "page_count": page_count,
            "has_text_layer": has_text_layer,
            "is_landscape": is_landscape,
            "is_likely_idea_card": has_text_layer and 2 <= page_count <= 80,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ocr_page(page: Any) -> str:
    """Run Tesseract OCR on a PyMuPDF page object."""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        pix = page.get_pixmap(dpi=200)  # type: ignore[attr-defined]
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img)
    except Exception as exc:
        logger.debug("OCR failed on page: %s", exc)
        return ""


def _extract_pdf_tables(file_bytes: bytes, page_count: int) -> list[dict]:
    """Try camelot first, fall back to tabula for table extraction."""
    chunks: list[dict] = []

    # Write to a temp file (camelot/tabula require file paths)
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        chunks = _camelot_tables(tmp_path, page_count)
        if not chunks:
            chunks = _tabula_tables(tmp_path, page_count)
    except Exception as exc:
        logger.debug("PDF table extraction failed: %s", exc)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return chunks


def _camelot_tables(pdf_path: str, page_count: int) -> list[dict]:
    """Extract tables via camelot-py (lattice mode)."""
    try:
        import camelot  # type: ignore

        pages = f"1-{min(page_count, 60)}"
        tables = camelot.read_pdf(pdf_path, pages=pages, flavor="lattice")
        return _camelot_to_chunks(tables, "camelot_lattice", 0.85)
    except Exception as exc:
        logger.debug("camelot failed: %s", exc)
        return []


def _tabula_tables(pdf_path: str, page_count: int) -> list[dict]:
    """Extract tables via tabula-py (stream mode fallback)."""
    try:
        import tabula  # type: ignore

        dfs = tabula.read_pdf(
            pdf_path,
            pages=f"1-{min(page_count, 60)}",
            multiple_tables=True,
            pandas_options={"header": 0},
        )
        chunks: list[dict] = []
        for i, df in enumerate(dfs):
            if df.empty:
                continue
            rows = [list(df.columns)] + df.fillna("").astype(str).values.tolist()
            tbl = serialize_table(rows)
            if not tbl["full_text"]:
                continue
            chunks.append(
                {
                    "chunk_id": f"pdf-table-tabula-{i}",
                    "source": "pdf_table",
                    "text": tbl["full_text"],
                    "table_summary": tbl["summary"],
                    "row_count": tbl["row_count"],
                    "col_count": tbl["col_count"],
                    "word_count": len(tbl["full_text"].split()),
                    "is_boilerplate": False,
                    "weight_multiplier": 0.85,
                    "extraction_confidence": 0.70,
                    "extraction_method": "tabula_stream",
                }
            )
        return chunks
    except Exception as exc:
        logger.debug("tabula failed: %s", exc)
        return []


def _camelot_to_chunks(tables: Any, method: str, confidence: float) -> list[dict]:
    chunks: list[dict] = []
    for i, table in enumerate(tables):
        try:
            df = table.df
            if df.empty:
                continue
            rows = df.fillna("").astype(str).values.tolist()
            if not rows:
                continue
            tbl = serialize_table(rows)
            if not tbl["full_text"]:
                continue
            chunks.append(
                {
                    "chunk_id": f"pdf-table-{method}-{i}",
                    "source": "pdf_table",
                    "text": tbl["full_text"],
                    "table_summary": tbl["summary"],
                    "row_count": tbl["row_count"],
                    "col_count": tbl["col_count"],
                    "page_num": getattr(table, "page", None),
                    "word_count": len(tbl["full_text"].split()),
                    "is_boilerplate": False,
                    "weight_multiplier": 0.9,
                    "extraction_confidence": confidence,
                    "extraction_method": method,
                }
            )
        except Exception as exc:
            logger.debug("Failed to serialize camelot table %d: %s", i, exc)
    return chunks
