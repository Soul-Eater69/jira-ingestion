"""
Core MarkItDown extraction helper — shared by all format-specific modules.

MarkItDown converts any supported file to Markdown text. Format-specific
modules (pptx.py, pdf.py, docx.py) parse that Markdown into structured chunks.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_markdown(file_bytes: bytes, filename: str) -> str:
    """
    Convert file bytes to Markdown text via MarkItDown.

    Args:
        file_bytes: Raw file content.
        filename:   Original filename (used by MarkItDown for format detection).

    Returns:
        Markdown string, or empty string on failure.
    """
    try:
        from markitdown import MarkItDown  # type: ignore
    except ImportError as exc:
        raise ImportError("markitdown is required: pip install 'markitdown[all]'") from exc

    md = MarkItDown()
    stream = io.BytesIO(file_bytes)
    stream.name = filename  # MarkItDown uses the name for MIME sniffing
    result = md.convert_stream(stream)
    return result.text_content or ""


def word_count(text: str) -> int:
    return len(text.split())
