"""
Table serialization utilities shared across PPTX, PDF, and DOCX extractors.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def serialize_table(
    rows: list[list[str]],
    context: Optional[str] = None,
    max_data_rows: int = 15,
) -> dict:
    """
    Convert a list-of-rows (including header row) into a readable text block.

    Args:
        rows:          [[header1, header2, ...], [val1, val2, ...], ...]
        context:       Optional source label prepended to the output.
        max_data_rows: Maximum number of data rows to serialise (rest are summarised).

    Returns:
        {"full_text": str, "summary": str, "row_count": int, "col_count": int}
    """
    if not rows or len(rows) < 1:
        return {"full_text": "", "summary": "", "row_count": 0, "col_count": 0}

    headers = [cell.strip() for cell in rows[0]]
    data_rows = [[cell.strip() for cell in row] for row in rows[1:]]

    parts: list[str] = []
    if context:
        parts.append(f"Table from: {context}")
    parts.append(f"Columns: {' | '.join(headers)}")

    for i, row in enumerate(data_rows[:max_data_rows]):
        cells = []
        for j, cell in enumerate(row):
            if j < len(headers) and headers[j]:
                cells.append(f"{headers[j]}: {cell}")
            else:
                cells.append(cell)
        parts.append(f"Row {i + 1}: {' | '.join(cells)}")

    if len(data_rows) > max_data_rows:
        parts.append(f"... and {len(data_rows) - max_data_rows} more rows")

    full_text = "\n".join(parts)
    summary = _quick_summary(headers, data_rows[:5])

    return {
        "full_text": full_text,
        "summary": summary,
        "row_count": len(data_rows),
        "col_count": len(headers),
    }


def serialize_pptx_table(table_obj: Any, context: Optional[str] = None) -> dict:
    """Serialise a python-pptx Table object."""
    rows = [[cell.text.strip() for cell in row.cells] for row in table_obj.rows]
    return serialize_table(rows, context=context)


def serialize_docx_table(table_obj: Any, context: Optional[str] = None) -> dict:
    """Serialise a python-docx Table object."""
    rows = [[cell.text.strip() for cell in row.cells] for row in table_obj.rows]
    return serialize_table(rows, context=context)


def _quick_summary(headers: list[str], sample_rows: list[list[str]]) -> str:
    """Generate a one-line summary without an LLM call."""
    if not headers:
        return ""
    col_names = ", ".join(h for h in headers if h)
    row_preview = ""
    if sample_rows:
        first = sample_rows[0]
        vals = [v for v in first if v][:3]
        if vals:
            row_preview = f" (e.g. {'; '.join(vals)})"
    return f"Table with columns: {col_names}{row_preview}"
