"""
Table utilities for MarkItDown-based extraction.

MarkItDown renders tables as Markdown pipe tables:
    | Header1 | Header2 |
    |---------|---------|
    | val1    | val2    |

This module parses those into structured dicts and produces
a readable serialized form for chunking.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Markdown table parser
# ---------------------------------------------------------------------------

def parse_markdown_tables(md_text: str) -> list[dict]:
    """
    Extract all Markdown pipe tables from a text block.

    Returns a list of serialized table dicts (same schema as serialize_table).
    """
    tables: list[dict] = []
    for match in _TABLE_BLOCK_RE.finditer(md_text):
        block = match.group(0)
        rows = _parse_table_block(block)
        if rows:
            tables.append(serialize_table(rows))
    return tables


def serialize_table(
    rows: list[list[str]],
    context: Optional[str] = None,
    max_data_rows: int = 15,
) -> dict:
    """
    Convert a list-of-rows (including header row) into a readable text block.

    Args:
        rows:          [[header1, header2, ...], [val1, val2, ...], ...]
        context:       Optional source label prepended to output.
        max_data_rows: Maximum data rows to serialise.

    Returns:
        {"full_text": str, "summary": str, "row_count": int, "col_count": int}
    """
    if not rows:
        return {"full_text": "", "summary": "", "row_count": 0, "col_count": 0}

    headers = [cell.strip() for cell in rows[0]]
    data_rows = [[cell.strip() for cell in row] for row in rows[1:]]

    parts: list[str] = []
    if context:
        parts.append(f"Table from: {context}")
    parts.append(f"Columns: {' | '.join(h for h in headers if h)}")

    for i, row in enumerate(data_rows[:max_data_rows]):
        cells = []
        for j, cell in enumerate(row):
            label = headers[j] if j < len(headers) and headers[j] else ""
            cells.append(f"{label}: {cell}" if label else cell)
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Matches a contiguous block of pipe-table lines (header + separator + rows)
_TABLE_BLOCK_RE = re.compile(
    r"(\|[^\n]+\|\n)"          # header row
    r"(\|[-:| ]+\|\n)"         # separator row
    r"(\|[^\n]+\|\n?)+",       # one or more data rows
    re.MULTILINE,
)


def _parse_table_block(block: str) -> list[list[str]]:
    """Parse a Markdown pipe-table block into a list of rows."""
    rows: list[list[str]] = []
    for line in block.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        # Skip separator rows (---|---)
        if re.match(r"^\|[-:| ]+\|$", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells:
            rows.append(cells)
    return rows


def _quick_summary(headers: list[str], sample_rows: list[list[str]]) -> str:
    if not headers:
        return ""
    col_names = ", ".join(h for h in headers if h)
    preview = ""
    if sample_rows:
        vals = [v for v in sample_rows[0] if v][:3]
        if vals:
            preview = f" (e.g. {'; '.join(vals)})"
    return f"Table with columns: {col_names}{preview}"
