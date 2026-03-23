"""
Data quality tier classification.

Tier A: Attachment + usable description   → weight 1.0
Tier B: Attachment only                   → weight 0.85
Tier C: Description only                  → weight 0.60
Tier D: Metadata only                     → weight 0.30
"""

from __future__ import annotations

TIER_WEIGHTS: dict[str, float] = {
    "A": 1.0,
    "B": 0.85,
    "C": 0.60,
    "D": 0.30,
}


def determine_quality_tier(
    content_source: str | None,
    desc_class: str,
    chunks: list[dict],
) -> str:
    """
    Determine the quality tier for a ticket based on available content.

    Args:
        content_source: 'pptx' | 'pdf' | 'docx' | None
        desc_class:     'empty' | 'junk' | 'thin' | 'usable' | 'rich'
        chunks:         All extracted content chunks so far.
    """
    has_attachment = content_source in ("pptx", "ppt", "pdf", "docx", "doc")
    has_usable_desc = desc_class in ("rich", "usable")
    has_thin_desc = desc_class == "thin"

    if has_attachment:
        return "A" if has_usable_desc else "B"

    if (has_usable_desc or has_thin_desc) and chunks:
        return "C"

    return "D"
