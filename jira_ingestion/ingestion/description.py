"""
Description field handling.

Classifies the Jira description into: empty | junk | thin | usable | rich
and builds appropriate chunks depending on whether an attachment exists.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Junk patterns — descriptions that contain no meaningful signal
# ---------------------------------------------------------------------------

JUNK_PATTERNS: list[re.Pattern] = [
    re.compile(r"^please see attach", re.IGNORECASE),
    re.compile(r"^see attached", re.IGNORECASE),
    re.compile(r"^tbd$", re.IGNORECASE),
    re.compile(r"^todo$", re.IGNORECASE),
    re.compile(r"^n/a$", re.IGNORECASE),
    re.compile(r"^none$", re.IGNORECASE),
    re.compile(r"^\[insert .+\]", re.IGNORECASE),
    re.compile(r"^as discussed", re.IGNORECASE),
    re.compile(r"see confluence", re.IGNORECASE),
    re.compile(r"^details in the deck", re.IGNORECASE),
    re.compile(r"^refer to the attached", re.IGNORECASE),
    re.compile(r"^attached", re.IGNORECASE),
]

# Thresholds (can be overridden via config)
JUNK_MAX_WORDS = 10
THIN_MAX_WORDS = 50
RICH_MIN_WORDS = 150

# Jira markup patterns to strip
_JIRA_MARKUP_RE = re.compile(
    r"\{code.*?\}.*?\{code\}"
    r"|\{panel.*?\}.*?\{panel\}"
    r"|\{noformat.*?\}.*?\{noformat\}"
    r"|\{[a-z]+[^}]*\}",   # generic {tag} markup
    re.DOTALL | re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^h[1-6]\.", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"\|\|.*?\|\||\|.*?\|")
_BOLD_ITALIC_RE = re.compile(r"[*_]{1,2}(.+?)[*_]{1,2}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_description(raw_text: Optional[str]) -> tuple[str, Optional[dict]]:
    """
    Classify the Jira description text.

    Returns:
        (label, data)
        label: 'empty' | 'junk' | 'thin' | 'usable' | 'rich'
        data:  dict with 'text', 'word_count', and strategy hints, or None
    """
    if not raw_text or not raw_text.strip():
        return "empty", None

    text = clean_jira_markup(raw_text)
    word_count = len(text.split())
    text_lower = text.lower().strip()

    if word_count < JUNK_MAX_WORDS:
        return "junk", None

    for pattern in JUNK_PATTERNS:
        if pattern.match(text_lower):
            return "junk", None

    if word_count < THIN_MAX_WORDS:
        return "thin", {"text": text, "needs_enrichment": True, "word_count": word_count}

    if word_count < RICH_MIN_WORDS:
        return "usable", {"text": text, "chunk_strategy": "single", "word_count": word_count}

    return "rich", {"text": text, "chunk_strategy": "split_paragraphs", "word_count": word_count}


def clean_jira_markup(text: str) -> str:
    """Strip Jira wiki markup, leaving plain readable text."""
    # Remove code/panel blocks
    text = _JIRA_MARKUP_RE.sub("", text)
    # Convert headings to plain text
    text = _HEADING_RE.sub("", text)
    # Strip table markup
    text = _TABLE_ROW_RE.sub(" ", text)
    # Strip bold/italic markers, keep content
    text = _BOLD_ITALIC_RE.sub(r"\1", text)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_description_chunks(
    desc_class: str,
    desc_data: Optional[dict],
    metadata: dict,
    has_attachment: bool,
) -> list[dict]:
    """
    Build description chunks following the decision matrix from Section 5.3.

    Decision matrix:
      Attachment + Rich/Usable  → Supplementary chunk at 0.7x weight
      Attachment + Thin/Junk    → Skip
      No attachment + Rich      → Split into paragraph chunks
      No attachment + Usable    → Single chunk with title prepended
      No attachment + Thin      → Single merged chunk (metadata scaffold)
      No attachment + Junk/Empty→ Nothing (Tier D)
    """
    if desc_class in ("empty", "junk"):
        return []

    if desc_data is None:
        return []

    text: str = desc_data["text"]
    title: str = metadata.get("summary", "")

    # --- Case 1: Attachment exists ---
    if has_attachment:
        if desc_class in ("rich", "usable"):
            return [_make_desc_chunk(text, weight=0.7, idx=0)]
        # thin or junk with attachment → skip
        return []

    # --- Case 2: No attachment ---
    if desc_class == "rich":
        return _split_into_paragraphs(text, title)

    if desc_class == "usable":
        combined = f"{title}\n\n{text}" if title else text
        return [_make_desc_chunk(combined, weight=1.0, idx=0)]

    if desc_class == "thin":
        # Merge with metadata fields for a thin-but-usable chunk
        meta_context = _build_meta_scaffold(metadata)
        combined = f"{title}\n\n{text}\n\n{meta_context}" if meta_context else f"{title}\n\n{text}"
        return [_make_desc_chunk(combined, weight=0.8, idx=0)]

    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_desc_chunk(text: str, weight: float, idx: int) -> dict:
    return {
        "chunk_id": f"description-{idx}",
        "source": "description",
        "text": text,
        "word_count": len(text.split()),
        "is_boilerplate": False,
        "weight_multiplier": weight,
        "extraction_confidence": 0.90,
        "extraction_method": "description",
    }


def _split_into_paragraphs(text: str, title: str) -> list[dict]:
    """Split a rich description into paragraph-level chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip().split()) >= 10]
    if not paragraphs:
        combined = f"{title}\n\n{text}" if title else text
        return [_make_desc_chunk(combined, weight=1.0, idx=0)]

    chunks: list[dict] = []
    for i, para in enumerate(paragraphs):
        full = f"{title}\n\n{para}" if (i == 0 and title) else para
        chunks.append(_make_desc_chunk(full, weight=1.0, idx=i))

    return chunks


def _build_meta_scaffold(metadata: dict) -> str:
    """Build a short context string from key metadata fields."""
    parts: list[str] = []
    if metadata.get("components"):
        parts.append(f"Components: {', '.join(metadata['components'])}")
    if metadata.get("labels"):
        parts.append(f"Labels: {', '.join(metadata['labels'])}")
    if metadata.get("business_unit"):
        parts.append(f"Business Unit: {metadata['business_unit']}")
    return ". ".join(parts)
