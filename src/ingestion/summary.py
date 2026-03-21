"""
Summary generation — Section 10 of the architecture spec.

Tier A/B: LLM summarises the top 8 chunks.
Tier C:   Description text as summary.
Tier D:   Concatenated metadata.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

SUMMARY_INSTRUCTION = (
    "Summarize this idea card. Focus on: what problem it solves, "
    "what is proposed, which business areas are affected, "
    "and any technical systems or products mentioned. "
    "Be concise — 300 to 500 words."
)


def generate_summary(
    quality_tier: str,
    chunks: list[dict],
    metadata: dict,
    llm_client: Optional[Any] = None,
    model: Optional[str] = None,
) -> str:
    """
    Generate a summary string for the document.

    Args:
        quality_tier: 'A' | 'B' | 'C' | 'D'
        chunks:       All available content chunks.
        metadata:     Metadata dict with 'metadata_text', 'summary', etc.
        llm_client:   OpenAI-compatible client. If None, falls back to non-LLM.
        model:        LLM model name override.
    """
    if quality_tier in ("A", "B") and len(chunks) >= 3 and llm_client is not None:
        return _llm_summary(chunks, llm_client, model)

    # Fallback: longest chunk or metadata
    best_chunk = _best_chunk(chunks)
    if best_chunk and len(best_chunk["text"].split()) >= 30:
        return best_chunk["text"][:800]

    return metadata.get("metadata_text", metadata.get("summary", ""))


def _llm_summary(
    chunks: list[dict],
    llm_client: Any,
    model: Optional[str],
) -> str:
    """Use an LLM to summarise the top 8 content chunks."""
    from src.config import LLM_MODEL

    model = model or LLM_MODEL

    # Pick top 8 chunks by word count, excluding boilerplate
    top_chunks = sorted(
        [c for c in chunks if not c.get("is_boilerplate")],
        key=lambda c: c.get("word_count", len(c.get("text", "").split())),
        reverse=True,
    )[:8]

    if not top_chunks:
        return ""

    input_text = "\n---\n".join(c["text"] for c in top_chunks)
    prompt = f"{SUMMARY_INSTRUCTION}\n\n{input_text}"

    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("LLM summary failed: %s — falling back", exc)
        return _best_chunk(top_chunks)["text"][:800] if top_chunks else ""


def _best_chunk(chunks: list[dict]) -> Optional[dict]:
    """Return the chunk with the most words."""
    content_chunks = [c for c in chunks if not c.get("is_boilerplate")]
    if not content_chunks:
        return None
    return max(content_chunks, key=lambda c: c.get("word_count", len(c.get("text", "").split())))


# Re-export Any for the type hint above
from typing import Any  # noqa: E402  (placed here to avoid circular import confusion)
