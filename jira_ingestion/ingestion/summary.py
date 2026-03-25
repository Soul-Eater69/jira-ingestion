"""
Summary generation — Section 10 of the architecture spec.

Tier A/B: LLM summarises the top 8 chunks.
Tier C:   Description text as summary.
Tier D:   Concatenated metadata.

Also provides generate_derived_fields() which produces focused LLM-derived
summaries and keyword lists (problem, solution, capabilities, workflow terms,
business entities, product mentions).  These augment raw text — never replace.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

SUMMARY_INSTRUCTION = (
    "Summarize this idea card. Focus on: what problem it solves, "
    "what is proposed, which business areas are affected, "
    "and any technical systems or products mentioned. "
    "Be concise — 300 to 500 words."
)

_PROBLEM_PROMPT = (
    "In 2-3 sentences, describe the core problem or challenge this idea addresses. "
    "Be specific and factual."
)
_SOLUTION_PROMPT = (
    "In 2-3 sentences, describe the proposed solution or approach. "
    "Be specific and factual."
)
_CAPABILITY_KEYWORDS_PROMPT = (
    "List the key capability or feature keywords from this idea card. "
    "Return only a comma-separated list of 5 to 10 terms. No explanations."
)
_WORKFLOW_TERMS_PROMPT = (
    "List the workflow, process, or operational terms mentioned in this idea card. "
    "Return only a comma-separated list of up to 8 terms. No explanations."
)
_BUSINESS_ENTITIES_PROMPT = (
    "List the business entities (teams, departments, organisations, business units) "
    "mentioned in this idea card. "
    "Return only a comma-separated list. No explanations."
)
_PRODUCT_MENTIONS_PROMPT = (
    "List all product, system, platform, or technology names mentioned. "
    "Return only a comma-separated list. No explanations."
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


def generate_derived_fields(
    chunks: list[dict],
    metadata: dict,
    llm_client: Optional[Any],
    model: Optional[str] = None,
) -> dict:
    """
    Generate LLM-derived focused summaries and keyword lists.

    These are stored in the ``derived`` layer of the observed document.
    They augment raw and cleaned text — they never replace it.

    Args:
        chunks:      All content chunks (boilerplate excluded internally).
        metadata:    Metadata dict (used for title/summary context if chunks sparse).
        llm_client:  OpenAI-compatible client. Returns empty dict if None.
        model:       LLM model name override.

    Returns:
        Dict with keys matching DerivedFields TypedDict.
    """
    if not llm_client:
        return {}

    model = model or "gpt-4o-mini"

    # Build input text from top 6 non-boilerplate chunks (up to 4 000 chars)
    top_chunks = sorted(
        [c for c in chunks if not c.get("is_boilerplate")],
        key=lambda c: c.get("word_count", len(c.get("text", "").split())),
        reverse=True,
    )[:6]

    if not top_chunks:
        # Fall back to metadata text if no content chunks available
        input_text = metadata.get("metadata_text", "")
    else:
        input_text = "\n---\n".join(c["text"] for c in top_chunks)

    input_text = input_text[:4000]

    if not input_text.strip():
        return {}

    def _ask(prompt: str, max_tokens: int = 300) -> str:
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"{prompt}\n\n{input_text}"}],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("LLM derived field call failed: %s", exc)
            return ""

    def _ask_list(prompt: str) -> List[str]:
        raw = _ask(prompt, max_tokens=150)
        if not raw:
            return []
        return [t.strip() for t in raw.split(",") if t.strip()]

    return {
        "ticket_summary_llm": _ask(SUMMARY_INSTRUCTION, max_tokens=700),
        "problem_summary_llm": _ask(_PROBLEM_PROMPT, max_tokens=200),
        "solution_summary_llm": _ask(_SOLUTION_PROMPT, max_tokens=200),
        "capability_keywords_llm": _ask_list(_CAPABILITY_KEYWORDS_PROMPT),
        "workflow_terms_llm": _ask_list(_WORKFLOW_TERMS_PROMPT),
        "business_entities_llm": _ask_list(_BUSINESS_ENTITIES_PROMPT),
        "product_mentions_llm": _ask_list(_PRODUCT_MENTIONS_PROMPT),
    }


def _llm_summary(
    chunks: list[dict],
    llm_client: Any,
    model: Optional[str],
) -> str:
    """Use an LLM to summarise the top 8 content chunks."""
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

    model = model or "gpt-4o-mini"

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
