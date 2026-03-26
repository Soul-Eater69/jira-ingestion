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

_DERIVED_INSTRUCTION = """\
You are analyzing a Jira idea card. Given the content below, return a JSON object with exactly these keys:

- ticket_summary_llm: A 2-3 sentence plain-language summary of the whole idea.
- problem_summary_llm: 1-2 sentences on the core problem or challenge being addressed.
- solution_summary_llm: 1-2 sentences on the proposed solution or capability.
- capability_keywords_llm: List of up to 8 short capability or feature keywords (e.g. "payment processing", "real-time alerts").
- workflow_terms_llm: List of up to 6 business process or workflow terms mentioned (e.g. "approval workflow", "onboarding").
- business_entities_llm: List of up to 8 organizational or domain entities (e.g. "Finance", "Risk Management", "Digital").
- product_mentions_llm: List of up to 8 product, system, or platform names mentioned (e.g. "SAP", "Salesforce", "API Gateway").

Return only valid JSON. No markdown fences. No extra text.
"""


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


def generate_derived_artifacts(
    quality_tier: str,
    chunks: list[dict],
    metadata: dict,
    llm_client: Optional[Any] = None,
    model: Optional[str] = None,
) -> dict:
    """
    Generate structured derived artifacts using an LLM.

    Populates the `derived` layer with:
        ticket_summary_llm, problem_summary_llm, solution_summary_llm,
        capability_keywords_llm, workflow_terms_llm, business_entities_llm,
        product_mentions_llm

    Falls back to empty strings / lists if LLM is unavailable or fails.

    Args:
        quality_tier:  'A' | 'B' | 'C' | 'D'
        chunks:        All content chunks.
        metadata:      Ticket metadata dict.
        llm_client:    OpenAI-compatible client. If None, returns empty stubs.
        model:         LLM model override.

    Returns:
        dict matching DerivedLayer TypedDict.
    """
    _empty: dict = {
        "ticket_summary_llm": "",
        "problem_summary_llm": "",
        "solution_summary_llm": "",
        "capability_keywords_llm": [],
        "workflow_terms_llm": [],
        "business_entities_llm": [],
        "product_mentions_llm": [],
    }

    if llm_client is None or quality_tier == "D":
        return _empty

    # Select top chunks by word count, skip boilerplate
    top_chunks = sorted(
        [c for c in chunks if not c.get("is_boilerplate")],
        key=lambda c: c.get("word_count", len(c.get("text", "").split())),
        reverse=True,
    )[:10]

    if not top_chunks:
        return _empty

    input_text = "\n---\n".join(c["text"] for c in top_chunks)
    # Add metadata context
    meta_context = metadata.get("metadata_text", "")
    if meta_context:
        input_text = f"Ticket metadata: {meta_context}\n\n---\n\n{input_text}"

    prompt = f"{_DERIVED_INSTRUCTION}\n\nContent:\n{input_text[:6000]}"
    model_name = model or "gpt-4o-mini"

    try:
        import json as _json
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if model added them anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = _json.loads(raw)
        return {
            "ticket_summary_llm": str(parsed.get("ticket_summary_llm", "")),
            "problem_summary_llm": str(parsed.get("problem_summary_llm", "")),
            "solution_summary_llm": str(parsed.get("solution_summary_llm", "")),
            "capability_keywords_llm": list(parsed.get("capability_keywords_llm") or []),
            "workflow_terms_llm": list(parsed.get("workflow_terms_llm") or []),
            "business_entities_llm": list(parsed.get("business_entities_llm") or []),
            "product_mentions_llm": list(parsed.get("product_mentions_llm") or []),
        }
    except Exception as exc:
        logger.warning("generate_derived_artifacts failed: %s — returning stubs", exc)
        return _empty


def _best_chunk(chunks: list[dict]) -> Optional[dict]:
    """Return the chunk with the most words."""
    content_chunks = [c for c in chunks if not c.get("is_boilerplate")]
    if not content_chunks:
        return None
    return max(content_chunks, key=lambda c: c.get("word_count", len(c.get("text", "").split())))


# Re-export Any for the type hint above
from typing import Any  # noqa: E402  (placed here to avoid circular import confusion)
