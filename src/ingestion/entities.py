"""
Entity extraction for graph readiness — Section 9 of the architecture spec.

Uses dictionary-based matching (not NER) against product/system/capability
registries loaded from JSON files or synced from the graph DB.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

ENTITY_TYPES = ("products", "systems", "capabilities", "platforms", "business_units")

# ---------------------------------------------------------------------------
# Dictionary loading
# ---------------------------------------------------------------------------

def load_entity_dictionaries(dict_path: Optional[str] = None) -> dict[str, dict[str, str]]:
    """
    Load entity dictionaries from JSON files.

    Each file (products.json, systems.json, etc.) has the format:
        [{"canonical": "Name", "aliases": ["alias1", "alias2"]}, ...]

    Returns a dict keyed by entity type, where each value is a mapping
    from lowercase alias/canonical → canonical name.
    """
    if dict_path is None:
        dict_path = os.environ.get("ENTITY_DICT_PATH", "data/entity_dicts/")

    base = Path(dict_path)
    lookups: dict[str, dict[str, str]] = {t: {} for t in ENTITY_TYPES}

    for etype in ENTITY_TYPES:
        filepath = base / f"{etype}.json"
        if not filepath.exists():
            logger.debug("Entity dictionary not found: %s", filepath)
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                entries = json.load(fh)
            for entry in entries:
                canonical = entry.get("canonical", "")
                if not canonical:
                    continue
                lookups[etype][canonical.lower()] = canonical
                for alias in entry.get("aliases", []):
                    if alias:
                        lookups[etype][alias.lower()] = canonical
        except Exception as exc:
            logger.warning("Failed to load %s: %s", filepath, exc)

    total = sum(len(v) for v in lookups.values())
    logger.debug("Loaded %d entity dictionary entries across %d types", total, len(ENTITY_TYPES))
    return lookups


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_entities(
    chunks: list[dict],
    metadata: dict,
    dictionaries: dict[str, dict[str, str]],
) -> dict[str, list[dict]]:
    """
    Extract entity mentions from all chunks and metadata fields.

    Args:
        chunks:       All content chunks (slides, sections, description, etc.)
        metadata:     Metadata dict from extract_metadata()
        dictionaries: Lookup tables from load_entity_dictionaries()

    Returns:
        Dict keyed by entity type, each a list of mention dicts:
        {"term": str, "source": str, "chunk_id": str, "confidence": float}
    """
    mentions: dict[str, list[dict]] = {t: [] for t in ENTITY_TYPES}

    if not any(dictionaries.values()):
        logger.debug("Entity dictionaries are empty — skipping extraction")
        return mentions

    # --- Structured field: Jira components → direct product/system hints ---
    for comp in metadata.get("components", []):
        comp_lower = comp.lower()
        for etype in ("products", "systems"):
            canonical = dictionaries[etype].get(comp_lower)
            if canonical:
                mentions[etype].append(
                    {
                        "term": canonical,
                        "source": "jira_component",
                        "chunk_id": "metadata",
                        "confidence": 1.0,
                    }
                )

    # --- Structured field: product-linked issues ---
    for link in (metadata.get("classified_links") or {}).get("product", []):
        summary = link.get("summary", "")
        if summary:
            mentions["products"].append(
                {
                    "term": summary,
                    "source": "jira_link",
                    "chunk_id": "metadata",
                    "confidence": 0.9,
                }
            )

    # --- Build combined text sources ---
    text_sources: list[dict] = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if text:
            text_sources.append(
                {
                    "text": text,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "source": chunk.get("source", ""),
                }
            )

    for field in ("summary", "metadata_text"):
        val = metadata.get(field, "")
        if val:
            text_sources.append(
                {"text": val, "chunk_id": "metadata", "source": "metadata"}
            )

    # --- Dictionary scan across all text sources ---
    for text_source in text_sources:
        text_lower = text_source["text"].lower()
        for etype, lookup in dictionaries.items():
            for term_lower, canonical in lookup.items():
                if term_lower in text_lower:
                    # Use word-boundary check to reduce false positives
                    pattern = r"\b" + re.escape(term_lower) + r"\b"
                    if re.search(pattern, text_lower):
                        mentions[etype].append(
                            {
                                "term": canonical,
                                "source": text_source["source"],
                                "chunk_id": text_source["chunk_id"],
                                "confidence": 0.8,
                            }
                        )

    # --- Deduplicate: keep highest-confidence mention per (type, term) ---
    for etype in mentions:
        seen: dict[str, dict] = {}
        for m in mentions[etype]:
            key = m["term"]
            if key not in seen or m["confidence"] > seen[key]["confidence"]:
                seen[key] = m
        mentions[etype] = list(seen.values())

    return mentions


# ---------------------------------------------------------------------------
# Dictionary sync utilities
# ---------------------------------------------------------------------------

def sync_entity_dictionaries(
    graph_db: Any,
    vs_index: Any,
    dict_path: Optional[str] = None,
) -> None:
    """
    Pull the latest entities from the graph DB and VS index and write to disk.
    Intended to run on a scheduled basis (daily for products/systems).

    Args:
        graph_db:   Graph DB client with a .query() method.
        vs_index:   VS index client with a .query() method.
        dict_path:  Output directory for JSON files.
    """
    if dict_path is None:
        dict_path = os.environ.get("ENTITY_DICT_PATH", "data/entity_dicts/")

    base = Path(dict_path)
    base.mkdir(parents=True, exist_ok=True)

    try:
        products = graph_db.query("MATCH (p:Product) RETURN p.name, p.aliases")
        _save_dictionary(base / "products.json", products)

        systems = graph_db.query(
            "MATCH (p:Product) WHERE p.type IN ['service','system'] RETURN p.name, p.aliases"
        )
        _save_dictionary(base / "systems.json", systems)
    except Exception as exc:
        logger.error("Failed to sync product/system dictionaries: %s", exc)

    try:
        capabilities_raw = vs_index.query(
            "SELECT stage_name, stage_description FROM stages"
        )
        capabilities = [
            {
                "canonical": row["stage_name"],
                "aliases": _extract_key_phrases(row.get("stage_description", "")),
            }
            for row in capabilities_raw
        ]
        _save_dictionary(base / "capabilities.json", capabilities)
    except Exception as exc:
        logger.error("Failed to sync capability dictionary: %s", exc)

    logger.info("Entity dictionary sync complete.")


def _save_dictionary(filepath: Path, entries: list[dict]) -> None:
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
    logger.info("Saved %d entries to %s", len(entries), filepath)


def _extract_key_phrases(description: str) -> list[str]:
    """Simple key-phrase extractor for VS stage descriptions."""
    if not description:
        return []
    # Remove common stop words and return meaningful noun phrases
    stop = {
        "the", "a", "an", "in", "of", "for", "and", "or", "to", "is", "are",
        "that", "this", "with", "on", "at", "by", "from",
    }
    words = re.findall(r"[a-z]+", description.lower())
    content_words = [w for w in words if w not in stop and len(w) > 3]
    # Return top 5 distinctive words as aliases
    return list(dict.fromkeys(content_words))[:5]


# ---------------------------------------------------------------------------
# Seed empty dictionaries if none exist
# ---------------------------------------------------------------------------

def ensure_default_dictionaries(dict_path: Optional[str] = None) -> None:
    """
    Create empty placeholder dictionary files if none exist yet.
    This prevents errors on first run before graph DB sync.
    """
    if dict_path is None:
        dict_path = os.environ.get("ENTITY_DICT_PATH", "data/entity_dicts/")

    base = Path(dict_path)
    base.mkdir(parents=True, exist_ok=True)

    for etype in ENTITY_TYPES:
        filepath = base / f"{etype}.json"
        if not filepath.exists():
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump([], fh)
            logger.debug("Created empty dictionary: %s", filepath)
