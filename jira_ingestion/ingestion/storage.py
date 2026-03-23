"""
Document persistence — save assembled documents to disk before (or after) indexing.

Supported formats:
    JSON       — one file per ticket, human-readable, optional embedding stripping
    JSONL      — one line per ticket, good for batch processing / streaming
    Parquet    — columnar, compact, good for bulk analysis (requires pyarrow)

Usage:
    from jira_ingestion.ingestion.storage import DocumentStore

    store = DocumentStore(output_dir="output/documents")
    store.save(document)                          # → output/documents/IDEA-1234.json
    store.save(document, fmt="jsonl")             # → output/documents/ingested.jsonl (appended)
    store.save(document, fmt="parquet")           # → output/documents/IDEA-1234.parquet

    doc = store.load("IDEA-1234")                 # reload a saved document
    docs = store.load_all()                       # load all JSON docs in the dir
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

logger = logging.getLogger(__name__)

Format = Literal["json", "jsonl", "parquet"]

_JSONL_FILENAME = "ingested.jsonl"


class DocumentStore:
    """
    Manages reading and writing assembled documents to a local directory.

    The store separates embeddings from the human-readable payload:
      - <ticket_key>.json         → full document with embeddings stripped (readable)
      - <ticket_key>.embeddings.json → raw embedding vectors (compact, optional)

    This keeps the human-readable files small while preserving the vectors
    for re-indexing without re-calling the embedding API.
    """

    def __init__(
        self,
        output_dir: str = "output/documents",
        save_embeddings: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.save_embeddings = save_embeddings
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        document: dict,
        fmt: Format = "json",
        include_embeddings: bool = False,
    ) -> Path:
        """
        Persist a document to disk.

        Args:
            document:           Assembled document dict from assemble_document().
            fmt:                Output format: 'json' | 'jsonl' | 'parquet'.
            include_embeddings: If True, embed vectors are kept inline in the
                                output file. If False (default), they are stored
                                in a separate <ticket_key>.embeddings.json file.

        Returns:
            Path to the written file.
        """
        ticket_key = document.get("ticket_key", "unknown")

        if fmt == "json":
            return self._save_json(document, ticket_key, include_embeddings)
        elif fmt == "jsonl":
            return self._save_jsonl(document, ticket_key, include_embeddings)
        elif fmt == "parquet":
            return self._save_parquet(document, ticket_key)
        else:
            raise ValueError(f"Unknown format: {fmt!r}. Choose 'json', 'jsonl', or 'parquet'.")

    def _save_json(
        self, document: dict, ticket_key: str, include_embeddings: bool
    ) -> Path:
        payload, embeddings = _split_embeddings(document)

        # Main document file
        out_path = self.output_dir / f"{ticket_key}.json"
        doc_to_write = document if include_embeddings else payload
        _write_json(out_path, doc_to_write)
        logger.info("Saved document → %s", out_path)

        # Separate embeddings file
        if self.save_embeddings and not include_embeddings and embeddings:
            emb_path = self.output_dir / f"{ticket_key}.embeddings.json"
            _write_json(emb_path, embeddings)
            logger.debug("Saved embeddings → %s", emb_path)

        return out_path

    def _save_jsonl(
        self, document: dict, ticket_key: str, include_embeddings: bool
    ) -> Path:
        payload, embeddings = _split_embeddings(document)
        doc_to_write = document if include_embeddings else payload

        out_path = self.output_dir / _JSONL_FILENAME
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(doc_to_write, ensure_ascii=False, default=str) + "\n")
        logger.info("Appended to JSONL → %s (%s)", out_path, ticket_key)

        if self.save_embeddings and not include_embeddings and embeddings:
            emb_path = self.output_dir / f"{ticket_key}.embeddings.json"
            _write_json(emb_path, embeddings)

        return out_path

    def _save_parquet(self, document: dict, ticket_key: str) -> Path:
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required for Parquet output: pip install pyarrow"
            ) from exc

        flat = _flatten_for_parquet(document)
        table = pa.Table.from_pydict({k: [v] for k, v in flat.items()})
        out_path = self.output_dir / f"{ticket_key}.parquet"
        pq.write_table(table, str(out_path), compression="snappy")
        logger.info("Saved Parquet → %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, ticket_key: str, with_embeddings: bool = False) -> Optional[dict]:
        """
        Load a previously saved document by ticket key.

        Args:
            ticket_key:      e.g. "IDEA-1234"
            with_embeddings: If True, merge the embeddings file back in.
        """
        path = self.output_dir / f"{ticket_key}.json"
        if not path.exists():
            logger.warning("Document not found: %s", path)
            return None

        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)

        if with_embeddings:
            emb_path = self.output_dir / f"{ticket_key}.embeddings.json"
            if emb_path.exists():
                with open(emb_path, "r", encoding="utf-8") as fh:
                    embeddings = json.load(fh)
                doc = _merge_embeddings(doc, embeddings)

        return doc

    def load_all(self, with_embeddings: bool = False) -> Iterator[dict]:
        """Yield all JSON documents in the output directory."""
        for path in sorted(self.output_dir.glob("*.json")):
            if path.name.endswith(".embeddings.json"):
                continue
            ticket_key = path.stem
            doc = self.load(ticket_key, with_embeddings=with_embeddings)
            if doc:
                yield doc

    def load_jsonl(self, with_embeddings: bool = False) -> Iterator[dict]:
        """Yield all documents from the JSONL file."""
        path = self.output_dir / _JSONL_FILENAME
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    if with_embeddings:
                        key = doc.get("ticket_key", "")
                        emb_path = self.output_dir / f"{key}.embeddings.json"
                        if emb_path.exists():
                            with open(emb_path, "r", encoding="utf-8") as ef:
                                doc = _merge_embeddings(doc, json.load(ef))
                    yield doc
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSONL line: %s", exc)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def list_keys(self) -> list[str]:
        """Return all ticket keys that have been saved."""
        return [
            p.stem
            for p in sorted(self.output_dir.glob("*.json"))
            if not p.name.endswith(".embeddings.json")
        ]

    def exists(self, ticket_key: str) -> bool:
        return (self.output_dir / f"{ticket_key}.json").exists()

    def summary(self) -> dict:
        """Return a quick stats summary of what's stored."""
        keys = self.list_keys()
        emb_files = list(self.output_dir.glob("*.embeddings.json"))
        jsonl_path = self.output_dir / _JSONL_FILENAME
        return {
            "output_dir": str(self.output_dir),
            "document_count": len(keys),
            "has_embeddings_files": len(emb_files),
            "jsonl_exists": jsonl_path.exists(),
            "ticket_keys": keys,
        }


# ---------------------------------------------------------------------------
# Embedding split / merge helpers
# ---------------------------------------------------------------------------

def _split_embeddings(document: dict) -> tuple[dict, dict]:
    """
    Separate embedding vectors from the document payload.

    Returns:
        (payload_without_embeddings, embeddings_dict)
    """
    embeddings: dict = {"ticket_key": document.get("ticket_key"), "vectors": {}}
    payload = _deep_copy_strip(document, embeddings["vectors"])
    return payload, embeddings


def _deep_copy_strip(obj: Any, vectors: dict, path: str = "") -> Any:
    """Recursively copy obj, replacing embedding lists with placeholder strings."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            current_path = f"{path}.{k}" if path else k
            if k.endswith("embedding") and isinstance(v, list):
                # Store vector under its path key, replace with placeholder
                vectors[current_path] = v
                result[k] = f"<vector dim={len(v)}>"
            else:
                result[k] = _deep_copy_strip(v, vectors, current_path)
        return result
    if isinstance(obj, list):
        return [_deep_copy_strip(item, vectors, f"{path}[{i}]") for i, item in enumerate(obj)]
    return obj


def _merge_embeddings(document: dict, embeddings: dict) -> dict:
    """Merge embedding vectors back into a stripped document."""
    vectors = embeddings.get("vectors", {})
    if not vectors:
        return document

    import copy
    doc = copy.deepcopy(document)

    for path, vector in vectors.items():
        _set_nested(doc, path.split("."), vector)

    return doc


def _set_nested(obj: Any, keys: list[str], value: Any) -> None:
    """Set a value in a nested dict using a key path."""
    for key in keys[:-1]:
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        else:
            return
    last = keys[-1]
    if isinstance(obj, dict):
        obj[last] = value


# ---------------------------------------------------------------------------
# Parquet flattening
# ---------------------------------------------------------------------------

def _flatten_for_parquet(document: dict) -> dict[str, Any]:
    """Flatten the document into a single-level dict suitable for a Parquet row."""
    obs = document.get("observed", {})
    sup = document.get("supervision", {})
    stats = obs.get("stats", {})
    meta = obs.get("metadata", {})
    trainability = sup.get("trainability", {})

    return {
        "ticket_key": document.get("ticket_key", ""),
        "schema_version": document.get("schema_version", ""),
        "ingested_at": document.get("ingested_at", ""),
        "quality_tier": obs.get("quality_tier", ""),
        "content_source": obs.get("content_source", ""),
        "created": obs.get("created", ""),
        "summary_text": obs.get("summary_text", ""),
        "metadata_text": obs.get("metadata_text", ""),
        "vs_labels": json.dumps(sup.get("vs_labels", []), ensure_ascii=False),
        "vs_label_source": sup.get("vs_label_source", ""),
        "chunk_count": stats.get("chunk_count", 0),
        "section_count": stats.get("section_count", 0),
        "total_word_count": stats.get("total_word_count", 0),
        "has_tables": stats.get("has_tables", False),
        "has_speaker_notes": stats.get("has_speaker_notes", False),
        "entity_mention_count": stats.get("entity_mention_count", 0),
        "business_unit": meta.get("business_unit", ""),
        "product_area": meta.get("product_area", ""),
        "labels": json.dumps(meta.get("labels", []), ensure_ascii=False),
        "components": json.dumps(meta.get("components", []), ensure_ascii=False),
        "reporter": meta.get("reporter", ""),
        "priority": meta.get("priority", ""),
        "has_gold_vs_labels": trainability.get("has_gold_vs_labels", False),
        "is_trainable_for_vs": trainability.get("is_trainable_for_vs", False),
        "source_quality_score": trainability.get("source_quality_score", 0.0),
        "triage_primary": (obs.get("triage") or {}).get("primary_attachment", ""),
        "triage_score": (obs.get("triage") or {}).get("triage_score"),
        "entity_products": json.dumps(
            [m["term"] for m in obs.get("entity_mentions", {}).get("products", [])],
            ensure_ascii=False,
        ),
        "entity_capabilities": json.dumps(
            [m["term"] for m in obs.get("entity_mentions", {}).get("capabilities", [])],
            ensure_ascii=False,
        ),
        "chunks_json": json.dumps(
            _strip_embeddings_from_list(obs.get("chunks", [])), ensure_ascii=False
        ),
        "section_chunks_json": json.dumps(
            _strip_embeddings_from_list(obs.get("section_chunks", [])), ensure_ascii=False
        ),
    }


def _strip_embeddings_from_list(chunks: list[dict]) -> list[dict]:
    return [{k: v for k, v in c.items() if not k.endswith("embedding")} for c in chunks]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
