"""
Indexing design — Section 13 of the architecture spec.

Implements three retrieval indexes + one supervision store:
  - Coarse index (deck-level, one entry per ticket)
  - Fine index (chunk-level)
  - Metadata index (BM25 keyword search)
  - Supervision store (ground-truth labels — isolated from retrieval)

The VectorIndex and SupervisionStore classes are thin abstractions.
The default implementation writes to Pinecone. Swap the backend by
subclassing BaseVectorIndex / BaseSupervisonStore.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class BaseVectorIndex(ABC):
    @abstractmethod
    def upsert(self, id: str, vector: list[float], metadata: dict, text: str) -> None: ...

    @abstractmethod
    def get(self, id: str) -> Optional[dict]: ...


class BaseSupervisionStore(ABC):
    @abstractmethod
    def upsert(self, id: str, data: dict) -> None: ...

    @abstractmethod
    def get(self, id: str) -> Optional[dict]: ...


class BaseMetadataIndex(ABC):
    @abstractmethod
    def upsert(self, id: str, fields: dict) -> None: ...

    @abstractmethod
    def get(self, id: str) -> Optional[dict]: ...


# ---------------------------------------------------------------------------
# Pinecone implementation
# ---------------------------------------------------------------------------

class PineconeVectorIndex(BaseVectorIndex):
    """Wraps a Pinecone index for coarse or fine vector search."""

    def __init__(self, index_name: str) -> None:
        self._index = self._init(index_name)

    def _init(self, index_name: str) -> Any:
        try:
            from pinecone import Pinecone  # type: ignore
            from src.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT

            pc = Pinecone(api_key=PINECONE_API_KEY)
            return pc.Index(index_name)
        except Exception as exc:
            logger.warning("Pinecone init failed (%s) — using in-memory fallback", exc)
            return None

    def upsert(self, id: str, vector: list[float], metadata: dict, text: str) -> None:
        if self._index is None:
            return
        meta = {**metadata, "_text": text[:4000]}  # Pinecone metadata value limit
        try:
            self._index.upsert(vectors=[{"id": id, "values": vector, "metadata": meta}])
        except Exception as exc:
            logger.error("Pinecone upsert failed for %s: %s", id, exc)

    def get(self, id: str) -> Optional[dict]:
        if self._index is None:
            return None
        try:
            result = self._index.fetch(ids=[id])
            vectors = result.get("vectors", {})
            return vectors.get(id)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# In-memory implementation (for testing / local runs)
# ---------------------------------------------------------------------------

class InMemoryVectorIndex(BaseVectorIndex):
    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def upsert(self, id: str, vector: list[float], metadata: dict, text: str) -> None:
        self._store[id] = {"id": id, "values": vector, "metadata": metadata, "text": text}

    def get(self, id: str) -> Optional[dict]:
        return self._store.get(id)

    def __len__(self) -> int:
        return len(self._store)


class InMemorySupervisionStore(BaseSupervisionStore):
    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def upsert(self, id: str, data: dict) -> None:
        self._store[id] = data

    def get(self, id: str) -> Optional[dict]:
        return self._store.get(id)


class InMemoryMetadataIndex(BaseMetadataIndex):
    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def upsert(self, id: str, fields: dict) -> None:
        self._store[id] = fields

    def get(self, id: str) -> Optional[dict]:
        return self._store.get(id)


# ---------------------------------------------------------------------------
# Indexing functions (Section 13.4)
# ---------------------------------------------------------------------------

def index_retrieval_view(
    document: dict,
    coarse_index: BaseVectorIndex,
    fine_index: BaseVectorIndex,
    metadata_index: BaseMetadataIndex,
) -> None:
    """
    Index the observed (retrieval) view of the document.
    NO label fields are stored in any retrieval index.
    """
    ticket_key = document["ticket_key"]
    obs = document["observed"]

    # --- Coarse index (deck-level) ---
    coarse_meta = {
        "quality_tier": obs["quality_tier"],
        "content_source": obs["content_source"],
        "created": obs.get("created", ""),
        "business_unit": obs["metadata"].get("business_unit") or "",
        "chunk_count": obs["stats"]["chunk_count"],
        "entity_product_count": len(obs["entity_mentions"].get("products", [])),
        "entity_capability_count": len(obs["entity_mentions"].get("capabilities", [])),
    }
    coarse_index.upsert(
        id=ticket_key,
        vector=obs["summary_embedding"],
        metadata=coarse_meta,
        text=f"{obs['summary_text']} {obs['metadata_text']}",
    )

    # --- Fine index (chunk-level) ---
    all_chunks = obs["chunks"] + obs.get("section_chunks", [])
    for chunk in all_chunks:
        embedding = chunk.get("embedding")
        if not embedding:
            continue  # chunk was not embedded (e.g., boilerplate)
        fine_meta = {
            "source": chunk.get("source", ""),
            "weight_multiplier": chunk.get("weight_multiplier", 1.0),
            "extraction_confidence": chunk.get("extraction_confidence", 1.0),
            "slide_num": chunk.get("slide_num") or chunk.get("page_num"),
            "has_table": chunk.get("has_table", False),
            "parent_ticket": ticket_key,
            "parent_quality_tier": obs["quality_tier"],
            # NO vs_labels here — spec Section 13.2
        }
        fine_index.upsert(
            id=f"{ticket_key}/{chunk['chunk_id']}",
            vector=embedding,
            metadata=fine_meta,
            text=chunk.get("text", ""),
        )

    # --- Metadata index (BM25) ---
    entity_terms: list[str] = []
    for etype, mentions in obs["entity_mentions"].items():
        entity_terms.extend(m["term"] for m in mentions)

    metadata_index.upsert(
        id=ticket_key,
        fields={
            "metadata_text": obs["metadata_text"],
            "components": obs["metadata"].get("components", []),
            "labels": obs["metadata"].get("labels", []),
            "business_unit": obs["metadata"].get("business_unit", ""),
            "summary_text": obs["summary_text"],
            "entity_terms": entity_terms,
        },
    )


def index_supervision_view(
    document: dict,
    supervision_store: BaseSupervisionStore,
) -> None:
    """
    Index ground-truth labels and trainability flags in the supervision store.
    This store is NEVER accessed during retrieval.
    """
    ticket_key = document["ticket_key"]
    sup = document["supervision"]

    supervision_store.upsert(
        id=ticket_key,
        data={
            "vs_labels": sup["vs_labels"],
            "vs_label_source": sup["vs_label_source"],
            "product_stage_labels": sup.get("product_stage_labels", []),
            "impacted_products": sup.get("impacted_products", []),
            "impacted_products_source": sup.get("impacted_products_source"),
            "trainability": sup["trainability"],
        },
    )


# ---------------------------------------------------------------------------
# Index factory — create the right backend based on config
# ---------------------------------------------------------------------------

def create_indexes(use_pinecone: bool = False) -> tuple[
    BaseVectorIndex, BaseVectorIndex, BaseMetadataIndex, BaseSupervisionStore
]:
    """
    Factory function that returns (coarse, fine, metadata, supervision) indexes.

    Set use_pinecone=True when PINECONE_API_KEY is configured.
    Falls back to in-memory for local development / testing.
    """
    if use_pinecone:
        from src.config import COARSE_INDEX_NAME, FINE_INDEX_NAME
        coarse = PineconeVectorIndex(COARSE_INDEX_NAME)
        fine = PineconeVectorIndex(FINE_INDEX_NAME)
    else:
        coarse = InMemoryVectorIndex()
        fine = InMemoryVectorIndex()

    metadata = InMemoryMetadataIndex()
    supervision = InMemorySupervisionStore()
    return coarse, fine, metadata, supervision
