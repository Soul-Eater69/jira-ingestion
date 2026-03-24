"""
Indexing design — Section 13 of the architecture spec.

Three retrieval indexes + one supervision store:
  - Coarse index (deck-level, one entry per ticket)
  - Fine index (chunk-level)
  - Metadata index (BM25 keyword search)
  - Supervision store (ground-truth labels — isolated from retrieval)

Default backend: LangGraph InMemoryStore with cosine-similarity search.
Swap to a persistent store (Chroma, FAISS, Azure AI Search, etc.) by
passing a LangChain-compatible vector store to LangChainVectorIndex.
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
# LangGraph InMemoryStore implementation (default)
# ---------------------------------------------------------------------------

class LangGraphVectorIndex(BaseVectorIndex):
    """
    Vector index backed by LangGraph's InMemoryStore.

    Documents are stored via the LangGraph store interface (namespace / key / value).
    Pre-computed embedding vectors are kept separately for cosine-similarity search.

    For production, replace with LangChainVectorIndex backed by Chroma, FAISS,
    Azure AI Search, or any other LangChain-compatible vector store.
    """

    def __init__(self, namespace: str) -> None:
        try:
            from langgraph.store.memory import InMemoryStore  # type: ignore
            self._store = InMemoryStore()
        except ImportError as exc:
            raise ImportError(
                "langgraph is required: pip install 'langgraph>=0.2.0'"
            ) from exc

        self._namespace = (namespace,)
        self._vectors: dict[str, list[float]] = {}   # id → embedding vector

    def upsert(self, id: str, vector: list[float], metadata: dict, text: str) -> None:
        self._store.put(
            self._namespace,
            id,
            {"text": text, "metadata": metadata},
        )
        if vector:
            self._vectors[id] = vector

    def get(self, id: str) -> Optional[dict]:
        item = self._store.get(self._namespace, id)
        if item is None:
            return None
        return {
            "id": id,
            "values": self._vectors.get(id, []),
            **item.value,
        }

    def search(self, query_vector: list[float], top_k: int = 10) -> list[dict]:
        """
        Cosine-similarity search over stored vectors.
        Returns top_k results sorted by descending similarity score.
        """
        if not self._vectors:
            return []

        try:
            import numpy as np
        except ImportError:
            logger.warning("numpy not available — similarity search disabled")
            return []

        q = np.array(query_vector, dtype=float)
        q_norm = q / (np.linalg.norm(q) + 1e-9)

        scores: list[tuple[float, str]] = []
        for doc_id, vec in self._vectors.items():
            v = np.array(vec, dtype=float)
            v_norm = v / (np.linalg.norm(v) + 1e-9)
            scores.append((float(np.dot(q_norm, v_norm)), doc_id))

        scores.sort(reverse=True)
        results = []
        for score, doc_id in scores[:top_k]:
            doc = self.get(doc_id)
            if doc:
                results.append({**doc, "score": score})
        return results

    def __len__(self) -> int:
        return len(self._vectors)


# ---------------------------------------------------------------------------
# LangChain vector store adapter (swap-in for any persistent backend)
# ---------------------------------------------------------------------------

class LangChainVectorIndex(BaseVectorIndex):
    """
    Wraps any LangChain VectorStore as a pipeline index.

    Accepts pre-computed embedding vectors via add_embeddings so the
    pipeline's own embedding step is used rather than the store's.

    Example — Chroma:
        from langchain_chroma import Chroma
        from langchain_core.embeddings import FakeEmbeddings
        store = Chroma(
            collection_name="tickets_coarse",
            embedding_function=FakeEmbeddings(size=3072),
            persist_directory="./chroma_db",
        )
        coarse_index = LangChainVectorIndex(store)

    Example — FAISS:
        from langchain_community.vectorstores import FAISS
        from langchain_core.embeddings import FakeEmbeddings
        store = FAISS.from_texts([], FakeEmbeddings(size=3072))
        coarse_index = LangChainVectorIndex(store)
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._data: dict[str, dict] = {}   # id → full record (for get())

    def upsert(self, id: str, vector: list[float], metadata: dict, text: str) -> None:
        try:
            self._store.add_embeddings(
                texts=[text],
                embeddings=[vector],
                metadatas=[{**metadata, "_id": id}],
                ids=[id],
            )
        except Exception as exc:
            logger.error("LangChain vector store upsert failed for %s: %s", id, exc)
        self._data[id] = {"id": id, "values": vector, "metadata": metadata, "text": text}

    def get(self, id: str) -> Optional[dict]:
        return self._data.get(id)


# ---------------------------------------------------------------------------
# In-memory fallback (no external deps — for unit tests)
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
        "updated_at": obs.get("updated_at", ""),   # used by idempotency check
        "ingested_at": document.get("ingested_at", ""),
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
            continue
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
    Index ground-truth labels in the supervision store.
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
# Index factory
# ---------------------------------------------------------------------------

def create_indexes(
    backend: str = "langgraph",
    langchain_stores: Optional[dict[str, Any]] = None,
) -> tuple[BaseVectorIndex, BaseVectorIndex, BaseMetadataIndex, BaseSupervisionStore]:
    """
    Factory that returns (coarse, fine, metadata, supervision) indexes.

    Args:
        backend: 'langgraph' (default) | 'langchain' | 'memory'
        langchain_stores: Required when backend='langchain'.
            {"coarse": <VectorStore>, "fine": <VectorStore>}

    Examples:
        # LangGraph in-memory (default, good for dev)
        coarse, fine, meta, sup = create_indexes()

        # LangChain Chroma (persistent)
        from langchain_chroma import Chroma
        from langchain_core.embeddings import FakeEmbeddings
        stores = {
            "coarse": Chroma("tickets_coarse", FakeEmbeddings(size=3072), persist_directory="./db"),
            "fine":   Chroma("tickets_fine",   FakeEmbeddings(size=3072), persist_directory="./db"),
        }
        coarse, fine, meta, sup = create_indexes(backend="langchain", langchain_stores=stores)
    """
    if backend == "langgraph":
        coarse = LangGraphVectorIndex("tickets_coarse")
        fine   = LangGraphVectorIndex("tickets_fine")

    elif backend == "langchain":
        if not langchain_stores:
            raise ValueError("langchain_stores must be provided when backend='langchain'")
        coarse = LangChainVectorIndex(langchain_stores["coarse"])
        fine   = LangChainVectorIndex(langchain_stores["fine"])

    else:  # "memory" — unit tests / CI
        coarse = InMemoryVectorIndex()
        fine   = InMemoryVectorIndex()

    metadata   = InMemoryMetadataIndex()
    supervision = InMemorySupervisionStore()
    return coarse, fine, metadata, supervision
