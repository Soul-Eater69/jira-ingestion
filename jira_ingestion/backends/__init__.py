"""
jira_ingestion.backends — Concrete retrieval and supervision store implementations.

This submodule exposes backend classes that are deliberately kept out of the
main package root (``jira_ingestion``). Normal callers should use
``create_indexes()`` and let the factory pick the right backend.

Direct imports are intended for:
  - Swapping to a specific backend (e.g. LangChain Chroma)
  - Unit tests that need an in-memory store
  - Custom wiring beyond what ``create_indexes()`` provides

Usage
-----
In-memory (unit tests / CI):

    from jira_ingestion.backends import (
        InMemoryVectorIndex,
        InMemoryMetadataIndex,
        InMemorySupervisionStore,
    )
    coarse = InMemoryVectorIndex()
    fine   = InMemoryVectorIndex()
    meta   = InMemoryMetadataIndex()
    sup    = InMemorySupervisionStore()

Restart-safe JSON stores:

    from jira_ingestion.backends import (
        JsonBackedMetadataIndex,
        JsonBackedSupervisionStore,
    )
    meta = JsonBackedMetadataIndex("output/metadata.json")
    sup  = JsonBackedSupervisionStore("output/supervision.json")

LangGraph (default production):

    from jira_ingestion.backends import LangGraphVectorIndex
    coarse = LangGraphVectorIndex("tickets_coarse")

LangChain Chroma (persistent vector store):

    from langchain_chroma import Chroma
    from langchain_core.embeddings import FakeEmbeddings
    from jira_ingestion.backends import LangChainVectorIndex

    store = Chroma(
        collection_name="tickets_coarse",
        embedding_function=FakeEmbeddings(size=3072),
        persist_directory="./chroma_db",
    )
    coarse = LangChainVectorIndex(store)

Abstract base classes (for custom implementations):

    from jira_ingestion.backends import (
        BaseVectorIndex,
        BaseMetadataIndex,
        BaseSupervisionStore,
    )
"""

from jira_ingestion.ingestion.indexing import (
    # Abstract interfaces
    BaseVectorIndex,
    BaseMetadataIndex,
    BaseSupervisionStore,
    # In-memory (unit tests)
    InMemoryVectorIndex,
    InMemoryMetadataIndex,
    InMemorySupervisionStore,
    # JSON-backed persistent stores
    JsonBackedMetadataIndex,
    JsonBackedSupervisionStore,
    # LangGraph (default production backend)
    LangGraphVectorIndex,
    # LangChain adapter (any LangChain-compatible vector store)
    LangChainVectorIndex,
)

__all__ = [
    # Abstract interfaces
    "BaseVectorIndex",
    "BaseMetadataIndex",
    "BaseSupervisionStore",
    # In-memory
    "InMemoryVectorIndex",
    "InMemoryMetadataIndex",
    "InMemorySupervisionStore",
    # JSON-backed
    "JsonBackedMetadataIndex",
    "JsonBackedSupervisionStore",
    # Production backends
    "LangGraphVectorIndex",
    "LangChainVectorIndex",
]
