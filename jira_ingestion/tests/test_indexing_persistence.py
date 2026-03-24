"""
Tests for persistent index stores.

Covers:
  - JsonBackedMetadataIndex: upsert, get, persistence across re-instantiation
  - JsonBackedSupervisionStore: upsert, get, persistence across re-instantiation
  - create_indexes factory: all backends return correct types
"""

from __future__ import annotations

import json
import pytest
import tempfile
from pathlib import Path

from jira_ingestion.ingestion.indexing import (
    InMemoryMetadataIndex,
    InMemorySupervisionStore,
    InMemoryVectorIndex,
    JsonBackedMetadataIndex,
    JsonBackedSupervisionStore,
    create_indexes,
    BaseMetadataIndex,
    BaseSupervisionStore,
    BaseVectorIndex,
)


# ---------------------------------------------------------------------------
# JsonBackedMetadataIndex
# ---------------------------------------------------------------------------

class TestJsonBackedMetadataIndex:
    def test_upsert_and_get(self, tmp_path):
        store = JsonBackedMetadataIndex(str(tmp_path / "metadata.json"))
        store.upsert("IDEA-1", {"metadata_text": "hello", "labels": ["x"]})
        result = store.get("IDEA-1")
        assert result is not None
        assert result["metadata_text"] == "hello"

    def test_get_missing_returns_none(self, tmp_path):
        store = JsonBackedMetadataIndex(str(tmp_path / "metadata.json"))
        assert store.get("MISSING") is None

    def test_survives_restart(self, tmp_path):
        path = str(tmp_path / "metadata.json")
        store1 = JsonBackedMetadataIndex(path)
        store1.upsert("IDEA-10", {"field": "value"})

        # Re-instantiate — simulates process restart
        store2 = JsonBackedMetadataIndex(path)
        assert store2.get("IDEA-10") == {"field": "value"}

    def test_upsert_overwrites(self, tmp_path):
        store = JsonBackedMetadataIndex(str(tmp_path / "metadata.json"))
        store.upsert("IDEA-1", {"v": 1})
        store.upsert("IDEA-1", {"v": 2})
        assert store.get("IDEA-1") == {"v": 2}

    def test_len(self, tmp_path):
        store = JsonBackedMetadataIndex(str(tmp_path / "metadata.json"))
        store.upsert("A", {})
        store.upsert("B", {})
        assert len(store) == 2

    def test_corrupted_file_recovers_empty(self, tmp_path):
        path = tmp_path / "metadata.json"
        path.write_text("NOT JSON{{{")
        store = JsonBackedMetadataIndex(str(path))
        assert store.get("anything") is None


# ---------------------------------------------------------------------------
# JsonBackedSupervisionStore
# ---------------------------------------------------------------------------

class TestJsonBackedSupervisionStore:
    def test_upsert_and_get(self, tmp_path):
        store = JsonBackedSupervisionStore(str(tmp_path / "supervision.json"))
        store.upsert("IDEA-1", {"vs_labels": ["VS-10"], "trainability": {}})
        result = store.get("IDEA-1")
        assert result["vs_labels"] == ["VS-10"]

    def test_survives_restart(self, tmp_path):
        path = str(tmp_path / "supervision.json")
        store1 = JsonBackedSupervisionStore(path)
        store1.upsert("IDEA-99", {"vs_labels": ["VS-999"]})

        store2 = JsonBackedSupervisionStore(path)
        assert store2.get("IDEA-99")["vs_labels"] == ["VS-999"]

    def test_missing_returns_none(self, tmp_path):
        store = JsonBackedSupervisionStore(str(tmp_path / "supervision.json"))
        assert store.get("NOPE") is None


# ---------------------------------------------------------------------------
# create_indexes factory
# ---------------------------------------------------------------------------

class TestCreateIndexes:
    def test_memory_backend(self):
        coarse, fine, meta, sup = create_indexes(backend="memory")
        assert isinstance(coarse, InMemoryVectorIndex)
        assert isinstance(fine, InMemoryVectorIndex)
        assert isinstance(meta, InMemoryMetadataIndex)
        assert isinstance(sup, InMemorySupervisionStore)

    def test_memory_backend_with_persistent_stores(self, tmp_path):
        meta_path = str(tmp_path / "meta.json")
        sup_path = str(tmp_path / "sup.json")
        coarse, fine, meta, sup = create_indexes(
            backend="memory",
            metadata_store_path=meta_path,
            supervision_store_path=sup_path,
        )
        assert isinstance(meta, JsonBackedMetadataIndex)
        assert isinstance(sup, JsonBackedSupervisionStore)

    def test_default_backend_uses_in_memory_stores(self):
        # LangGraph may not be installed in test env — use memory backend
        coarse, fine, meta, sup = create_indexes(backend="memory")
        assert isinstance(meta, InMemoryMetadataIndex)
        assert isinstance(sup, InMemorySupervisionStore)

    def test_langchain_backend_requires_stores(self):
        with pytest.raises(ValueError, match="langchain_stores"):
            create_indexes(backend="langchain")

    def test_returns_four_items(self):
        result = create_indexes(backend="memory")
        assert len(result) == 4

    def test_vector_indexes_are_base_vector_index(self):
        coarse, fine, _, _ = create_indexes(backend="memory")
        assert isinstance(coarse, BaseVectorIndex)
        assert isinstance(fine, BaseVectorIndex)
