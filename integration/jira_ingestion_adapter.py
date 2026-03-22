"""
Integration adapter — wires IDP-VALUE-STREAMS clients into the jira-ingestion pipeline.

Copy this file to: IDP-VALUE-STREAMS/src/services/jira_ingestion_adapter.py

It replaces:
  - Pinecone   → Azure Search  (via your AzureSearchClient)
  - OpenAI raw → your EmbeddingClient / LLMClient wrappers
  - Supervision store → in-memory (swap for DB when needed)

Usage from ingestion_service.py:
    from src.services.jira_ingestion_adapter import run_ingestion

    doc = await run_ingestion(
        ticket_key="IDMT-4125",
        jira_client=self.jira_client,
        azure_search=self.azure_search,
        embedding_client=self.embedding_client,
        llm_client=self.llm_client,
    )
"""

from __future__ import annotations

from typing import Any, Optional

# ── IDP-VALUE-STREAMS clients (adjust import paths to match your project) ──
from src.clients.embedding import EmbeddingClient
from src.clients.llm import LLMClient
from src.clients.azure_search import AzureSearchClient

# ── jira-ingestion pipeline ────────────────────────────────────────────────
from src.ingestion.pipeline import ingest_ticket
from src.ingestion.indexing import (
    BaseVectorIndex,
    BaseMetadataIndex,
    BaseSupervisionStore,
)
from src.ingestion.storage import DocumentStore


# ---------------------------------------------------------------------------
# Azure Search adapters
# ---------------------------------------------------------------------------

class AzureCoarseIndex(BaseVectorIndex):
    """Deck-level (one doc per ticket) vector index backed by Azure Search."""

    def __init__(self, client: AzureSearchClient, index_name: str = "tickets-coarse"):
        self._client = client
        self._index = index_name

    def upsert(self, id: str, vector: list[float], metadata: dict, text: str) -> None:
        self._client.upload_documents(self._index, [{
            "id": id,
            "embedding": vector,
            "content": text,
            **metadata,
        }])

    def get(self, id: str) -> Optional[dict]:
        return self._client.get_document(self._index, id)


class AzureFineIndex(BaseVectorIndex):
    """Chunk-level vector index backed by Azure Search."""

    def __init__(self, client: AzureSearchClient, index_name: str = "tickets-fine"):
        self._client = client
        self._index = index_name

    def upsert(self, id: str, vector: list[float], metadata: dict, text: str) -> None:
        # Azure Search document IDs cannot contain '/'
        self._client.upload_documents(self._index, [{
            "id": id.replace("/", "_"),
            "embedding": vector,
            "content": text,
            **metadata,
        }])

    def get(self, id: str) -> Optional[dict]:
        return self._client.get_document(self._index, id.replace("/", "_"))


class AzureMetadataIndex(BaseMetadataIndex):
    """BM25 keyword metadata index backed by Azure Search."""

    def __init__(self, client: AzureSearchClient, index_name: str = "tickets-metadata"):
        self._client = client
        self._index = index_name

    def upsert(self, id: str, fields: dict) -> None:
        self._client.upload_documents(self._index, [{"id": id, **fields}])

    def get(self, id: str) -> Optional[dict]:
        return self._client.get_document(self._index, id)


class InMemorySupervisionStore(BaseSupervisionStore):
    """In-memory store for VS ground-truth labels. Swap for a DB when ready."""

    def __init__(self) -> None:
        self._store: dict = {}

    def upsert(self, id: str, data: dict) -> None:
        self._store[id] = data

    def get(self, id: str) -> Optional[dict]:
        return self._store.get(id)


# ---------------------------------------------------------------------------
# Embedding adapter
# Translates your EmbeddingClient into the OpenAI-compatible interface
# expected by src/ingestion/embedding.py
# ---------------------------------------------------------------------------

class EmbeddingAdapter:
    """Wraps your EmbeddingClient so the pipeline can call client.embeddings.create(...)"""

    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client
        self.embeddings = self  # pipeline accesses via .embeddings.create(...)

    def create(self, input: list[str], model: str) -> "_EmbeddingResponse":
        # Adjust 'embed_batch' to match your EmbeddingClient's method name
        vectors: list[list[float]] = self._client.embed_batch(input)
        return _EmbeddingResponse(vectors)


class _EmbeddingResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [_EmbeddingItem(i, v) for i, v in enumerate(vectors)]


class _EmbeddingItem:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


# ---------------------------------------------------------------------------
# LLM adapter
# Translates your LLMClient into the OpenAI-compatible interface
# expected by src/ingestion/summary.py
# ---------------------------------------------------------------------------

class LLMAdapter:
    """Wraps your LLMClient so the pipeline can call client.chat.completions.create(...)"""

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self.chat = _ChatNamespace(client)


class _ChatNamespace:
    def __init__(self, client: LLMClient) -> None:
        self.completions = _CompletionsNamespace(client)


class _CompletionsNamespace:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def create(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 700,
        temperature: float = 0.3,
    ) -> "_LLMResponse":
        prompt = messages[-1]["content"] if messages else ""
        # Adjust 'complete' to match your LLMClient's method name
        text: str = self._client.complete(prompt, max_tokens=max_tokens)
        return _LLMResponse(text)


class _LLMResponse:
    def __init__(self, text: str) -> None:
        self.choices = [_LLMChoice(text)]


class _LLMChoice:
    def __init__(self, text: str) -> None:
        self.message = _LLMMessage(text)


class _LLMMessage:
    def __init__(self, text: str) -> None:
        self.content = text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_ingestion(
    ticket_key: str,
    jira_client: Any,
    azure_search: AzureSearchClient,
    embedding_client: EmbeddingClient,
    llm_client: LLMClient,
    output_dir: str = "output/documents",
    storage_fmt: str = "json",
    dict_path: str = "jira-ingestion/data/entity_dicts/",
) -> dict:
    """
    Run the full jira-ingestion pipeline for one ticket using
    IDP-VALUE-STREAMS clients.

    Args:
        ticket_key:       e.g. "IDMT-4125"
        jira_client:      Authenticated JiraValueStreamClient instance.
        azure_search:     Your AzureSearchClient instance.
        embedding_client: Your EmbeddingClient instance.
        llm_client:       Your LLMClient instance.
        output_dir:       Directory to save the assembled document JSON.
        storage_fmt:      'json' | 'jsonl' | 'parquet'
        dict_path:        Path to entity dictionary JSON files.
    """
    return await ingest_ticket(
        ticket_key=ticket_key,
        jira_client=jira_client,
        coarse_index=AzureCoarseIndex(azure_search),
        fine_index=AzureFineIndex(azure_search),
        metadata_index=AzureMetadataIndex(azure_search),
        supervision_store=InMemorySupervisionStore(),
        llm_client=LLMAdapter(llm_client),
        embedding_client=EmbeddingAdapter(embedding_client),
        storage_dir=output_dir,
        storage_fmt=storage_fmt,
        dict_path=dict_path,
        trigger="service",
    )


# ---------------------------------------------------------------------------
# Local PPTX helper (for files already on disk in src/static/idea_cards/)
# ---------------------------------------------------------------------------

def ingest_local_pptx(
    file_path: str,
    ticket_key: str | None = None,
    output_dir: str = "output/documents",
) -> dict:
    """
    Assemble a document from a local PPTX without going through Jira.
    Useful for files already in src/static/idea_cards/.

    Example:
        doc = ingest_local_pptx("src/static/idea_cards/IDMT-19761.pptx")
        print(doc["observed"]["summary_text"])
    """
    from pathlib import Path
    from src.ingestion.pipeline import assemble_document
    from src.ingestion.storage import DocumentStore

    path = Path(file_path)
    file_bytes = path.read_bytes()
    key = ticket_key or path.stem  # e.g. "IDMT-19761"

    ticket_data = {
        "key": key,
        "fields": {
            "summary": key,
            "description": None,
            "issuelinks": [],
            "attachment": [],
            "comment": {},
            "components": [],
            "labels": [],
            "reporter": None,
            "created": None,
            "priority": None,
        },
        "themes": [],
        "attachments": [{
            "id": "local",
            "filename": path.name,
            "mimeType": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "size": len(file_bytes),
            "content": "",
            "file_bytes": file_bytes,   # pre-loaded — skips the download step
            "ext": path.suffix.lstrip(".").lower(),
            "triage_score": 100,
            "triage_reasons": ["local file — bypassed triage"],
            "confirmed": True,
        }],
    }

    doc = assemble_document(
        ticket_data=ticket_data,
        download_fn=lambda att: att.get("file_bytes", b""),
    )
    DocumentStore(output_dir).save(doc)
    return doc
