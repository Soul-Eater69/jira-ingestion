"""
jira-ingestion — Jira ticket ingestion pipeline.

Public API:

    from jira_ingestion import (
        JiraIngestionConfig,
        JiraValueStreamClient,
        assemble_document,
        ingest_ticket,
        create_indexes,
    )
"""

from src.config import JiraIngestionConfig
from src.clients.jira.value_stream_client import JiraValueStreamClient
from src.ingestion.pipeline import assemble_document, ingest_ticket
from src.ingestion.indexing import (
    create_indexes,
    LangGraphVectorIndex,
    LangChainVectorIndex,
    InMemoryVectorIndex,
    InMemorySupervisionStore,
    InMemoryMetadataIndex,
    index_retrieval_view,
    index_supervision_view,
)

__all__ = [
    "JiraIngestionConfig",
    "JiraValueStreamClient",
    "assemble_document",
    "ingest_ticket",
    "create_indexes",
    "LangGraphVectorIndex",
    "LangChainVectorIndex",
    "InMemoryVectorIndex",
    "InMemorySupervisionStore",
    "InMemoryMetadataIndex",
    "index_retrieval_view",
    "index_supervision_view",
]
