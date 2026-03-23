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

from jira_ingestion.config import JiraIngestionConfig
from jira_ingestion.clients.jira.value_stream_client import JiraValueStreamClient
from jira_ingestion.ingestion.pipeline import assemble_document, ingest_ticket
from jira_ingestion.ingestion.indexing import (
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
