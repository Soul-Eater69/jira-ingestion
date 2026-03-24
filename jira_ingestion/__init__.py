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

Typed schemas (no Pydantic required):

    from jira_ingestion.models import (
        AttachmentMeta,
        TicketInput,
        ChunkRecord,
        PipelineDocument,
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
from jira_ingestion.models import (
    AttachmentMeta,
    TicketInput,
    ChunkRecord,
    ClassifiedLinks,
    TicketMetadata,
    ObservedDocument,
    SupervisionDocument,
    PipelineDocument,
)

__all__ = [
    # Core pipeline
    "JiraIngestionConfig",
    "JiraValueStreamClient",
    "assemble_document",
    "ingest_ticket",
    "create_indexes",
    # Index backends
    "LangGraphVectorIndex",
    "LangChainVectorIndex",
    "InMemoryVectorIndex",
    "InMemorySupervisionStore",
    "InMemoryMetadataIndex",
    "index_retrieval_view",
    "index_supervision_view",
    # Typed schemas
    "AttachmentMeta",
    "TicketInput",
    "ChunkRecord",
    "ClassifiedLinks",
    "TicketMetadata",
    "ObservedDocument",
    "SupervisionDocument",
    "PipelineDocument",
]
