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
    JsonBackedMetadataIndex,
    JsonBackedSupervisionStore,
    index_retrieval_view,
    index_supervision_view,
)
from jira_ingestion.dlq import DeadLetterQueue
from jira_ingestion.retries import RetryPolicy, retry_async, retry_sync
from jira_ingestion.telemetry import record_ingest, record_skip, record_failure, timed_stage
from jira_ingestion.models.validation import validate_ticket_input
from jira_ingestion.models import (
    AttachmentMeta,
    TicketInput,
    ChunkRecord,
    ClassifiedLinks,
    TicketMetadata,
    TriageArtifact,
    AttachmentInventoryItem,
    RetrievalViews,
    ProductLabels,
    CommentRecord,
    CommentsEnriched,
    ProvenanceRecord,
    RawLayer,
    ObservedDocument,
    SupervisionDocument,
    DerivedLayer,
    PipelineDocument,
    PreChunkDocument,
)
from jira_ingestion.ingestion.metadata import (
    extract_metadata,
    classify_links,
    extract_product_fields,
    extract_comments_enriched,
)
from jira_ingestion.ingestion.triage import build_triage_artifact

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
    "JsonBackedMetadataIndex",
    "JsonBackedSupervisionStore",
    "index_retrieval_view",
    "index_supervision_view",
    # Typed schemas
    "AttachmentMeta",
    "TicketInput",
    "ChunkRecord",
    "ClassifiedLinks",
    "TicketMetadata",
    "TriageArtifact",
    "AttachmentInventoryItem",
    "RetrievalViews",
    "ProductLabels",
    "CommentRecord",
    "CommentsEnriched",
    "ProvenanceRecord",
    "RawLayer",
    "ObservedDocument",
    "SupervisionDocument",
    "DerivedLayer",
    "PipelineDocument",
    "PreChunkDocument",
    # Metadata helpers
    "extract_metadata",
    "classify_links",
    "extract_product_fields",
    "extract_comments_enriched",
    "build_triage_artifact",
    # Platform modules
    "DeadLetterQueue",
    "RetryPolicy",
    "retry_async",
    "retry_sync",
    "record_ingest",
    "record_skip",
    "record_failure",
    "timed_stage",
    "validate_ticket_input",
]
