"""
jira_ingestion — Jira ticket ingestion pipeline.

Public contract
===============

Framework layer (normal callers)
---------------------------------
    from jira_ingestion import (
        JiraIngestionConfig,   # all thresholds, field mapping, persistence toggles
        JiraValueStreamClient, # authenticated Jira access
        ingest_ticket,         # primary operational entry point
        assemble_document,     # pure assembly — no I/O side effects
        create_indexes,        # bootstrap retrieval + supervision stores
    )

Stable schema types
-------------------
    from jira_ingestion import (
        TicketInput,        # expected shape of get_ticket_data() response
        PipelineDocument,   # top-level output of assemble_document()
        PreChunkDocument,   # 04_assembled_prechunk.json artifact shape
        ChunkRecord,        # individual content chunk
        TriageArtifact,     # 03_triage_output.json artifact shape
    )

Advanced / backend layer
------------------------
    from jira_ingestion.backends import (
        InMemoryVectorIndex,       # unit tests
        LangGraphVectorIndex,      # default production backend
        LangChainVectorIndex,      # any LangChain-compatible store
        JsonBackedMetadataIndex,   # restart-safe JSON metadata store
        JsonBackedSupervisionStore,# restart-safe JSON supervision store
    )

Internal engineering utilities (maintainers)
--------------------------------------------
    from jira_ingestion.retries   import RetryPolicy, retry_async, retry_sync
    from jira_ingestion.telemetry import record_ingest, record_skip, timed_stage
    from jira_ingestion.dlq       import DeadLetterQueue
    from jira_ingestion.models.validation import validate_ticket_input
    from jira_ingestion.ingestion.metadata import (
        extract_metadata, classify_links, extract_product_fields,
        extract_stage_labels, extract_comments_enriched,
    )
    from jira_ingestion.ingestion.triage   import build_triage_artifact
    from jira_ingestion.ingestion.indexing import (
        index_retrieval_view, index_supervision_view,
    )
    from jira_ingestion.models import (
        AttachmentMeta, ClassifiedLinks, TicketMetadata,
        TriageScores, AttachmentInventoryItem, RetrievalViews,
        ProductLabels, CommentRecord, CommentsEnriched,
        ProvenanceRecord, RawLayer, ObservedDocument,
        SupervisionDocument, DerivedLayer,
    )

Debug artifacts contract
------------------------
When ``storage_dir`` is supplied to ``ingest_ticket``, the following files
are written automatically to ``<storage_dir>/_debug/<ticket_key>/``:

    01_raw_ticket.json          — raw Jira API payload
    02_attachment_contents.json — per-attachment inventory + triage scores
    03_triage_output.json       — primary/supplementary decision + multi-score
    04_assembled_prechunk.json  — canonical pre-chunk document for inspection
    05_debug_report.json        — compact summary of all pipeline decisions

These files are part of the framework contract and will remain stable.
"""

from jira_ingestion.config import JiraIngestionConfig
from jira_ingestion.clients.jira.value_stream_client import JiraValueStreamClient
from jira_ingestion.ingestion.pipeline import assemble_document, ingest_ticket
from jira_ingestion.ingestion.indexing import create_indexes
from jira_ingestion.models import (
    TicketInput,
    PipelineDocument,
    PreChunkDocument,
    ChunkRecord,
    TriageArtifact,
)

__all__ = [
    # Framework layer
    "JiraIngestionConfig",
    "JiraValueStreamClient",
    "ingest_ticket",
    "assemble_document",
    "create_indexes",
    # Stable schemas
    "TicketInput",
    "PipelineDocument",
    "PreChunkDocument",
    "ChunkRecord",
    "TriageArtifact",
]
