"""
Typed document schemas for the Jira ingestion pipeline.

These TypedDicts define the contract at every stage boundary and serve as
living documentation of the pipeline's data shapes.  They are lightweight
(stdlib only — no Pydantic required) but give IDEs and type-checkers enough
information to catch contract drift early.

Import pattern:
    from jira_ingestion.models import (
        AttachmentMeta,
        TicketInput,
        ChunkRecord,
        ObservedDocument,
        SupervisionDocument,
        PipelineDocument,
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict, Required


# ---------------------------------------------------------------------------
# Stage 0 — raw Jira input
# ---------------------------------------------------------------------------

class AttachmentMeta(TypedDict, total=False):
    """Raw attachment metadata dict as returned by the Jira API."""
    id: Required[str]
    filename: Required[str]
    content: str          # download URL
    mimeType: str
    size: int
    created: str
    author: Dict[str, Any]
    # Added by triage layers
    ext: str
    triage_score: int
    triage_reasons: List[str]
    peek_metadata: Dict[str, Any]
    file_bytes: bytes
    confirmed: bool


class TicketInput(TypedDict, total=False):
    """
    Output of JiraValueStreamClient.get_ticket_data().
    The pipeline expects exactly this shape — see pipeline.py.
    """
    key: Required[str]
    fields: Required[Dict[str, Any]]
    attachments: List[AttachmentMeta]
    themes: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Stage 2 — per-chunk record (produced by extraction, chunking, description)
# ---------------------------------------------------------------------------

class ChunkRecord(TypedDict, total=False):
    """A single content chunk as assembled by the pipeline."""
    chunk_id: Required[str]
    source: Required[str]   # pptx_slide | pdf_page | description | comment | section | supplementary
    text: Required[str]
    word_count: int
    is_boilerplate: bool
    weight_multiplier: float
    extraction_confidence: float
    extraction_method: str
    # Optional slide/page metadata
    slide_num: Optional[int]
    page_num: Optional[int]
    has_table: bool
    has_notes: bool
    section_title: Optional[str]
    # Added after embedding step
    embedding: List[float]


# ---------------------------------------------------------------------------
# Stage 3 — assembled metadata
# ---------------------------------------------------------------------------

class ClassifiedLinks(TypedDict):
    vs: List[Dict[str, Any]]
    product: List[Dict[str, Any]]
    dependency: List[Dict[str, Any]]
    parent: List[Dict[str, Any]]
    related: List[Dict[str, Any]]
    implementation: List[Dict[str, Any]]
    unknown: List[Dict[str, Any]]


class TicketMetadata(TypedDict, total=False):
    ticket_key: Required[str]
    title: Required[str]
    summary: str
    reporter: str
    created: str
    labels: List[str]
    components: List[str]
    business_unit: str
    product_area: str
    priority: str
    epic_key: Optional[str]
    substantive_comments: List[str]
    metadata_text: str
    classified_links: ClassifiedLinks


# ---------------------------------------------------------------------------
# Stage 4 — observed (retrieval) view
# ---------------------------------------------------------------------------

class TriageSummary(TypedDict, total=False):
    primary_attachment: Optional[str]
    triage_score: Optional[int]
    triage_reasons: List[str]
    supplementary_attachments: List[str]
    attachment_count_total: int
    attachment_count_viable: int


class PipelineStats(TypedDict):
    chunk_count: int
    section_count: int
    non_boilerplate_slides: int
    total_word_count: int
    has_tables: bool
    has_speaker_notes: bool
    table_count: int
    entity_mention_count: int


class ObservedDocument(TypedDict, total=False):
    quality_tier: Required[Literal["A", "B", "C", "D"]]
    created: str
    updated_at: str
    content_source: str
    triage: TriageSummary
    summary_text: str
    summary_embedding: List[float]
    chunks: List[ChunkRecord]
    section_chunks: List[ChunkRecord]
    entity_mentions: Dict[str, List[Dict[str, Any]]]
    metadata: TicketMetadata
    metadata_text: str
    metadata_embedding: List[float]
    stats: PipelineStats


# ---------------------------------------------------------------------------
# Stage 4 — supervision (labels/training) view
# ---------------------------------------------------------------------------

class TrainabilityRecord(TypedDict):
    has_gold_vs_labels: bool
    has_gold_product_labels: bool
    is_trainable_for_vs: bool
    is_trainable_for_product: bool
    source_quality_score: float
    label_snapshot_time: str


class SupervisionDocument(TypedDict, total=False):
    vs_labels: List[str]
    vs_label_source: str
    product_stage_labels: List[str]
    impacted_products: List[str]
    impacted_products_source: Optional[str]
    trainability: TrainabilityRecord


# ---------------------------------------------------------------------------
# Top-level pipeline document
# ---------------------------------------------------------------------------

class PipelineDocument(TypedDict):
    """
    The unified document returned by assemble_document() and ingest_ticket().
    Schema version 2.0.
    """
    ticket_key: str
    schema_version: str       # "2.0"
    ingested_at: str          # ISO 8601 UTC
    observed: ObservedDocument
    supervision: SupervisionDocument
