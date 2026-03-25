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
    issue_type: str
    status: str
    resolution: str
    requesting_org: str
    delivery_org: str
    epic_key: Optional[str]
    substantive_comments: List[str]
    metadata_text: str
    classified_links: ClassifiedLinks


# ---------------------------------------------------------------------------
# Triage multi-dimensional scores
# ---------------------------------------------------------------------------

class TriageScores(TypedDict, total=False):
    """
    Multi-dimensional triage scores computed from extracted text.
    All values are in the range [0.0, 1.0].
    """
    extraction_quality: float    # native text vs OCR, readability, formatting damage
    semantic_density: float      # business/process signal density
    idea_card_likeness: float    # problem/solution/value structure match
    retrieval_readiness: float   # usefulness for value stream / product prediction


# ---------------------------------------------------------------------------
# Stage 4 — observed (retrieval) view
# ---------------------------------------------------------------------------

class TriageSummary(TypedDict, total=False):
    primary_attachment: Optional[str]
    primary_attachment_id: Optional[str]
    triage_score: Optional[int]
    triage_reasons: List[str]
    triage_scores: TriageScores           # multi-dimensional scores
    supplementary_attachments: List[str]
    attachment_count_total: int
    attachment_count_viable: int
    att_quality: str                      # "good" | "fallback" | "none"
    selection_reason: str


class AttachmentRecord(TypedDict, total=False):
    """
    Per-attachment extraction inventory.
    Stored in observed.attachment_inventory — one record per attachment processed.
    """
    attachment_id: Required[str]
    filename: Required[str]
    mime_type: str
    size: int
    created_at: str
    extraction_status: str        # "success" | "failed" | "skipped"
    extraction_method: str        # "pptx" | "pdf_native" | "pdf_ocr" | "docx" | "none"
    extraction_confidence: float
    raw_text: str
    cleaned_text: str
    text_length: int
    text_preview: str             # first 200 chars of cleaned text
    triage_score: int
    triage_reasons: List[str]
    triage_scores: TriageScores


class CommentRecord(TypedDict, total=False):
    """Structured representation of a single Jira comment."""
    comment_id: str
    author: str
    created_at: str
    body_raw: str
    body_cleaned: str
    word_count: int
    is_substantive: bool          # True if >50 words and not a bot comment


class RetrievalViews(TypedDict, total=False):
    """
    Multiple focused retrieval views for a ticket.
    Each view is a short text optimised for a specific retrieval angle.
    These are derived dynamically — never hardcoded keyword slices.
    """
    overview: str                 # high-level ticket summary + context
    problem_objective: str        # problem statement and objectives
    solution_capability: str      # proposed solution and capability signals
    value_metrics: str            # value proposition, ROI, KPIs
    attachment_focused: str       # primary attachment content condensed


class ProvenanceRecord(TypedDict, total=False):
    """
    Quality and provenance fields for downstream trust scoring.
    Used to decide whether a ticket is suitable for retrieval / training.
    """
    has_description: bool
    has_attachments: bool
    has_primary_attachment: bool
    has_theme_links: bool
    has_product_labels: bool
    source_quality_score: float
    primary_evidence_type: str    # "pptx" | "pdf" | "docx" | "description" | "none"
    primary_evidence_name: str
    selection_reason: str
    extraction_provenance: str    # "native" | "ocr" | "text_only" | "fallback"


class DerivedFields(TypedDict, total=False):
    """
    LLM-derived summaries and keyword lists.
    These augment raw/cleaned text — they never replace it.
    """
    ticket_summary_llm: str
    problem_summary_llm: str
    solution_summary_llm: str
    capability_keywords_llm: List[str]
    workflow_terms_llm: List[str]
    business_entities_llm: List[str]
    product_mentions_llm: List[str]


class PipelineStats(TypedDict):
    chunk_count: int
    section_count: int
    non_boilerplate_slides: int
    total_word_count: int
    has_tables: bool
    has_speaker_notes: bool
    table_count: int
    entity_mention_count: int
    comment_count: int
    attachment_count: int


class ObservedDocument(TypedDict, total=False):
    quality_tier: Required[Literal["A", "B", "C", "D"]]
    created: str
    updated_at: str
    content_source: str
    triage: TriageSummary
    # Raw layer — never overwritten after initial capture
    description_raw: str
    comments_raw: List[str]
    # Cleaned layer — sanitised for retrieval use
    description_cleaned: str
    comments_cleaned: List[str]
    # Derived layer — LLM outputs, augment raw; never replace
    derived: DerivedFields
    # Structured records
    attachment_inventory: List[AttachmentRecord]
    comment_records: List[CommentRecord]
    retrieval_views: RetrievalViews
    provenance: ProvenanceRecord
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
    # Value stream / theme labels (ground truth — not embedded in retrieval text)
    vs_labels: List[str]
    vs_label_source: str
    theme_links_raw: List[Dict[str, Any]]        # raw VS link dicts from Jira
    linked_value_stream_ids: List[str]
    linked_value_stream_names: List[str]
    linked_value_stream_statuses: List[str]
    # Product labels
    product_stage_labels: List[str]
    impacted_products: List[str]                  # legacy flat list
    impacted_products_source: Optional[str]
    impacted_product_ids: List[str]
    impacted_product_names: List[str]
    # IT product labels
    impacted_it_products_raw: List[Dict[str, Any]]
    impacted_it_product_ids: List[str]
    impacted_it_product_names: List[str]
    # Graph-ready candidate signals (derived from entity extraction + issue links)
    candidate_product_mentions: List[str]
    capability_mentions: List[str]
    process_terms: List[str]
    system_mentions: List[str]
    business_function_mentions: List[str]
    trainability: TrainabilityRecord


# ---------------------------------------------------------------------------
# Top-level pipeline document
# ---------------------------------------------------------------------------

class PipelineDocument(TypedDict):
    """
    The unified document returned by assemble_document() and ingest_ticket().
    Schema version 3.0.
    """
    ticket_key: str
    schema_version: str       # "3.0"
    ingested_at: str          # ISO 8601 UTC
    observed: ObservedDocument
    supervision: SupervisionDocument
