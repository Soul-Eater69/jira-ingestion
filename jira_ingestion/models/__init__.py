"""
Typed document schemas for the Jira ingestion pipeline.

These TypedDicts define the contract at every stage boundary.  They are
lightweight (stdlib only — no Pydantic required) but give IDEs and type-
checkers enough information to catch contract drift early.

Schema version history:
  2.0  — observed / supervision split
  3.0  — raw / observed / supervision / derived layer separation,
          multi-score triage artifact, retrieval views, product labels,
          attachment inventory, enriched comments

Import pattern:
    from jira_ingestion.models import (
        AttachmentMeta,
        TicketInput,
        ChunkRecord,
        TriageArtifact,
        AttachmentInventoryItem,
        RetrievalViews,
        ProductLabels,
        CommentRecord,
        ProvenanceRecord,
        RawLayer,
        ObservedDocument,
        SupervisionDocument,
        DerivedLayer,
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
# Triage artifact — multi-score, first-class stage output
# ---------------------------------------------------------------------------

class TriageScores(TypedDict):
    """Multi-dimensional triage scores (0.0 – 1.0 each)."""
    extraction_quality: float    # native text vs OCR, formatting damage
    semantic_density: float      # business/process signal density
    idea_card_likeness: float    # matches idea card structure
    retrieval_readiness: float   # useful for value stream prediction


class TriageArtifact(TypedDict, total=False):
    """
    Structured triage result — persisted as 03_triage_output.json.

    Separates the primary evidence decision from raw scores so downstream
    systems can audit how content was selected.
    """
    primary_attachment: Optional[str]
    primary_attachment_id: Optional[str]
    supplementary_attachments: List[str]
    att_quality: str              # 'good' | 'fallback' | 'none'
    quality_tier: str             # 'A' | 'B' | 'C' | 'none'
    selection_reason: str
    scores: TriageScores
    per_attachment_scores: List[Dict[str, Any]]
    attachment_count_total: int
    attachment_count_viable: int
    # Backward compat with v2.0 schema
    triage_score: Optional[int]
    triage_reasons: List[str]


# ---------------------------------------------------------------------------
# Attachment inventory — per-attachment extraction record
# ---------------------------------------------------------------------------

class AttachmentInventoryItem(TypedDict, total=False):
    """
    Full record for one attachment — persisted as 02_attachment_contents.json.
    Every ticket should have a visible attachment inventory.
    """
    attachment_id: Required[str]
    filename: Required[str]
    mime_type: str
    size: int
    created_at: str
    is_primary: bool
    is_supplementary: bool
    extraction_status: str    # 'extracted' | 'skipped' | 'failed' | 'filtered'
    extraction_method: str    # 'markitdown' | 'pptx' | 'pdf' | 'docx' | 'none'
    extraction_confidence: float
    raw_text: str
    cleaned_text: str
    text_length: int
    text_preview: str         # first 300 chars
    triage_score: Optional[int]
    triage_reasons: List[str]
    scores: Dict[str, float]  # TriageScores as plain dict


# ---------------------------------------------------------------------------
# Comment record — raw + cleaned
# ---------------------------------------------------------------------------

class CommentRecord(TypedDict, total=False):
    """One comment with both raw and cleaned representations."""
    comment_id: str
    author: str
    created: str
    body_raw: str
    body_cleaned: str
    word_count: int
    is_substantive: bool    # word_count >= 50


class CommentsEnriched(TypedDict):
    """Enriched comment container stored in the raw layer."""
    comments_raw: List[CommentRecord]
    comments_cleaned: List[str]       # substantive cleaned bodies only
    comment_count: int
    substantive_count: int
    important_spans: List[str]        # key sentences / mentions


# ---------------------------------------------------------------------------
# Retrieval views — multiple focused representations
# ---------------------------------------------------------------------------

class RetrievalViews(TypedDict, total=False):
    """
    Multiple focused retrieval views derived dynamically from ticket content.
    No supervision labels should appear in any of these views.
    """
    overview: str               # summary + key metadata context
    problem_objective: str      # problem/background/challenge content
    solution_capability: str    # solution/approach/capability content
    value_proposition: str      # business value/ROI/metrics content
    attachment_focused: str     # primary attachment text, trimmed for retrieval


# ---------------------------------------------------------------------------
# Product labels — supervision view, never embedded into retrieval text
# ---------------------------------------------------------------------------

class ProductLabels(TypedDict, total=False):
    """
    Impacted product / IT product labels extracted from Jira custom fields.
    Stored in the supervision layer only — not searchable via retrieval.
    """
    raw: List[Any]       # raw field value from Jira API
    ids: List[str]       # extracted IDs
    names: List[str]     # extracted display names


# ---------------------------------------------------------------------------
# Provenance record — quality + source traceability
# ---------------------------------------------------------------------------

class ProvenanceRecord(TypedDict, total=False):
    """Quality and provenance metadata for downstream trust decisions."""
    quality_tier: str              # 'A' | 'B' | 'C' | 'D'
    has_description: bool
    has_attachments: bool
    has_primary_attachment: bool
    has_theme_links: bool
    has_product_labels: bool
    has_it_product_labels: bool
    source_quality_score: float
    primary_evidence_type: str     # 'pptx' | 'pdf' | 'docx' | 'description' | 'none'
    primary_evidence_name: Optional[str]
    selection_reason: str
    extraction_provenance: str     # 'native' | 'ocr' | 'description' | 'none'


# ---------------------------------------------------------------------------
# Raw layer — immutable source data
# ---------------------------------------------------------------------------

class RawLayer(TypedDict, total=False):
    """
    Immutable raw data layer.  Never overwrite.  Persisted as 01_raw_ticket.json.
    """
    description: Optional[str]           # raw Jira description field
    comments: CommentsEnriched            # raw + cleaned comments
    issue_links: List[Dict[str, Any]]     # raw issuelinks array
    attachment_inventory: List[AttachmentInventoryItem]


# ---------------------------------------------------------------------------
# Stage 2 — per-chunk record
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
    epic_key: Optional[str]
    requesting_org: str
    delivery_org: str
    substantive_comments: List[str]
    metadata_text: str
    classified_links: ClassifiedLinks


# ---------------------------------------------------------------------------
# Stage 4 — observed (retrieval) view — schema v3.0
# ---------------------------------------------------------------------------

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
    substantive_comment_count: int


class ObservedDocument(TypedDict, total=False):
    """
    Retrieval view — contains NO supervision labels.
    What is stored here is safe to embed and search.
    """
    quality_tier: Required[Literal["A", "B", "C", "D"]]
    created: str
    updated_at: str
    content_source: str
    provenance: ProvenanceRecord
    triage: TriageArtifact
    description_class: str          # 'rich' | 'usable' | 'thin' | 'junk' | 'empty'
    description_cleaned: str        # cleaned text, empty string if junk/empty
    primary_attachment_text: str    # extracted primary attachment text
    comments_cleaned: List[str]     # substantive comment texts
    retrieval_views: RetrievalViews
    retrieval_text: str             # combined text for embedding (no labels)
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
# Stage 4 — supervision layer — labels only, never embedded for retrieval
# ---------------------------------------------------------------------------

class TrainabilityRecord(TypedDict):
    has_gold_vs_labels: bool
    has_gold_product_labels: bool
    is_trainable_for_vs: bool
    is_trainable_for_product: bool
    source_quality_score: float
    label_snapshot_time: str


class SupervisionDocument(TypedDict, total=False):
    """
    Supervision view — ground-truth labels for training and evaluation.
    NEVER accessed during retrieval.  Stored in the supervision store only.
    """
    # Value stream labels (strongest supervision signal)
    vs_labels: List[str]           # display names (for backward compat)
    vs_label_source: str
    linked_value_stream_ids: List[str]
    linked_value_stream_names: List[str]
    linked_value_stream_statuses: List[str]
    theme_links_raw: List[Dict[str, Any]]
    # Product labels
    impacted_products: ProductLabels
    impacted_it_products: ProductLabels
    # Other
    product_stage_labels: List[str]
    trainability: TrainabilityRecord


# ---------------------------------------------------------------------------
# Derived layer — LLM outputs, populated if LLM client available
# ---------------------------------------------------------------------------

class DerivedLayer(TypedDict, total=False):
    """
    LLM-derived artifacts — augment raw/cleaned text, never replace it.
    Populated after triage, before chunking, if an LLM client is provided.
    """
    ticket_summary_llm: str
    problem_summary_llm: str
    solution_summary_llm: str
    capability_keywords_llm: List[str]
    workflow_terms_llm: List[str]
    business_entities_llm: List[str]
    product_mentions_llm: List[str]


# ---------------------------------------------------------------------------
# Top-level pipeline document — schema v3.0
# ---------------------------------------------------------------------------

class PipelineDocument(TypedDict):
    """
    The unified document returned by assemble_document() and ingest_ticket().

    Schema v3.0 adds:
      - raw     layer: immutable source data
      - derived layer: LLM outputs
      - Expanded supervision with product labels, VS IDs
      - Multi-score triage artifact
      - Multiple retrieval views
      - Attachment inventory
      - Enriched comments
    """
    ticket_key: str
    schema_version: str       # "3.0"
    ingested_at: str          # ISO 8601 UTC
    raw: RawLayer
    observed: ObservedDocument
    supervision: SupervisionDocument
    derived: DerivedLayer


# ---------------------------------------------------------------------------
# Pre-chunk assembled document — persisted as 04_assembled_prechunk.json
# ---------------------------------------------------------------------------

class PreChunkDocument(TypedDict, total=False):
    """
    Assembled canonical ticket document before chunking.
    Inspect this artifact to debug content selection decisions.
    """
    ticket_key: Required[str]
    summary: str
    description_raw: str
    description_cleaned: str
    description_class: str
    triage: TriageArtifact
    primary_attachment_text: str
    supplementary_previews: List[Dict[str, str]]  # [{filename, preview}]
    linked_themes: List[Dict[str, Any]]
    labels: List[str]
    components: List[str]
    org_metadata: Dict[str, str]        # business_unit, product_area, etc.
    comments_enriched: CommentsEnriched
    retrieval_views: RetrievalViews
    retrieval_text: str
    provenance: ProvenanceRecord
