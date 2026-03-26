"""
Pipeline configuration — passed explicitly by the caller, no env loading.

Usage:
    from jira_ingestion import JiraIngestionConfig

    config = JiraIngestionConfig()                     # all defaults
    config = JiraIngestionConfig(max_slides=40, ocr_enabled=False)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JiraIngestionConfig:
    # --- Extraction limits ---
    max_slides: int = 60
    max_supplementary: int = 2
    ocr_enabled: bool = True
    section_min_slides: int = 8

    # --- Embedding / LLM ---
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o-mini"

    # --- Entity dictionaries ---
    entity_dict_path: str = "data/entity_dicts"

    # --- Triage thresholds ---
    min_file_size_bytes: int = 15_000
    min_pdf_size_bytes: int = 50_000
    max_file_size_bytes: int = 100_000_000
    layer1_skip_peek_score: int = 60
    layer1_skip_peek_gap: int = 20

    # --- Description thresholds ---
    desc_junk_max_words: int = 10
    desc_thin_max_words: int = 50
    desc_rich_min_words: int = 150

    # --- Entity extraction ---
    entity_confidence_text_match: float = 0.8
    entity_confidence_component_match: float = 1.0

    # --- Table summary cache ---
    table_summary_cache_ttl: int = 7 * 24 * 3600

    # --- Jira field mapping ---
    # Maps logical field names to Jira customfield_* IDs.
    # Override per tenant/project to keep ingestion portable across Jira instances.
    # Example for a non-standard tenant:
    #   jira_field_map={"business_unit": "customfield_10099", ...}
    jira_field_map: dict[str, str] = field(default_factory=lambda: {
        "business_unit": "customfield_10002",
        "product_area": "customfield_10003",
        "epic_link": "customfield_10014",
        "story_points": "customfield_10016",
        "sprint": "customfield_10020",
        "team": "customfield_10001",
        "epic_name": "customfield_10010",
        # Product / supervision label fields — override per tenant
        "impacted_products": "customfield_10040",
        "impacted_it_products": "customfield_10041",
        "requesting_org": "customfield_10050",
        "delivery_org": "customfield_10051",
        # Product stage / release field (e.g. fixVersions or a custom stage field)
        # Set to empty string "" to disable and fall back to fixVersions only
        "product_stage": "customfield_10060",
    })

    # --- Lineage / artifact persistence ---
    enable_raw_artifact_persistence: bool = True
    enable_attachment_text_persistence: bool = True
    enable_debug_stage_persistence: bool = False  # verbose — off by default
    enable_prechunk_persistence: bool = True       # pre-chunk assembled doc
    enable_attachment_inventory: bool = True        # per-attachment inventory JSON

    # --- Retrieval views ---
    enable_retrieval_views: bool = True  # build multi-focused retrieval views

    # --- HTTP / retry ---
    http_timeout_seconds: int = 120
    http_max_retries: int = 3
    http_retry_backoff_seconds: float = 1.5

    # --- Persistent store paths (None = use in-memory) ---
    metadata_store_path: Optional[str] = None
    supervision_store_path: Optional[str] = None
