"""
Pipeline configuration — passed explicitly by the caller, no env loading.

Usage:
    from jira_ingestion import JiraIngestionConfig

    config = JiraIngestionConfig()                     # all defaults
    config = JiraIngestionConfig(max_slides=40, ocr_enabled=False)
"""

from __future__ import annotations

from dataclasses import dataclass, field


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

    # --- Section chunking ---
    # (alias for section_min_slides — kept for backwards compat)

    # --- Entity extraction ---
    entity_confidence_text_match: float = 0.8
    entity_confidence_component_match: float = 1.0

    # --- Table summary cache ---
    table_summary_cache_ttl: int = 7 * 24 * 3600
