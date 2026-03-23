"""
Main ingestion pipeline — Section 14 of the architecture spec.

Entry points:
    ingest_ticket(ticket_key, ...)  — single-ticket ingestion
    assemble_document(ticket_data, ...) — pure assembly (no I/O side effects)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def ingest_ticket(
    ticket_key: str,
    jira_client: Any,
    coarse_index: Any,
    fine_index: Any,
    metadata_index: Any,
    supervision_store: Any,
    trigger: str = "webhook",
    llm_client: Optional[Any] = None,
    embedding_client: Optional[Any] = None,
    dict_path: Optional[str] = None,
    force_reprocess: bool = False,
    storage_dir: Optional[str] = None,
    storage_fmt: str = "json",
    config: Optional[Any] = None,
) -> dict:
    """
    Fetch, process, and index a single Jira ticket.

    Args:
        ticket_key:        Jira issue key (e.g. "IDEA-1234").
        jira_client:       Authenticated JiraValueStreamClient instance.
        coarse_index:      Deck-level vector index.
        fine_index:        Chunk-level vector index.
        metadata_index:    BM25 metadata index.
        supervision_store: Supervision/label store.
        trigger:           'webhook' | 'backfill'
        llm_client:        OpenAI-compatible LLM client (optional).
        embedding_client:  OpenAI-compatible embedding client (optional).
        dict_path:         Path to entity dictionary files.
        force_reprocess:   If True, skip idempotency check.
        storage_dir:       If set, persist the assembled document here before
                           indexing. Accepts a directory path.
        storage_fmt:       'json' | 'jsonl' | 'parquet' (default 'json').
        config:            JiraIngestionConfig instance (uses defaults if None).
    """
    # Idempotency check
    if not force_reprocess:
        existing = coarse_index.get(ticket_key)
        if existing:
            ticket_data = await jira_client.get_ticket_data(ticket_key)
            updated = ticket_data["fields"].get("updated", "")
            stored_updated = (existing.get("metadata") or {}).get("updated_at", "")
            if updated and stored_updated and updated <= stored_updated:
                logger.info("Skipping %s — not modified since last ingest", ticket_key)
                return existing

    # Fetch ticket
    ticket_data = await jira_client.get_ticket_data(ticket_key)

    # Build download function for triage (wraps the async client in a sync callable)
    import asyncio

    def sync_download(att: dict) -> bytes:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(jira_client.download_attachment(att))

    # Assemble document
    document = assemble_document(
        ticket_data=ticket_data,
        download_fn=sync_download,
        llm_client=llm_client,
        embedding_client=embedding_client,
        dict_path=dict_path,
        config=config,
    )

    # Persist to disk before indexing (optional)
    if storage_dir:
        from .storage import DocumentStore
        store = DocumentStore(output_dir=storage_dir)
        store.save(document, fmt=storage_fmt)  # type: ignore[arg-type]

    # Index
    from .indexing import index_retrieval_view, index_supervision_view

    index_retrieval_view(document, coarse_index, fine_index, metadata_index)
    index_supervision_view(document, supervision_store)

    obs = document["observed"]
    sup = document["supervision"]
    logger.info(
        "Ingested %s [tier=%s chunks=%d sections=%d trainable=%s trigger=%s]",
        ticket_key,
        obs["quality_tier"],
        obs["stats"]["chunk_count"],
        obs["stats"]["section_count"],
        sup["trainability"]["is_trainable_for_vs"],
        trigger,
    )

    return document


# ---------------------------------------------------------------------------
# Pure assembly — can be used for testing / batch processing
# ---------------------------------------------------------------------------

def assemble_document(
    ticket_data: dict,
    download_fn: Optional[Callable[[dict], bytes]] = None,
    llm_client: Optional[Any] = None,
    embedding_client: Optional[Any] = None,
    dict_path: Optional[str] = None,
    config: Optional[Any] = None,
) -> dict:
    """
    Assemble a unified document from raw ticket data.

    This function is pure in the sense that it has no I/O side effects
    beyond calling download_fn and the external LLM/embedding APIs.

    Args:
        ticket_data:       Output of JiraValueStreamClient.get_ticket_data()
        download_fn:       Callable(attachment_dict) -> bytes (for triage & extraction)
        llm_client:        OpenAI-compatible client for summaries (optional)
        embedding_client:  OpenAI-compatible client for embeddings (optional)
        dict_path:         Path to entity dictionary JSON files.
        config:            JiraIngestionConfig instance (uses defaults if None).

    Returns:
        Unified document dict matching the schema in Section 12.
    """
    from src.config import JiraIngestionConfig
    from .metadata import extract_metadata, classify_links
    from .triage import triage_attachments
    from .description import classify_description, build_description_chunks
    from .quality import determine_quality_tier
    from .chunking import build_section_chunks, build_supplementary_chunk
    from .summary import generate_summary
    from .entities import extract_entities, load_entity_dictionaries, ensure_default_dictionaries
    from .embedding import embed_batch

    cfg = config if config is not None else JiraIngestionConfig()
    resolved_dict_path = dict_path or cfg.entity_dict_path

    ticket_key: str = ticket_data["key"]
    fields: dict = ticket_data.get("fields", {})

    # ------------------------------------------------------------------
    # 1. Metadata extraction
    # ------------------------------------------------------------------
    meta = extract_metadata(fields, ticket_key)
    meta["classified_links"] = classify_links(fields.get("issuelinks", []))

    # ------------------------------------------------------------------
    # 2. Attachment triage
    # ------------------------------------------------------------------
    attachments = ticket_data.get("attachments", [])
    primary, supplementary, att_quality = triage_attachments(
        attachments=attachments,
        ticket_summary=meta["summary"],
        download_fn=download_fn,
    )

    # ------------------------------------------------------------------
    # 3. Description classification
    # ------------------------------------------------------------------
    desc_class, desc_data = classify_description(fields.get("description"))

    # ------------------------------------------------------------------
    # 4. Primary content extraction
    # ------------------------------------------------------------------
    chunks: list[dict] = []
    content_source: Optional[str] = None

    if primary and download_fn is not None:
        ext = primary.get("ext", "")
        content_source = ext if ext in ("pptx", "ppt", "pdf", "docx", "doc") else None
        file_bytes = primary.get("file_bytes") or _safe_download(download_fn, primary)
        if file_bytes and content_source:
            result = _extract_primary(file_bytes, ext, cfg)
            chunks.extend(result.get("chunks", []))

    # ------------------------------------------------------------------
    # 5. Supplementary attachments (up to config.max_supplementary)
    # ------------------------------------------------------------------
    for supp in supplementary[:cfg.max_supplementary]:
        if download_fn is None:
            break
        supp_bytes = _safe_download(download_fn, supp)
        if supp_bytes:
            chunk = build_supplementary_chunk(supp_bytes, supp.get("ext", ""), supp.get("id", ""))
            if chunk:
                chunks.append(chunk)

    # ------------------------------------------------------------------
    # 6. Description chunks
    # ------------------------------------------------------------------
    has_attachment = content_source is not None
    desc_chunks = build_description_chunks(desc_class, desc_data, meta, has_attachment)
    chunks.extend(desc_chunks)

    # ------------------------------------------------------------------
    # 7. Comment chunks
    # ------------------------------------------------------------------
    for i, comment in enumerate(meta.get("substantive_comments", [])[:2]):
        chunks.append(
            {
                "chunk_id": f"comment-{i}",
                "source": "comment",
                "text": comment[:1500],
                "word_count": len(comment.split()),
                "is_boilerplate": False,
                "weight_multiplier": 0.5,
                "extraction_confidence": 0.6,
                "extraction_method": "comment",
            }
        )

    # ------------------------------------------------------------------
    # 8. Quality tier
    # ------------------------------------------------------------------
    quality_tier = determine_quality_tier(content_source, desc_class, chunks)

    # ------------------------------------------------------------------
    # 9. [v2] Section chunks
    # ------------------------------------------------------------------
    slide_chunks = [c for c in chunks if c["source"] in ("pptx_slide", "pdf_page")]
    section_chunks: list[dict] = []
    if len(slide_chunks) >= cfg.section_min_slides:
        section_chunks = build_section_chunks(slide_chunks)

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    all_for_summary = chunks + section_chunks
    summary_text = generate_summary(
        quality_tier=quality_tier,
        chunks=all_for_summary,
        metadata=meta,
        llm_client=llm_client,
    )

    # ------------------------------------------------------------------
    # 11. [v2] Entity extraction
    # ------------------------------------------------------------------
    ensure_default_dictionaries(resolved_dict_path)
    dictionaries = load_entity_dictionaries(resolved_dict_path)
    entity_mentions = extract_entities(chunks, meta, dictionaries)

    # ------------------------------------------------------------------
    # 12. Embeddings
    # ------------------------------------------------------------------
    texts_to_embed: list[str] = [summary_text]
    texts_to_embed.extend(c["text"] for c in chunks)
    texts_to_embed.extend(c["text"] for c in section_chunks)
    texts_to_embed.append(meta["metadata_text"])

    embeddings: list[list[float]] = []
    if embedding_client is not None:
        embeddings = embed_batch(texts_to_embed, embedding_client, model=cfg.embedding_model)
    else:
        logger.warning("No embedding client provided — embeddings will be empty lists")
        embeddings = [[] for _ in texts_to_embed]

    # Assign embeddings
    idx = 0
    summary_embedding = embeddings[idx]; idx += 1
    for c in chunks:
        c["embedding"] = embeddings[idx]; idx += 1
    for c in section_chunks:
        c["embedding"] = embeddings[idx]; idx += 1
    metadata_embedding = embeddings[idx]

    # ------------------------------------------------------------------
    # 13. Trainability
    # ------------------------------------------------------------------
    has_gold_vs = bool(meta.get("classified_links", {}).get("vs", []))
    avg_confidence = (
        sum(c.get("extraction_confidence", 1.0) for c in chunks) / max(len(chunks), 1)
    )
    from .quality import TIER_WEIGHTS

    source_quality_score = TIER_WEIGHTS[quality_tier] * avg_confidence

    # ------------------------------------------------------------------
    # 14. [v2] Assemble with observed / supervision split
    # ------------------------------------------------------------------
    now = datetime.now(timezone.utc).isoformat()

    # Stats
    non_boilerplate_slides = len(
        [c for c in chunks if c["source"] in ("pptx_slide", "pdf_page") and not c.get("is_boilerplate")]
    )
    stats = {
        "chunk_count": len(chunks),
        "section_count": len(section_chunks),
        "non_boilerplate_slides": non_boilerplate_slides,
        "total_word_count": sum(c.get("word_count", len(c["text"].split())) for c in chunks),
        "has_tables": any("table" in c.get("source", "") for c in chunks),
        "has_speaker_notes": any(c.get("has_notes") for c in chunks),
        "table_count": len([c for c in chunks if "table" in c.get("source", "")]),
        "entity_mention_count": sum(len(v) for v in entity_mentions.values()),
    }

    return {
        "ticket_key": ticket_key,
        "schema_version": "2.0",
        "ingested_at": now,
        "observed": {
            "quality_tier": quality_tier,
            "created": meta.get("created", ""),
            "content_source": content_source or "none",
            "triage": {
                "primary_attachment": primary["filename"] if primary else None,
                "triage_score": primary.get("triage_score") if primary else None,
                "triage_reasons": primary.get("triage_reasons", []) if primary else [],
                "supplementary_attachments": [s["filename"] for s in supplementary],
                "attachment_count_total": len(attachments),
                "attachment_count_viable": len(
                    [a for a in attachments if a.get("size", 0) >= 15_000]
                ),
            },
            "summary_text": summary_text,
            "summary_embedding": summary_embedding,
            "chunks": chunks,
            "section_chunks": section_chunks,
            "entity_mentions": entity_mentions,
            "metadata": {**meta, "classified_links": meta["classified_links"]},
            "metadata_text": meta["metadata_text"],
            "metadata_embedding": metadata_embedding,
            "stats": stats,
        },
        "supervision": {
            "vs_labels": [lnk["summary"] for lnk in meta["classified_links"].get("vs", [])],
            "vs_label_source": "jira_issuelinks",
            "product_stage_labels": [],
            "impacted_products": [],
            "impacted_products_source": None,
            "trainability": {
                "has_gold_vs_labels": has_gold_vs,
                "has_gold_product_labels": False,
                "is_trainable_for_vs": has_gold_vs and quality_tier in ("A", "B", "C"),
                "is_trainable_for_product": False,
                "source_quality_score": round(source_quality_score, 4),
                "label_snapshot_time": now,
            },
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_primary(file_bytes: bytes, ext: str, config: Any) -> dict:
    """Run structured extraction on the primary attachment."""
    try:
        if ext in ("pptx", "ppt"):
            from .extraction.pptx import extract_pptx
            return extract_pptx(file_bytes, max_slides=config.max_slides)

        elif ext == "pdf":
            from .extraction.pdf import extract_pdf
            return extract_pdf(file_bytes, ocr_enabled=config.ocr_enabled, max_pages=config.max_slides)

        elif ext in ("docx", "doc"):
            from .extraction.docx import extract_docx
            return extract_docx(file_bytes)

    except Exception as exc:
        logger.error("Primary extraction failed for ext=%s: %s", ext, exc)

    return {"chunks": []}


def _safe_download(download_fn: Callable, att: dict) -> Optional[bytes]:
    """Download with error handling."""
    try:
        return download_fn(att)
    except Exception as exc:
        logger.warning("Download failed for %s: %s", att.get("filename"), exc)
        return None
