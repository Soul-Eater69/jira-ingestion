"""
Main ingestion pipeline — schema v3.0.

Layer separation:
  raw        — immutable source data (description, comments, issue links, attachment inventory)
  observed   — retrieval view, NO supervision labels
  supervision — ground-truth labels only (value streams, products)
  derived    — LLM outputs (populated when llm_client provided)

Entry points:
    ingest_ticket(ticket_key, ...)  — single-ticket ingestion
    assemble_document(ticket_data, ...) — pure assembly (no I/O side effects)

Debug artifacts written when storage_dir is set:
    01_raw_ticket.json
    02_attachment_contents.json
    03_triage_output.json
    04_assembled_prechunk.json
    05_debug_report.json
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

if TYPE_CHECKING:
    from .indexing import BaseVectorIndex, BaseMetadataIndex, BaseSupervisionStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ts(value: str) -> Optional[datetime]:
    """Parse an ISO 8601 timestamp string. Returns None on empty/malformed."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.debug("Could not parse timestamp: %r", value)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def ingest_ticket(
    ticket_key: str,
    jira_client: Any,
    coarse_index: "BaseVectorIndex",
    fine_index: "BaseVectorIndex",
    metadata_index: "BaseMetadataIndex",
    supervision_store: "BaseSupervisionStore",
    trigger: str = "webhook",
    llm_client: Optional[Any] = None,
    embedding_client: Optional[Any] = None,
    dict_path: Optional[str] = None,
    force_reprocess: bool = False,
    storage_dir: Optional[str] = None,
    storage_fmt: Literal["json", "jsonl", "parquet"] = "json",
    config: Optional[Any] = None,
) -> dict:
    """Fetch, process, and index a single Jira ticket."""
    from jira_ingestion.config import JiraIngestionConfig
    cfg = config if config is not None else JiraIngestionConfig()

    ticket_data = await jira_client.get_ticket_data(ticket_key, config=cfg)

    # Idempotency check
    if not force_reprocess:
        existing = coarse_index.get(ticket_key)
        if existing:
            updated = ticket_data["fields"].get("updated", "")
            stored_updated = (existing.get("metadata") or {}).get("updated_at", "")
            updated_dt = _parse_ts(updated)
            stored_dt = _parse_ts(stored_updated)
            if updated_dt and stored_dt and updated_dt <= stored_dt:
                logger.info("Skipping %s — not modified since last ingest", ticket_key)
                return existing

    # Pre-download top attachment candidates asynchronously
    from .triage import layer0_filter, layer1_score
    _prefetched: dict[str, bytes] = {}
    _attachments = ticket_data.get("attachments", [])
    _candidates = layer1_score(layer0_filter(_attachments))[:5]
    for _att in _candidates:
        _att_id = str(_att.get("id") or _att.get("filename", ""))
        try:
            _prefetched[_att_id] = await jira_client.download_attachment(_att)
        except Exception as _exc:
            logger.warning("Pre-fetch failed for %s: %s", _att.get("filename"), _exc)

    def cached_download(att: dict) -> bytes:
        att_id = str(att.get("id") or att.get("filename", ""))
        cached = _prefetched.get(att_id)
        if cached is not None:
            return cached
        raise RuntimeError(
            f"Bytes not pre-fetched for attachment '{att.get('filename')}'."
        )

    document = assemble_document(
        ticket_data=ticket_data,
        download_fn=cached_download,
        llm_client=llm_client,
        embedding_client=embedding_client,
        dict_path=dict_path,
        config=cfg,
    )

    if storage_dir:
        from .storage import DocumentStore
        store = DocumentStore(output_dir=storage_dir)

        # ── Numbered debug artifacts (always emitted when storage_dir is set) ──
        # 01: raw Jira ticket
        store.save_pipeline_artifact(ticket_key, 1, "raw_ticket", ticket_data)
        # 02: attachment inventory
        store.save_pipeline_artifact(
            ticket_key, 2, "attachment_contents",
            document["raw"].get("attachment_inventory", [])
        )
        # 03: triage output
        store.save_pipeline_artifact(ticket_key, 3, "triage_output", document["observed"]["triage"])
        # 04: pre-chunk assembled document
        store.save_pipeline_artifact(
            ticket_key, 4, "assembled_prechunk",
            _build_prechunk_artifact(document)
        )
        # 05: debug report (compact summary of all pipeline decisions)
        store.save_pipeline_artifact(
            ticket_key, 5, "debug_report",
            _build_debug_report(document)
        )

        # ── Optional additional artifacts ──
        if cfg.enable_attachment_text_persistence and _attachments:
            extracted = await jira_client.fetch_attachment_content(_attachments)
            store.save_extracted_attachments(ticket_key, extracted)

        if cfg.enable_debug_stage_persistence:
            store.save_debug_stage(ticket_key, "prefetched_ids", list(_prefetched.keys()))

        # ── Main assembled document ──
        store.save(document, fmt=storage_fmt)

    from .indexing import index_retrieval_view, index_supervision_view
    index_retrieval_view(document, coarse_index, fine_index, metadata_index)
    index_supervision_view(document, supervision_store)

    obs = document["observed"]
    sup = document["supervision"]
    logger.info(
        "Ingested %s [tier=%s chunks=%d trainable=%s trigger=%s]",
        ticket_key,
        obs["quality_tier"],
        obs["stats"]["chunk_count"],
        sup["trainability"]["is_trainable_for_vs"],
        trigger,
    )
    return document


# ---------------------------------------------------------------------------
# Pure assembly
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
    Assemble a unified v3.0 document from raw ticket data.

    Output layers:
        raw        — immutable source data
        observed   — retrieval view (no supervision labels)
        supervision — ground-truth labels
        derived    — LLM outputs
    """
    from jira_ingestion.config import JiraIngestionConfig
    from .metadata import extract_metadata, classify_links, extract_product_fields, extract_comments_enriched, extract_stage_labels
    from .triage import triage_attachments, build_triage_artifact, layer0_filter, layer1_score
    from .description import classify_description, build_description_chunks, clean_jira_markup
    from .quality import determine_quality_tier, TIER_WEIGHTS
    from .chunking import build_section_chunks, build_supplementary_chunk
    from .summary import generate_summary, generate_derived_artifacts
    from .entities import extract_entities, load_entity_dictionaries, ensure_default_dictionaries
    from .embedding import embed_batch

    cfg = config if config is not None else JiraIngestionConfig()
    resolved_dict_path = dict_path or cfg.entity_dict_path

    ticket_key: str = ticket_data["key"]
    fields: dict = ticket_data.get("fields", {})

    # ------------------------------------------------------------------
    # 1. Metadata extraction (retrieval-safe fields only)
    # ------------------------------------------------------------------
    meta = extract_metadata(fields, ticket_key, config=cfg)
    classified_links = classify_links(fields.get("issuelinks", []))
    meta["classified_links"] = classified_links

    jira_field_map: dict = getattr(cfg, "jira_field_map", {})

    # ------------------------------------------------------------------
    # 2. Product / supervision labels (not metadata — supervision layer)
    # ------------------------------------------------------------------
    product_fields = extract_product_fields(fields, jira_field_map)
    product_stage_labels = extract_stage_labels(fields, jira_field_map)

    # ------------------------------------------------------------------
    # 3. Comments — enriched (raw + cleaned)
    # ------------------------------------------------------------------
    comments_enriched = extract_comments_enriched(fields.get("comment") or {})

    # ------------------------------------------------------------------
    # 4. Description
    # ------------------------------------------------------------------
    raw_description = fields.get("description") or ""
    desc_class, desc_data = classify_description(raw_description)
    description_cleaned = desc_data["text"] if desc_data else ""

    # ------------------------------------------------------------------
    # 5. Attachment triage
    # ------------------------------------------------------------------
    attachments = ticket_data.get("attachments", [])
    all_l1_scored = layer1_score(layer0_filter(attachments))
    primary, supplementary, att_quality = triage_attachments(
        attachments=attachments,
        ticket_summary=meta["summary"],
        download_fn=download_fn,
    )
    triage_artifact = build_triage_artifact(
        primary=primary,
        supplementary=supplementary,
        att_quality=att_quality,
        all_scored=all_l1_scored,
        total_attachment_count=len(attachments),
    )

    # ------------------------------------------------------------------
    # 6. Attachment inventory (per-attachment extraction metadata)
    # ------------------------------------------------------------------
    attachment_inventory = _build_attachment_inventory(
        attachments, all_l1_scored, primary, supplementary
    )

    # ------------------------------------------------------------------
    # 7. Primary content extraction
    # ------------------------------------------------------------------
    chunks: list[dict] = []
    content_source: Optional[str] = None
    primary_attachment_text = ""

    if primary and download_fn is not None:
        ext = primary.get("ext", "")
        content_source = ext if ext in ("pptx", "ppt", "pdf", "docx", "doc") else None
        file_bytes = primary.get("file_bytes") or _safe_download(download_fn, primary)
        if file_bytes and content_source:
            result = _extract_primary(file_bytes, ext, cfg)
            chunks.extend(result.get("chunks", []))
            primary_attachment_text = " ".join(
                c["text"] for c in result.get("chunks", []) if not c.get("is_boilerplate")
            )

    # ------------------------------------------------------------------
    # 8. Supplementary attachments
    # ------------------------------------------------------------------
    supplementary_previews: list[dict] = []
    for supp in supplementary[:cfg.max_supplementary]:
        if download_fn is None:
            break
        supp_bytes = _safe_download(download_fn, supp)
        if supp_bytes:
            chunk = build_supplementary_chunk(supp_bytes, supp.get("ext", ""), supp.get("id", ""))
            if chunk:
                chunks.append(chunk)
                supplementary_previews.append({
                    "filename": supp.get("filename", ""),
                    "preview": chunk["text"][:300],
                })

    # ------------------------------------------------------------------
    # 9. Description chunks
    # ------------------------------------------------------------------
    has_attachment = content_source is not None
    desc_chunks = build_description_chunks(desc_class, desc_data, meta, has_attachment)
    chunks.extend(desc_chunks)

    # ------------------------------------------------------------------
    # 10. Comment chunks (lower weight — supplementary evidence)
    # ------------------------------------------------------------------
    for i, comment_text in enumerate(comments_enriched["comments_cleaned"][:2]):
        chunks.append({
            "chunk_id": f"comment-{i}",
            "source": "comment",
            "text": comment_text[:1500],
            "word_count": len(comment_text.split()),
            "is_boilerplate": False,
            "weight_multiplier": 0.5,
            "extraction_confidence": 0.6,
            "extraction_method": "comment",
        })

    # ------------------------------------------------------------------
    # 11. Quality tier
    # ------------------------------------------------------------------
    quality_tier = determine_quality_tier(content_source, desc_class, chunks)

    # ------------------------------------------------------------------
    # 12. Section chunks
    # ------------------------------------------------------------------
    slide_chunks = [c for c in chunks if c["source"] in ("pptx_slide", "pdf_page")]
    section_chunks: list[dict] = []
    if len(slide_chunks) >= cfg.section_min_slides:
        section_chunks = build_section_chunks(slide_chunks)

    # ------------------------------------------------------------------
    # 13. Summary
    # ------------------------------------------------------------------
    all_for_summary = chunks + section_chunks
    summary_text = generate_summary(
        quality_tier=quality_tier,
        chunks=all_for_summary,
        metadata=meta,
        llm_client=llm_client,
        model=cfg.llm_model,
    )

    # ------------------------------------------------------------------
    # 14. Entity extraction
    # ------------------------------------------------------------------
    ensure_default_dictionaries(resolved_dict_path)
    dictionaries = load_entity_dictionaries(resolved_dict_path)
    entity_mentions = extract_entities(chunks, meta, dictionaries)

    # ------------------------------------------------------------------
    # 15. Retrieval views (multiple focused representations)
    # ------------------------------------------------------------------
    retrieval_views: dict = {}
    if cfg.enable_retrieval_views:
        retrieval_views = _build_retrieval_views(
            meta=meta,
            description_cleaned=description_cleaned,
            primary_attachment_text=primary_attachment_text,
            comments_cleaned=comments_enriched["comments_cleaned"],
            chunks=chunks,
        )

    retrieval_text = _build_retrieval_text(
        summary=meta["summary"],
        description_cleaned=description_cleaned,
        primary_attachment_text=primary_attachment_text,
        retrieval_views=retrieval_views,
    )

    # ------------------------------------------------------------------
    # 16. Embeddings
    # ------------------------------------------------------------------
    texts_to_embed: list[str] = [summary_text]
    texts_to_embed.extend(c["text"] for c in chunks)
    texts_to_embed.extend(c["text"] for c in section_chunks)
    texts_to_embed.append(meta["metadata_text"])

    if embedding_client is not None:
        embeddings = embed_batch(texts_to_embed, embedding_client, model=cfg.embedding_model)
    else:
        logger.warning("No embedding client — embeddings will be empty lists")
        embeddings = [[] for _ in texts_to_embed]

    idx = 0
    summary_embedding = embeddings[idx]; idx += 1
    for c in chunks:
        c["embedding"] = embeddings[idx]; idx += 1
    for c in section_chunks:
        c["embedding"] = embeddings[idx]; idx += 1
    metadata_embedding = embeddings[idx]

    # ------------------------------------------------------------------
    # 17. Trainability
    # ------------------------------------------------------------------
    has_gold_vs = bool(classified_links.get("vs", []))
    has_gold_products = bool(product_fields["impacted_products"]["names"])
    avg_confidence = (
        sum(c.get("extraction_confidence", 1.0) for c in chunks) / max(len(chunks), 1)
    )
    source_quality_score = TIER_WEIGHTS[quality_tier] * avg_confidence

    # ------------------------------------------------------------------
    # 18. Provenance
    # ------------------------------------------------------------------
    provenance = {
        "quality_tier": quality_tier,
        "has_description": desc_class not in ("empty", "junk"),
        "has_attachments": len(attachments) > 0,
        "has_primary_attachment": primary is not None,
        "has_theme_links": has_gold_vs,
        "has_product_labels": has_gold_products,
        "has_it_product_labels": bool(product_fields["impacted_it_products"]["names"]),
        "source_quality_score": round(source_quality_score, 4),
        "primary_evidence_type": content_source or ("description" if desc_class in ("rich", "usable") else "none"),
        "primary_evidence_name": primary["filename"] if primary else None,
        "selection_reason": triage_artifact.get("selection_reason", ""),
        "extraction_provenance": "native" if content_source in ("pptx", "ppt", "docx", "doc") else (
            "ocr_or_native" if content_source == "pdf" else (
                "description" if desc_class in ("rich", "usable") else "none"
            )
        ),
    }

    # ------------------------------------------------------------------
    # 19. Stats
    # ------------------------------------------------------------------
    now = datetime.now(timezone.utc).isoformat()
    non_bp_slides = len([c for c in chunks if c["source"] in ("pptx_slide", "pdf_page") and not c.get("is_boilerplate")])
    stats = {
        "chunk_count": len(chunks),
        "section_count": len(section_chunks),
        "non_boilerplate_slides": non_bp_slides,
        "total_word_count": sum(c.get("word_count", len(c["text"].split())) for c in chunks),
        "has_tables": any("table" in c.get("source", "") for c in chunks),
        "has_speaker_notes": any(c.get("has_notes") for c in chunks),
        "table_count": len([c for c in chunks if "table" in c.get("source", "")]),
        "entity_mention_count": sum(len(v) for v in entity_mentions.values()),
        "comment_count": comments_enriched["comment_count"],
        "substantive_comment_count": comments_enriched["substantive_count"],
    }

    # ------------------------------------------------------------------
    # 19b. Derived layer — LLM structured outputs + entity signal fallback
    # ------------------------------------------------------------------
    derived = generate_derived_artifacts(
        quality_tier=quality_tier,
        chunks=all_for_summary,
        metadata=meta,
        llm_client=llm_client,
        model=cfg.llm_model,
    )

    # Always populate ticket_summary_llm from the heuristic summary if LLM
    # didn't produce one (ensures the field is never empty when there is content).
    if not derived["ticket_summary_llm"] and summary_text:
        derived["ticket_summary_llm"] = summary_text

    # Augment keyword/product/entity fields from entity_mentions when the LLM
    # did not populate them.  This gives useful signal even in no-LLM mode.
    if not derived["capability_keywords_llm"]:
        derived["capability_keywords_llm"] = [
            m["term"] for m in entity_mentions.get("capabilities", [])
        ][:8]
    if not derived["product_mentions_llm"]:
        derived["product_mentions_llm"] = [
            m["term"] for m in entity_mentions.get("products", [])
        ][:8]
    if not derived["business_entities_llm"]:
        # Merge components + business_unit + product_area as proxy org entities
        ent: list[str] = []
        bu = meta.get("business_unit", "")
        pa = meta.get("product_area", "")
        if bu:
            ent.append(bu)
        if pa:
            ent.append(pa)
        ent.extend(meta.get("components", []))
        derived["business_entities_llm"] = ent[:8]

    # Use ticket_summary_llm as the primary summary when LLM produced it but
    # heuristic summary was empty.
    if derived["ticket_summary_llm"] and not summary_text:
        summary_text = derived["ticket_summary_llm"]

    # ------------------------------------------------------------------
    # 20. Value stream supervision labels — structured
    # ------------------------------------------------------------------
    vs_links = classified_links.get("vs", [])
    vs_names = [lnk["summary"] for lnk in vs_links]
    vs_ids = [lnk["key"] for lnk in vs_links]
    vs_statuses = [lnk.get("status", "") for lnk in vs_links]

    # ------------------------------------------------------------------
    # 21. Assemble final document (v3.0)
    # ------------------------------------------------------------------
    return {
        "ticket_key": ticket_key,
        "schema_version": "3.0",
        "ingested_at": now,

        # ── RAW LAYER ─────────────────────────────────────────────────
        "raw": {
            "description": raw_description,
            "comments": comments_enriched,
            "issue_links": fields.get("issuelinks", []),
            "attachment_inventory": attachment_inventory,
        },

        # ── OBSERVED (RETRIEVAL) LAYER ─────────────────────────────────
        "observed": {
            "quality_tier": quality_tier,
            "created": meta.get("created", ""),
            "updated_at": fields.get("updated", ""),
            "content_source": content_source or "none",
            "provenance": provenance,
            "triage": triage_artifact,
            "description_class": desc_class,
            "description_cleaned": description_cleaned,
            "primary_attachment_text": primary_attachment_text[:5000] if primary_attachment_text else "",
            "comments_cleaned": comments_enriched["comments_cleaned"],
            "retrieval_views": retrieval_views,
            "retrieval_text": retrieval_text,
            "summary_text": summary_text,
            "summary_embedding": summary_embedding,
            "chunks": chunks,
            "section_chunks": section_chunks,
            "entity_mentions": entity_mentions,
            "metadata": meta,
            "metadata_text": meta["metadata_text"],
            "metadata_embedding": metadata_embedding,
            "stats": stats,
        },

        # ── SUPERVISION LAYER ─────────────────────────────────────────
        "supervision": {
            # Value stream labels — strongest supervision signal
            "vs_labels": vs_names,                   # backward compat
            "vs_label_source": "jira_issuelinks",
            "linked_value_stream_ids": vs_ids,
            "linked_value_stream_names": vs_names,
            "linked_value_stream_statuses": vs_statuses,
            "theme_links_raw": vs_links,
            # Product labels
            "impacted_products": product_fields["impacted_products"],
            "impacted_it_products": product_fields["impacted_it_products"],
            "product_stage_labels": product_stage_labels,
            # Trainability
            "trainability": {
                "has_gold_vs_labels": has_gold_vs,
                "has_gold_product_labels": has_gold_products,
                "is_trainable_for_vs": has_gold_vs and quality_tier in ("A", "B", "C"),
                "is_trainable_for_product": has_gold_products and quality_tier in ("A", "B", "C"),
                "source_quality_score": round(source_quality_score, 4),
                "label_snapshot_time": now,
            },
        },

        # ── DERIVED LAYER (LLM) ────────────────────────────────────────
        "derived": derived,
    }


# ---------------------------------------------------------------------------
# Retrieval view builder
# ---------------------------------------------------------------------------

# Section heading patterns for each view type
_PROBLEM_PATTERNS = re.compile(
    r"\b(problem|challenge|background|why now|pain point|issue|gap|opportunity|need|current state)\b",
    re.IGNORECASE,
)
_SOLUTION_PATTERNS = re.compile(
    r"\b(solution|approach|proposal|capability|what we.?re building|recommendation|initiative|scope|deliverable)\b",
    re.IGNORECASE,
)
_VALUE_PATTERNS = re.compile(
    r"\b(value|benefit|roi|metric|kpi|success criteri|outcome|impact|saving|cost|revenue|efficiency)\b",
    re.IGNORECASE,
)


def _build_retrieval_views(
    meta: dict,
    description_cleaned: str,
    primary_attachment_text: str,
    comments_cleaned: list[str],
    chunks: list[dict],
) -> dict:
    """
    Build multiple focused retrieval views from ticket content.

    Views are derived dynamically — no hardcoded per-ticket keywords.
    Supervision labels are never included in any view.
    """
    summary = meta.get("summary", "")
    components = ", ".join(meta.get("components", []))
    labels = ", ".join(meta.get("labels", []))
    business_unit = meta.get("business_unit", "")
    product_area = meta.get("product_area", "")

    # Overview: summary + key metadata context
    overview_parts = [summary]
    if business_unit:
        overview_parts.append(f"Business Unit: {business_unit}")
    if product_area:
        overview_parts.append(f"Product Area: {product_area}")
    if components:
        overview_parts.append(f"Components: {components}")
    if description_cleaned:
        overview_parts.append(description_cleaned[:600])
    overview = ". ".join(p for p in overview_parts if p)

    # Classify chunk texts into view buckets
    problem_texts: list[str] = []
    solution_texts: list[str] = []
    value_texts: list[str] = []
    attachment_texts: list[str] = []

    for chunk in chunks:
        text = chunk.get("text", "")
        source = chunk.get("source", "")
        if not text or chunk.get("is_boilerplate"):
            continue

        section_title = (chunk.get("section_title") or "").lower()
        probe = section_title + " " + text[:200]

        if source in ("pptx_slide", "pdf_page", "supplementary"):
            attachment_texts.append(text)

        if _PROBLEM_PATTERNS.search(probe):
            problem_texts.append(text)
        elif _SOLUTION_PATTERNS.search(probe):
            solution_texts.append(text)
        elif _VALUE_PATTERNS.search(probe):
            value_texts.append(text)

    # Fall back to description for views that matched nothing
    if not problem_texts and description_cleaned:
        problem_texts.append(description_cleaned)
    if not solution_texts and description_cleaned:
        solution_texts.append(description_cleaned[:800])

    # Add comments to problem/solution as lower-priority signals
    for c in comments_cleaned[:2]:
        if _PROBLEM_PATTERNS.search(c[:200]):
            problem_texts.append(c[:500])
        elif _SOLUTION_PATTERNS.search(c[:200]):
            solution_texts.append(c[:500])

    def _join(texts: list[str], limit: int = 1200) -> str:
        joined = " ".join(texts)
        return joined[:limit] if joined else ""

    return {
        "overview": overview[:1200],
        "problem_objective": _join(problem_texts),
        "solution_capability": _join(solution_texts),
        "value_proposition": _join(value_texts),
        "attachment_focused": _join(attachment_texts, limit=2000),
    }


def _build_retrieval_text(
    summary: str,
    description_cleaned: str,
    primary_attachment_text: str,
    retrieval_views: dict,
) -> str:
    """
    Build a single combined retrieval string for embedding.
    Priority: summary > attachment text > description > views.
    No supervision labels included.
    """
    parts: list[str] = []
    if summary:
        parts.append(summary)
    if primary_attachment_text:
        parts.append(primary_attachment_text[:2000])
    elif description_cleaned:
        parts.append(description_cleaned[:1500])
    overview = retrieval_views.get("overview", "")
    if overview and overview not in " ".join(parts):
        parts.append(overview[:600])
    return " ".join(p.strip() for p in parts if p.strip())[:4000]


# ---------------------------------------------------------------------------
# Attachment inventory builder
# ---------------------------------------------------------------------------

def _build_attachment_inventory(
    attachments: list[dict],
    all_scored: list[dict],
    primary: Optional[dict],
    supplementary: list[dict],
) -> list[dict]:
    """
    Build a per-attachment inventory record for every attachment on the ticket.
    Persisted as 02_attachment_contents.json.
    """
    scored_by_id: dict[str, dict] = {}
    for att in all_scored:
        att_id = str(att.get("id") or att.get("filename", ""))
        scored_by_id[att_id] = att

    primary_id = str(primary.get("id") or primary.get("filename", "")) if primary else None
    supplementary_ids = {str(s.get("id") or s.get("filename", "")) for s in supplementary}

    inventory: list[dict] = []
    for att in attachments:
        att_id = str(att.get("id") or att.get("filename", ""))
        scored = scored_by_id.get(att_id, {})
        filename = att.get("filename", "")
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        from .triage import KILL_EXTENSIONS, EXTRACTABLE_EXTENSIONS
        if ext in KILL_EXTENSIONS:
            status = "filtered"
        elif ext not in EXTRACTABLE_EXTENSIONS:
            status = "filtered"
        elif att.get("size", 0) < 15_000:
            status = "filtered"
        else:
            status = "extracted" if att_id == primary_id else (
                "supplementary" if att_id in supplementary_ids else "skipped"
            )

        from .triage import _compute_att_scores
        scores = _compute_att_scores(scored) if scored else {
            "extraction_quality": 0.0,
            "semantic_density": 0.0,
            "idea_card_likeness": 0.0,
            "retrieval_readiness": 0.0,
        }

        inventory.append({
            "attachment_id": att_id,
            "filename": filename,
            "mime_type": att.get("mimeType", ""),
            "size": att.get("size", 0),
            "created_at": att.get("created", ""),
            "is_primary": att_id == primary_id,
            "is_supplementary": att_id in supplementary_ids,
            "extraction_status": status,
            "extraction_method": ext if status not in ("filtered", "skipped") else "none",
            "extraction_confidence": scores["extraction_quality"],
            "raw_text": "",         # populated later by fetch_attachment_content
            "cleaned_text": "",
            "text_length": 0,
            "text_preview": "",
            "triage_score": scored.get("triage_score"),
            "triage_reasons": scored.get("triage_reasons", []),
            "scores": scores,
        })

    return inventory


# ---------------------------------------------------------------------------
# Pre-chunk artifact + debug report helpers
# ---------------------------------------------------------------------------

def _build_prechunk_artifact(document: dict) -> dict:
    """
    Build the 04_assembled_prechunk.json artifact from the assembled document.
    Summarises the pre-chunk state so it can be inspected before chunking.
    """
    obs = document["observed"]
    sup = document["supervision"]
    raw = document["raw"]
    return {
        "ticket_key": document["ticket_key"],
        "summary": obs["metadata"]["summary"],
        "description_raw": raw.get("description", ""),
        "description_cleaned": obs.get("description_cleaned", ""),
        "description_class": obs.get("description_class", ""),
        "triage": obs["triage"],
        "primary_attachment_text": obs.get("primary_attachment_text", "")[:3000],
        "supplementary_previews": [
            {"filename": s, "preview": ""}
            for s in obs["triage"].get("supplementary_attachments", [])
        ],
        "linked_themes": sup.get("theme_links_raw", []),
        "labels": obs["metadata"].get("labels", []),
        "components": obs["metadata"].get("components", []),
        "org_metadata": {
            "business_unit": obs["metadata"].get("business_unit", ""),
            "product_area": obs["metadata"].get("product_area", ""),
            "requesting_org": obs["metadata"].get("requesting_org", ""),
            "delivery_org": obs["metadata"].get("delivery_org", ""),
        },
        "comments_enriched": raw.get("comments", {}),
        "retrieval_views": obs.get("retrieval_views", {}),
        "retrieval_text": obs.get("retrieval_text", ""),
        "provenance": obs.get("provenance", {}),
    }


def _build_debug_report(document: dict) -> dict:
    """Build 05_debug_report.json — compact summary of all pipeline decisions."""
    obs = document["observed"]
    sup = document["supervision"]
    stats = obs["stats"]
    prov = obs.get("provenance", {})
    return {
        "ticket_key": document["ticket_key"],
        "schema_version": document["schema_version"],
        "quality_tier": obs["quality_tier"],
        "content_source": obs["content_source"],
        "description_class": obs.get("description_class", ""),
        "triage_summary": {
            "primary": obs["triage"].get("primary_attachment"),
            "att_quality": obs["triage"].get("att_quality"),
            "quality_tier": obs["triage"].get("quality_tier"),
            "scores": obs["triage"].get("scores", {}),
        },
        "provenance": prov,
        "stats": stats,
        "supervision_summary": {
            "vs_label_count": len(sup.get("linked_value_stream_names", [])),
            "vs_names": sup.get("linked_value_stream_names", []),
            "impacted_product_count": len(sup["impacted_products"].get("names", [])),
            "impacted_it_product_count": len(sup["impacted_it_products"].get("names", [])),
            "is_trainable_for_vs": sup["trainability"]["is_trainable_for_vs"],
        },
        "retrieval_view_lengths": {
            k: len(v) for k, v in obs.get("retrieval_views", {}).items()
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
    try:
        return download_fn(att)
    except Exception as exc:
        logger.warning("Download failed for %s: %s", att.get("filename"), exc)
        return None
