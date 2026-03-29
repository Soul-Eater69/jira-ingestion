from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib3
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Silence noisy warnings
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings(
    "ignore",
    message="Couldn't find ffmpeg or avconv.*",
    category=RuntimeWarning,
)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from jira_ingestion import (
    JiraIngestionConfig,
    JiraValueStreamClient,
    create_indexes,
    ingest_ticket,
)
from src.clients.embedding import EmbeddingClient
from src.clients.llm import IDPChatOpenAI
from src.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    JIRA_BASE_URL,
    JIRA_TOKEN,
)

# ---------------------------------------------------------------------------
# Ticket list / runtime knobs
# ---------------------------------------------------------------------------

TICKETS: List[str] = list(
    dict.fromkeys(
        [
            "IDMT-1320",
            "IDMT-4125",
            "IDMT-4124",
            "IDMT-1403",
            "IDMT-19761",
            "IDMT-23229",
        ]
    )
)

OUTPUT_DIR = Path(os.environ.get("TICKET_CHUNKS_DIR", "ticket_chunks"))
VERIFY_SSL = os.environ.get("VERIFY_SSL", "false").lower() == "true"
FORCE_REPROCESS = os.environ.get("FORCE_REPROCESS", "true").lower() == "true"
MAX_CONCURRENT = int(os.environ.get("BATCH_MAX_CONCURRENT", "1"))
ENABLE_LLM = os.environ.get("ENABLE_LLM", "true").lower() == "true"
ENABLE_EMBEDDINGS = os.environ.get("ENABLE_EMBEDDINGS", "true").lower() == "true"


# ---------------------------------------------------------------------------
# OpenAI-shaped adapters
# ---------------------------------------------------------------------------

class _Message:
    __slots__ = ("content",)

    def __init__(self, content: str):
        self.content = content


class _Choice:
    __slots__ = ("message",)

    def __init__(self, message: _Message):
        self.message = message


class _ChatResponse:
    __slots__ = ("choices",)

    def __init__(self, choices: List[_Choice]):
        self.choices = choices


class _CompletionsAPI:
    def __init__(self, llm: IDPChatOpenAI):
        self._llm = llm

    def create(
        self,
        *,
        model: str,
        messages: List[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> _ChatResponse:
        self._llm.model_name = model
        if temperature is not None:
            self._llm.temperature = temperature
        invoke_kwargs: Dict[str, Any] = {}
        if max_tokens is not None:
            invoke_kwargs["max_completion_tokens"] = max_tokens
        result = self._llm.invoke(messages, **invoke_kwargs)
        return _ChatResponse(choices=[_Choice(_Message(result.content))])


class _ChatAPI:
    __slots__ = ("completions",)

    def __init__(self, completions: _CompletionsAPI):
        self.completions = completions


class OpenAICompatibleLLM:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.chat = _ChatAPI(_CompletionsAPI(IDPChatOpenAI(model=model)))


class _EmbeddingData:
    __slots__ = ("index", "embedding")

    def __init__(self, index: int, embedding: List[float]):
        self.index = index
        self.embedding = embedding


class _EmbeddingResponse:
    __slots__ = ("data",)

    def __init__(self, data: List[_EmbeddingData]):
        self.data = data


class _EmbeddingsAPI:
    __slots__ = ("_client", "_model")

    def __init__(self, client: EmbeddingClient, model: str):
        self._client = client
        self._model = model

    def create(
        self,
        *,
        input: str | List[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> _EmbeddingResponse:
        texts = [input] if isinstance(input, str) else list(input)
        model_name = model or self._model
        vectors = self._client.embed_many(texts, model=model_name)
        return _EmbeddingResponse(
            data=[_EmbeddingData(index=i, embedding=v) for i, v in enumerate(vectors)]
        )


class OpenAICompatibleEmbeddingClient:
    __slots__ = ("embeddings",)

    def __init__(self, model: str = EMBEDDING_MODEL, dimension: int = EMBEDDING_DIMENSION):
        client = EmbeddingClient(model=model, dimension=dimension)
        self.embeddings = _EmbeddingsAPI(client, model=model)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Azure-ready record builders
# ---------------------------------------------------------------------------

def _attachment_references(chunk: dict) -> List[dict]:
    attachment_id = str(chunk.get("attachment_id") or "")
    attachment_name = str(chunk.get("attachment_name") or "")
    if not attachment_id and not attachment_name:
        return []

    source_type = str(chunk.get("source") or "")
    if source_type.startswith("supplementary_"):
        category = "supplementary"
    elif source_type in {"pdf_page", "pptx_slide", "docx_section", "section"}:
        category = "primary"
    else:
        category = "ticket"

    return [
        {
            "attachmentId": attachment_id,
            "attachmentName": attachment_name,
            "attachmentType": str(chunk.get("attachment_type") or ""),
            "category": category,
        }
    ]


def _content_type(chunk: dict) -> str:
    source = str(chunk.get("source") or "")
    if source == "section":
        return "section"
    if source == "docx_section":
        return "section"
    if source in {"pdf_page", "pptx_slide"}:
        return "page"
    if source.startswith("supplementary_"):
        return "supplementary"
    if source == "description":
        return "description"
    if source == "comment":
        return "comment"
    return source or "unknown"


def _resolve_page_range(chunk: dict) -> Optional[List[int]]:
    if chunk.get("page_range"):
        return chunk["page_range"]
    if chunk.get("source") == "pdf_page" and chunk.get("page_num") is not None:
        p = chunk["page_num"]
        return [p, p]
    return None


def _resolve_slide_range(chunk: dict) -> Optional[List[int]]:
    if chunk.get("slide_range"):
        return chunk["slide_range"]
    if chunk.get("source") == "pptx_slide" and chunk.get("slide_num") is not None:
        s = chunk["slide_num"]
        return [s, s]
    return None


def build_chunk_record(chunk: dict, ticket_id: str, obs: dict, meta: dict) -> dict:
    source_url = f"{JIRA_BASE_URL.rstrip('/')}/browse/{ticket_id}" if JIRA_BASE_URL else ticket_id
    content = str(chunk.get("text") or "")

    return {
        "id": str(chunk.get("chunk_uid") or chunk.get("chunk_id") or ""),
        "content": content,
        "content_vector": chunk.get("embedding") or None,
        "dataSource": "jira",
        "sourceId": ticket_id,
        "sourceURL": source_url,
        "title": str(meta.get("title") or meta.get("summary") or ticket_id),
        "contentType": _content_type(chunk),
        "headerHierarchy": str(chunk.get("header_hierarchy") or chunk.get("section_title") or ""),
        "tokenCount": int(chunk.get("token_count") or chunk.get("word_count") or len(content.split())),
        "project": ticket_id.split("-")[0] if "-" in ticket_id else "",
        "issueType": str(meta.get("issue_type") or ""),
        "status": str(meta.get("status") or ""),
        "priority": str(meta.get("priority") or ""),
        "reporter": str(meta.get("reporter") or ""),
        "createdDate": str(obs.get("created") or ""),
        "updatedDate": str(obs.get("updated_at") or ""),
        "contextKeywords": chunk.get("context_keywords") or [],
        "attachmentReferences": _attachment_references(chunk),
        "chunkProvenance": {
            "chunkId": str(chunk.get("chunk_id") or ""),
            "chunkIndex": int(chunk.get("chunk_index") or 0),
            "sourceType": str(chunk.get("source") or ""),
            "attachmentId": str(chunk.get("attachment_id") or ""),
            "attachmentName": str(chunk.get("attachment_name") or ""),
            "attachmentType": str(chunk.get("attachment_type") or ""),
            "pageRange": _resolve_page_range(chunk),
            "slideRange": _resolve_slide_range(chunk),
            "extractionMethod": str(chunk.get("extraction_method") or ""),
            "extractionConfidence": chunk.get("extraction_confidence"),
        },
    }


def build_valuestream_record(ticket_id: str, result: dict) -> dict:
    sup = result.get("supervision", {})
    obs = result.get("observed", {})
    meta = obs.get("metadata", {})

    impacted_products = sup.get("impacted_products", {}) if isinstance(sup.get("impacted_products"), dict) else {}
    impacted_it_products = (
        sup.get("impacted_it_products", {}) if isinstance(sup.get("impacted_it_products"), dict) else {}
    )

    return {
        "id": ticket_id,
        "ticketId": ticket_id,
        "project": ticket_id.split("-")[0] if "-" in ticket_id else "",
        "title": str(meta.get("title") or meta.get("summary") or ticket_id),
        "valueStreamIds": sup.get("linked_value_stream_ids", []) or [],
        "valueStreamNames": sup.get("linked_value_stream_names", []) or [],
        "valueStreamStatuses": sup.get("linked_value_stream_statuses", []) or [],
        "impactedProductIds": impacted_products.get("ids", []) or [],
        "impactedProductNames": impacted_products.get("names", []) or [],
        "impactedItProductIds": impacted_it_products.get("ids", []) or [],
        "impactedItProductNames": impacted_it_products.get("names", []) or [],
        "labelSource": str(sup.get("vs_label_source") or ""),
        "updatedDate": str(obs.get("updated_at") or ""),
    }


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def _create_memory_indexes() -> Tuple[Any, Any, Any, Any]:
    try:
        return create_indexes(
            backend="memory",
            metadata_store_path=None,
            supervision_store_path=None,
        )
    except TypeError:
        return create_indexes(backend="memory")


def _try_build_llm(model: str = "gpt-4o-mini") -> OpenAICompatibleLLM | None:
    if not ENABLE_LLM:
        logger.info("LLM disabled by ENABLE_LLM=false")
        return None
    try:
        return OpenAICompatibleLLM(model=model)
    except Exception as exc:
        logger.warning("LLM unavailable (%s) - continuing without LLM", exc)
        return None


def _try_build_embedding_client() -> OpenAICompatibleEmbeddingClient | None:
    if not ENABLE_EMBEDDINGS:
        logger.info("Embeddings disabled by ENABLE_EMBEDDINGS=false")
        return None
    try:
        return OpenAICompatibleEmbeddingClient(
            model=EMBEDDING_MODEL,
            dimension=EMBEDDING_DIMENSION,
        )
    except Exception as exc:
        logger.warning("Embedding client unavailable (%s) - continuing without embeddings", exc)
        return None


def _set_if_present(cfg: JiraIngestionConfig, name: str, value: Any) -> None:
    if hasattr(cfg, name):
        setattr(cfg, name, value)
    else:
        logger.info("Config field not present in this repo version: %s", name)


def build_config() -> JiraIngestionConfig:
    cfg = JiraIngestionConfig(
        llm_model="gpt-5-mini-idp",
        embedding_model=EMBEDDING_MODEL,

        # verify/production hybrid: no local pipeline artifacts
        enable_raw_artifact_persistence=False,
        enable_attachment_text_persistence=False,
        enable_debug_stage_persistence=False,
        enable_prechunk_persistence=False,

        # keep useful output layers
        enable_attachment_inventory=True,
        enable_retrieval_views=True,

        # production default
        skip_llm_summary=True,
        skip_llm_keywords=False,
        skip_llm_derived=True,
    )

    # Multi-doc safe knobs
    _set_if_present(cfg, "section_only_chunks", False)
    _set_if_present(cfg, "include_section_rollups_in_retrieval", False)
    _set_if_present(cfg, "section_min_slides", 1)
    _set_if_present(cfg, "max_prefetch_attachments", None)
    _set_if_present(cfg, "max_chunk_attachments", 10)

    return cfg


def _config_snapshot(cfg: JiraIngestionConfig) -> dict:
    keys = [
        "section_only_chunks",
        "include_section_rollups_in_retrieval",
        "section_min_slides",
        "max_prefetch_attachments",
        "max_chunk_attachments",
        "skip_llm_summary",
        "skip_llm_keywords",
        "skip_llm_derived",
        "enable_attachment_inventory",
        "enable_retrieval_views",
    ]
    snap = {}
    for k in keys:
        if hasattr(cfg, k):
            snap[k] = getattr(cfg, k)
    return snap


def _observed_attachment_refs(obs: dict) -> List[Tuple[str, str]]:
    return sorted({
        (
            str(c.get("attachment_id") or ""),
            str(c.get("attachment_name") or ""),
        )
        for c in (obs.get("chunks", []) or [])
        if c.get("attachment_id") or c.get("attachment_name")
    })


def _observed_chunk_type_counts(obs: dict) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for c in (obs.get("chunks", []) or []):
        k = str(c.get("source") or "unknown")
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[0]))


def _json_attachment_chunk_counts(chunk_records: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rec in chunk_records:
        refs = rec.get("attachmentReferences") or []
        if refs:
            name = str(refs[0].get("attachmentName") or refs[0].get("attachmentId") or "unknown")
        else:
            name = "<ticket_only>"
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

async def process_ticket(
    ticket_id: str,
    client: JiraValueStreamClient,
    coarse_index: Any,
    fine_index: Any,
    metadata_index: Any,
    supervision_store: Any,
    cfg: JiraIngestionConfig,
    llm_client: Any,
    embedding_client: Any,
) -> Tuple[List[dict], dict]:
    ticket_dir = OUTPUT_DIR / ticket_id
    ticket_dir.mkdir(parents=True, exist_ok=True)

    chunks_file = ticket_dir / "07_chunks.json"
    vs_file = ticket_dir / "08_valuestream_map.json"

    if not FORCE_REPROCESS and chunks_file.exists() and vs_file.exists():
        logger.info("Skipping %s (already completed)", ticket_id)
        chunks = json.loads(chunks_file.read_text(encoding="utf-8")).get("chunks", [])
        vs_map = json.loads(vs_file.read_text(encoding="utf-8"))
        return chunks, vs_map

    logger.info("Processing %s ...", ticket_id)

    result = await ingest_ticket(
        ticket_key=ticket_id,
        jira_client=client,
        coarse_index=coarse_index,
        fine_index=fine_index,
        metadata_index=metadata_index,
        supervision_store=supervision_store,
        config=cfg,
        llm_client=llm_client,
        embedding_client=embedding_client,
        storage_dir=None,
    )

    obs = result.get("observed", {})
    meta = obs.get("metadata", {})
    chunk_records = [
        build_chunk_record(c, ticket_id, obs, meta)
        for c in obs.get("chunks", [])
    ]

    logger.info("%s CONFIG %s", ticket_id, _config_snapshot(cfg))
    logger.info(
        "%s OBSERVED chunks=%d attachment_refs=%s",
        ticket_id,
        len(obs.get("chunks", []) or []),
        _observed_attachment_refs(obs),
    )
    logger.info(
        "%s OBSERVED source_type_counts=%s",
        ticket_id,
        _observed_chunk_type_counts(obs),
    )
    logger.info(
        "%s JSON attachment_chunk_counts=%s",
        ticket_id,
        _json_attachment_chunk_counts(chunk_records),
    )
    logger.info(
        "%s EMBEDDINGS populated=%d/%d",
        ticket_id,
        sum(1 for r in chunk_records if r.get("content_vector") is not None),
        len(chunk_records),
    )

    dump_json(
        chunks_file,
        {
            "ticket_id": ticket_id,
            "mapped_value_stream_ids": result.get("supervision", {}).get("linked_value_stream_ids", []) or [],
            "mapped_value_stream_names": result.get("supervision", {}).get("linked_value_stream_names", []) or [],
            "chunk_count": len(chunk_records),
            "chunks": chunk_records,
        },
    )

    vs_record = build_valuestream_record(ticket_id, result)
    dump_json(vs_file, vs_record)

    logger.info(
        "%s DONE chunks=%d vs_links=%d products=%d",
        ticket_id,
        len(chunk_records),
        len(vs_record["valueStreamIds"]),
        len(vs_record["impactedProductIds"]),
    )
    return chunk_records, vs_record


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = build_config()
    coarse, fine, metadata_index, supervision_store = _create_memory_indexes()
    llm_client = _try_build_llm()
    embedding_client = _try_build_embedding_client()

    logger.info("OUTPUT_DIR=%s", OUTPUT_DIR)
    logger.info(
        "FORCE_REPROCESS=%s MAX_CONCURRENT=%s VERIFY_SSL=%s ENABLE_LLM=%s ENABLE_EMBEDDINGS=%s",
        FORCE_REPROCESS,
        MAX_CONCURRENT,
        VERIFY_SSL,
        ENABLE_LLM,
        ENABLE_EMBEDDINGS,
    )
    logger.info("GLOBAL CONFIG %s", _config_snapshot(cfg))

    all_chunks: List[dict] = []
    all_vs_maps: List[dict] = []
    errors: List[dict] = []

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def _guarded(
        ticket_id: str,
        client: JiraValueStreamClient,
    ) -> Tuple[str, Optional[List[dict]], Optional[dict], Optional[str]]:
        async with sem:
            try:
                chunks, vs_map = await process_ticket(
                    ticket_id,
                    client,
                    coarse,
                    fine,
                    metadata_index,
                    supervision_store,
                    cfg,
                    llm_client,
                    embedding_client,
                )
                return ticket_id, chunks, vs_map, None
            except Exception as exc:
                logger.exception("Ticket %s failed", ticket_id)
                return ticket_id, None, None, str(exc)

    async with JiraValueStreamClient(
        base_url=JIRA_BASE_URL,
        token=JIRA_TOKEN,
        verify_ssl=VERIFY_SSL,
    ) as client:
        results = await asyncio.gather(*[_guarded(tid, client) for tid in TICKETS])

    for ticket_id, chunks, vs_map, err in results:
        if err:
            errors.append({"ticket_id": ticket_id, "error": err})
            dump_json(OUTPUT_DIR / f"{ticket_id}__ERROR.json", {"ticket_id": ticket_id, "error": err})
        else:
            all_chunks.extend(chunks or [])
            if vs_map is not None:
                all_vs_maps.append(vs_map)

    dump_json(
        OUTPUT_DIR / "_all_chunks.json",
        {
            "total_chunks": len(all_chunks),
            "tickets_processed": len(all_vs_maps),
            "tickets_failed": len(errors),
            "chunks": all_chunks,
        },
    )

    dump_json(
        OUTPUT_DIR / "_all_valuestream_maps.json",
        {
            "total_tickets": len(all_vs_maps),
            "tickets_with_vs_links": sum(1 for m in all_vs_maps if m["valueStreamIds"]),
            "tickets_with_products": sum(1 for m in all_vs_maps if m["impactedProductIds"]),
            "maps": all_vs_maps,
        },
    )

    if errors:
        dump_json(OUTPUT_DIR / "_errors.json", errors)

    print("\n" + "=" * 80)
    print(f"BATCH COMPLETE - {len(all_vs_maps)} succeeded, {len(errors)} failed")
    print(f"Chunk index records: {OUTPUT_DIR / '_all_chunks.json'}")
    print(f"VS map records:      {OUTPUT_DIR / '_all_valuestream_maps.json'}")
    for m in all_vs_maps:
        print(
            f"  {m['ticketId']}: "
            f"{OUTPUT_DIR / m['ticketId'] / '07_chunks.json'}, "
            f"{OUTPUT_DIR / m['ticketId'] / '08_valuestream_map.json'}"
        )
    if errors:
        print("Errors:")
        for e in errors:
            print(f"  {e['ticket_id']}: {e['error'][:160]}")


if __name__ == "__main__":
    asyncio.run(main())
