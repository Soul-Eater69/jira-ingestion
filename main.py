"""
CLI entrypoint — mirrors the demo script shown in the architecture image,
extended to run the full ingestion pipeline.

Usage:
    # Quick demo (just fetch and print ticket data)
    python main.py demo IDEA-1234

    # Full ingestion (fetch, extract, embed, index)
    python main.py ingest IDEA-1234

    # Batch ingestion from a file of ticket keys
    python main.py batch keys.txt
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Demo mode (matches the image script exactly)
# ---------------------------------------------------------------------------

async def demo(ticket_id: str) -> None:
    """Fetch ticket data and print a human-readable summary."""
    from src.clients.jira.value_stream_client import JiraValueStreamClient
    from src.config import JIRA_BASE_URL, JIRA_TOKEN, JIRA_VERIFY_SSL

    jira_vs_client = JiraValueStreamClient(JIRA_BASE_URL, JIRA_TOKEN, verify_ssl=JIRA_VERIFY_SSL)
    await jira_vs_client.authenticate()

    data = await jira_vs_client.get_ticket_data(ticket_id)

    print("Themes:")
    for theme in data["themes"]:
        print(f"  {theme['key']}: {theme['summary']} [{theme['status']}]")

    print(f"\nAttachments ({len(data['attachments'])}):")
    for att in data["attachments"]:
        print(f"  {att.get('filename')}  ({att.get('mimeType')})  -> {att.get('content')}")

    # --- Fetch and extract text from all attachments ---
    print("\nExtracting attachment content via MarkItDown...")
    contents = await jira_vs_client.fetch_attachment_content(data["attachments"])
    for item in contents:
        print(f"\n=== {item['filename']} ===")
        if item["error"]:
            print(f"  ERROR: {item['error']}")
        else:
            preview = item["text_content"][:500].replace("\n", " ")
            print(f"  {preview}...")

    await jira_vs_client.close()


# ---------------------------------------------------------------------------
# Full ingestion mode
# ---------------------------------------------------------------------------

async def ingest(
    ticket_id: str,
    output_file: Optional[str] = None,
    storage_dir: Optional[str] = None,
    storage_fmt: str = "json",
) -> dict:
    """Run the full ingestion pipeline for a single ticket."""
    from src.clients.jira.value_stream_client import JiraValueStreamClient
    from src.config import (
        JIRA_BASE_URL, JIRA_TOKEN, JIRA_VERIFY_SSL,
        OPENAI_API_KEY, INDEX_BACKEND,
    )
    from src.ingestion.pipeline import ingest_ticket
    from src.ingestion.indexing import create_indexes

    # Build clients
    jira_client = JiraValueStreamClient(JIRA_BASE_URL, JIRA_TOKEN, verify_ssl=JIRA_VERIFY_SSL)
    await jira_client.authenticate()

    llm_client = None
    embedding_client = None
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            llm_client = OpenAI(api_key=OPENAI_API_KEY)
            embedding_client = llm_client
        except ImportError:
            logger.warning("openai package not installed — running without LLM/embeddings")

    coarse, fine, meta_idx, supervision = create_indexes(backend=INDEX_BACKEND)

    # Default storage dir to output/documents if not given via --output-file
    resolved_storage_dir = storage_dir or (
        os.path.dirname(output_file) if output_file else "output/documents"
    )

    try:
        document = await ingest_ticket(
            ticket_key=ticket_id,
            jira_client=jira_client,
            coarse_index=coarse,
            fine_index=fine,
            metadata_index=meta_idx,
            supervision_store=supervision,
            llm_client=llm_client,
            embedding_client=embedding_client,
            trigger="cli",
            storage_dir=resolved_storage_dir,
            storage_fmt=storage_fmt,
        )
    finally:
        await jira_client.close()

    obs = document["observed"]
    sup = document["supervision"]
    print(f"\n{'='*60}")
    print(f"Ticket:      {document['ticket_key']}")
    print(f"Quality:     Tier {obs['quality_tier']}")
    print(f"Source:      {obs['content_source']}")
    print(f"Chunks:      {obs['stats']['chunk_count']} content, {obs['stats']['section_count']} sections")
    print(f"Entities:    {obs['stats']['entity_mention_count']} mentions")
    print(f"Trainable:   {sup['trainability']['is_trainable_for_vs']}")
    print(f"VS labels:   {sup['vs_labels']}")
    print(f"\nSummary preview:\n{obs['summary_text'][:400]}...")
    print(f"{'='*60}\n")

    if output_file:
        # Strip embeddings from output for readability
        doc_out = _strip_embeddings(document)
        with open(output_file, "w", encoding="utf-8") as fh:
            json.dump(doc_out, fh, indent=2, ensure_ascii=False)
        print(f"Full document saved to: {output_file}")

    return document


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

async def batch(
    keys_file: str,
    storage_dir: Optional[str] = None,
    storage_fmt: str = "jsonl",
) -> None:
    """Ingest multiple tickets from a newline-separated keys file."""
    with open(keys_file, "r", encoding="utf-8") as fh:
        keys = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

    print(f"Ingesting {len(keys)} tickets → format={storage_fmt} dir={storage_dir or 'output/documents'}...")
    for key in keys:
        try:
            await ingest(key, storage_dir=storage_dir, storage_fmt=storage_fmt)
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", key, exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_embeddings(obj: Any) -> Any:
    """Recursively remove embedding vectors from a document for clean JSON output."""
    if isinstance(obj, dict):
        return {
            k: (_strip_embeddings(v) if not k.endswith("embedding") else f"<vector len={len(v)}>")
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_strip_embeddings(i) for i in obj]
    return obj


from typing import Any  # noqa: E402


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Jira ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode")

    p_demo = sub.add_parser("demo", help="Fetch and print ticket data (no indexing)")
    p_demo.add_argument("ticket_id")

    p_ingest = sub.add_parser("ingest", help="Full ingestion for a single ticket")
    p_ingest.add_argument("ticket_id")
    p_ingest.add_argument("--output-dir", default="output/documents",
                          help="Directory to save the document (default: output/documents)")
    p_ingest.add_argument("--fmt", default="json", choices=["json", "jsonl", "parquet"],
                          help="Storage format (default: json)")
    p_ingest.add_argument("--output-file", default=None,
                          help="Deprecated: use --output-dir instead")

    p_batch = sub.add_parser("batch", help="Ingest multiple tickets from a keys file")
    p_batch.add_argument("keys_file", help="Newline-separated list of ticket keys")
    p_batch.add_argument("--output-dir", default="output/documents",
                         help="Directory to save documents (default: output/documents)")
    p_batch.add_argument("--fmt", default="jsonl", choices=["json", "jsonl", "parquet"],
                         help="Storage format (default: jsonl for batch)")

    ns = parser.parse_args()

    if ns.mode == "demo":
        asyncio.run(demo(ns.ticket_id))
    elif ns.mode == "ingest":
        asyncio.run(ingest(
            ns.ticket_id,
            output_file=ns.output_file,
            storage_dir=ns.output_dir,
            storage_fmt=ns.fmt,
        ))
    elif ns.mode == "batch":
        asyncio.run(batch(ns.keys_file, storage_dir=ns.output_dir, storage_fmt=ns.fmt))
    else:
        parser.print_help()
        sys.exit(1)
