#!/usr/bin/env python3
"""
Fetch real Jira tickets, extract metadata & classifications, save to JSON.

No chunking, no embeddings, no LLM — just fetch, classify, and store.

Usage:
    export JIRA_BASE_URL=https://jira.example.com
    export JIRA_TOKEN=your-token

    python test_pipeline_local.py IDMT-123
    python test_pipeline_local.py IDMT-123 IDMT-456 IDMT-789
    python test_pipeline_local.py IDMT-123 --output-dir output/tickets
    python test_pipeline_local.py IDMT-123 --verify-ssl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


async def fetch_and_classify(
    ticket_key: str,
    jira_client: Any,
) -> dict:
    """
    Fetch a single ticket from Jira and build a classified document.

    Returns a dict with:
      - ticket_key, fetched_at
      - metadata (title, reporter, labels, components, etc.)
      - classified_links (vs, product, dependency, parent, related, etc.)
      - description_classification (empty/junk/thin/usable/rich)
      - quality_tier (A/B/C/D)
      - themes (linked issues from get_ticket_data)
      - attachments (raw metadata, no download)
      - raw_fields (full Jira fields for inspection)
    """
    from jira_ingestion.ingestion.metadata import extract_metadata, classify_links
    from jira_ingestion.ingestion.description import classify_description
    from jira_ingestion.ingestion.quality import determine_quality_tier

    # Fetch from Jira
    ticket_data = await jira_client.get_ticket_data(ticket_key)
    fields = ticket_data.get("fields", {})

    # Metadata extraction
    meta = extract_metadata(fields, ticket_key)
    meta["classified_links"] = classify_links(fields.get("issuelinks", []))

    # Description classification
    desc_class, desc_data = classify_description(fields.get("description"))

    # Attachment info (no download)
    attachments = ticket_data.get("attachments", [])
    attachment_summary = []
    for att in attachments:
        ext = ""
        filename = att.get("filename", "")
        if "." in filename:
            ext = filename.rsplit(".", 1)[-1].lower()
        attachment_summary.append({
            "id": att.get("id"),
            "filename": filename,
            "mimeType": att.get("mimeType", ""),
            "size": att.get("size", 0),
            "created": att.get("created", ""),
            "author": (att.get("author") or {}).get("displayName", ""),
            "extension": ext,
        })

    # Determine quality tier (based on attachments + description, no chunks)
    has_extractable = any(
        a["extension"] in ("pptx", "ppt", "pdf", "docx", "doc")
        for a in attachment_summary
    )
    content_source = None
    if has_extractable:
        # Pick the most likely primary extension
        for a in attachment_summary:
            if a["extension"] in ("pptx", "ppt", "pdf", "docx", "doc"):
                content_source = a["extension"]
                break

    quality_tier = determine_quality_tier(content_source, desc_class, [])

    # Themes from get_ticket_data
    themes = ticket_data.get("themes", [])

    now = datetime.now(timezone.utc).isoformat()

    return {
        "ticket_key": ticket_key,
        "fetched_at": now,
        "quality_tier": quality_tier,
        "content_source": content_source or "none",
        "description_classification": desc_class,
        "description_word_count": desc_data["word_count"] if desc_data else 0,
        "metadata": {
            "title": meta["title"],
            "summary": meta["summary"],
            "reporter": meta["reporter"],
            "created": meta["created"],
            "labels": meta["labels"],
            "components": meta["components"],
            "business_unit": meta["business_unit"],
            "product_area": meta["product_area"],
            "priority": meta["priority"],
            "epic_key": meta["epic_key"],
            "substantive_comments": meta["substantive_comments"],
            "metadata_text": meta["metadata_text"],
        },
        "classified_links": meta["classified_links"],
        "themes": themes,
        "attachments": attachment_summary,
        "attachment_count": len(attachments),
        "raw_fields": _sanitize_fields(fields),
    }


def _sanitize_fields(fields: dict) -> dict:
    """Keep raw fields but strip large binary/nested data for readability."""
    sanitized = {}
    skip_keys = {"attachment", "comment", "worklog"}
    for k, v in fields.items():
        if k in skip_keys:
            continue
        # Truncate very long strings
        if isinstance(v, str) and len(v) > 2000:
            sanitized[k] = v[:2000] + "... [truncated]"
        else:
            sanitized[k] = v
    return sanitized


async def main_async(args: argparse.Namespace) -> None:
    from jira_ingestion.clients.jira.value_stream_client import JiraValueStreamClient

    base_url = os.environ.get("JIRA_BASE_URL", "")
    token = os.environ.get("JIRA_TOKEN", "")

    if not base_url or not token:
        print("Error: Set JIRA_BASE_URL and JIRA_TOKEN environment variables.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = JiraValueStreamClient(
        base_url=base_url,
        token=token,
        verify_ssl=args.verify_ssl,
    )

    try:
        logger.info("Authenticating with Jira at %s ...", base_url)
        await client.authenticate()

        results = []
        for ticket_key in args.tickets:
            logger.info("Fetching %s ...", ticket_key)
            try:
                doc = await fetch_and_classify(ticket_key, client)
                results.append(doc)

                # Save individual ticket JSON
                ticket_path = output_dir / f"{ticket_key}.json"
                ticket_path.write_text(json.dumps(doc, indent=2, default=str))
                logger.info("Saved %s → %s", ticket_key, ticket_path)

                # Print summary
                print(f"\n  {ticket_key}")
                print(f"    Quality:     {doc['quality_tier']}")
                print(f"    Source:      {doc['content_source']}")
                print(f"    Desc class:  {doc['description_classification']} ({doc['description_word_count']} words)")
                print(f"    Attachments: {doc['attachment_count']}")
                print(f"    Themes:      {len(doc['themes'])}")
                print(f"    Links:       {sum(len(v) for v in doc['classified_links'].values())}")
                print(f"    Components:  {doc['metadata']['components']}")
                print(f"    Labels:      {doc['metadata']['labels']}")

            except Exception as exc:
                logger.error("Failed to process %s: %s", ticket_key, exc)

        # Save combined output if multiple tickets
        if len(results) > 1:
            combined_path = output_dir / "all_tickets.json"
            combined_path.write_text(json.dumps(results, indent=2, default=str))
            logger.info("Combined output → %s", combined_path)

    finally:
        await client.close()

    print(f"\nDone. {len(results)} ticket(s) saved to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Jira tickets, classify metadata, save to JSON"
    )
    parser.add_argument("tickets", nargs="+", help="Jira ticket IDs (e.g. IDMT-123 IDMT-456)")
    parser.add_argument("--output-dir", default="output/tickets", help="Output directory")
    parser.add_argument("--verify-ssl", action="store_true", default=False, help="Verify SSL certs")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
