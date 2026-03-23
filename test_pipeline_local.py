#!/usr/bin/env python3
"""
Test the full ingestion pipeline locally with a file on disk.

Usage:
    python test_pipeline_local.py path/to/file.pptx
    python test_pipeline_local.py path/to/file.pdf --ticket-key TEST-42
    python test_pipeline_local.py path/to/file.docx --output-dir output/test

Runs extraction → metadata → triage → chunking → quality → description →
entities → summary.  Skips LLM and embeddings (no API keys needed).
Writes the assembled document to a local JSON file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def build_fake_ticket_data(
    file_path: Path,
    ticket_key: str,
) -> dict:
    """Build a synthetic ticket_data dict that mirrors Jira's API shape."""
    file_bytes = file_path.read_bytes()
    filename = file_path.name
    ext = file_path.suffix.lstrip(".").lower()
    size = len(file_bytes)

    # Guess MIME type
    mime_map = {
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "ppt": "application/vnd.ms-powerpoint",
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "doc": "application/msword",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "csv": "text/csv",
    }
    mime_type = mime_map.get(ext, "application/octet-stream")

    attachment = {
        "id": "local-1",
        "filename": filename,
        "mimeType": mime_type,
        "size": size,
        "content": f"file://{file_path.resolve()}",  # pseudo-URL
        "created": "2026-01-01T00:00:00.000+0000",
        "author": {"displayName": "Local Test"},
        # stash bytes so triage/extraction can use them without downloading
        "_local_bytes": file_bytes,
        "ext": ext,
    }

    return {
        "key": ticket_key,
        "attachments": [attachment],
        "themes": [],
        "fields": {
            "summary": file_path.stem.replace("_", " ").replace("-", " "),
            "description": "",
            "reporter": {"displayName": "Local Test"},
            "created": "2026-01-01T00:00:00.000+0000",
            "updated": "2026-01-01T00:00:00.000+0000",
            "labels": [],
            "components": [],
            "priority": {"name": "Medium"},
            "issuelinks": [],
            "comment": {"comments": []},
            "attachment": [attachment],
        },
    }


def local_download(att: dict) -> bytes:
    """Download function that reads from disk instead of Jira."""
    if "_local_bytes" in att:
        return att["_local_bytes"]

    content = att.get("content", "")
    if content.startswith("file://"):
        path = content[len("file://"):]
        return Path(path).read_bytes()

    raise RuntimeError(f"Cannot download: {content}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test ingestion pipeline with a local file")
    parser.add_argument("file", type=Path, help="Path to PPTX / PDF / DOCX file")
    parser.add_argument("--ticket-key", default="TEST-1", help="Fake ticket key (default: TEST-1)")
    parser.add_argument("--output-dir", default="output/test", help="Output directory for JSON")
    args = parser.parse_args()

    if not args.file.is_file():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    ext = args.file.suffix.lstrip(".").lower()
    if ext not in ("pptx", "ppt", "pdf", "docx", "doc", "xlsx", "csv"):
        print(f"Unsupported file type: .{ext}", file=sys.stderr)
        sys.exit(1)

    logger.info("Building fake ticket data from %s", args.file)
    ticket_data = build_fake_ticket_data(args.file, args.ticket_key)

    logger.info("Running assemble_document pipeline ...")
    from jira_ingestion.ingestion.pipeline import assemble_document

    document = assemble_document(
        ticket_data=ticket_data,
        download_fn=local_download,
        llm_client=None,
        embedding_client=None,
        dict_path=None,
    )

    # Save via DocumentStore
    from jira_ingestion.ingestion.storage import DocumentStore

    store = DocumentStore(output_dir=args.output_dir, save_embeddings=False)
    out_path = store.save(document, fmt="json")
    logger.info("Document saved to %s", out_path)

    # Print summary
    obs = document["observed"]
    sup = document["supervision"]
    print()
    print(f"  Ticket:          {document['ticket_key']}")
    print(f"  Quality tier:    {obs['quality_tier']}")
    print(f"  Content source:  {obs['content_source']}")
    print(f"  Chunks:          {obs['stats']['chunk_count']}")
    print(f"  Sections:        {obs['stats']['section_count']}")
    print(f"  Words:           {obs['stats']['total_word_count']}")
    print(f"  Entities:        {obs['stats']['entity_mention_count']}")
    print(f"  Summary:         {obs['summary_text'][:120]}...")
    print(f"  VS trainable:    {sup['trainability']['is_trainable_for_vs']}")
    print(f"  Quality score:   {sup['trainability']['source_quality_score']}")
    print(f"  Output:          {out_path}")
    print()


if __name__ == "__main__":
    main()
