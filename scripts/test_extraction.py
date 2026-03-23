"""
Standalone extraction test — fetch one ticket, run the full extraction
pipeline, and save the result to a local JSON file.

No vector DB, no OpenAI required (embeddings will be empty lists).

Usage (from jira-ingestion/):
    python scripts/test_extraction.py IDMT-1925
    python scripts/test_extraction.py IDMT-1925 --output output/test_IDMT-1925.json
    python scripts/test_extraction.py IDMT-1925 --no-embed   # skip embedding API call
    python scripts/test_extraction.py IDMT-1925 --verbose    # print each pipeline step
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# make src/ importable when run from jira-ingestion/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("test_extraction")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_embeddings(obj):
    """Replace embedding vectors with a placeholder for readable output."""
    if isinstance(obj, dict):
        return {
            k: (f"<vector len={len(v)}>" if k.endswith("embedding") and isinstance(v, list) and v else _strip_embeddings(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_strip_embeddings(i) for i in obj]
    return obj


def _print_summary(doc: dict) -> None:
    obs = doc["observed"]
    sup = doc["supervision"]
    stats = obs["stats"]

    print(f"\n{'='*60}")
    print(f"Ticket:       {doc['ticket_key']}")
    print(f"Quality tier: {obs['quality_tier']}")
    print(f"Content src:  {obs['content_source']}")
    print(f"Chunks:       {stats['chunk_count']} content  |  {stats['section_count']} section groups")
    print(f"Words:        {stats['total_word_count']}")
    print(f"Tables:       {stats['has_tables']}  speaker notes: {stats['has_speaker_notes']}")
    print(f"Entities:     {stats['entity_mention_count']} mentions")
    print(f"Trainable:    {sup['trainability']['is_trainable_for_vs']}")
    print(f"VS labels:    {sup['vs_labels']}")
    print(f"\nSummary:\n{obs['summary_text'][:500]}...")

    triage = obs.get("triage", {})
    print(f"\nTriage:")
    print(f"  primary:         {triage.get('primary_attachment')}")
    print(f"  triage_score:    {triage.get('triage_score')}")
    print(f"  supplementary:   {triage.get('supplementary_attachments', [])}")

    em = obs.get("entity_mentions", {})
    for etype, mentions in em.items():
        if mentions:
            terms = [m["term"] for m in mentions[:5]]
            print(f"\nEntities [{etype}]: {terms}{'...' if len(mentions) > 5 else ''}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(ticket_id: str, output: str, no_embed: bool, verbose: bool) -> None:
    from src.clients.jira.value_stream_client import JiraValueStreamClient
    from src.config import JIRA_BASE_URL, JIRA_TOKEN, JIRA_VERIFY_SSL, OPENAI_API_KEY
    from src.ingestion.pipeline import assemble_document

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── 1. Fetch from Jira ─────────────────────────────────────────────────
    logger.info("Connecting to Jira at %s", JIRA_BASE_URL)
    jira = JiraValueStreamClient(JIRA_BASE_URL, JIRA_TOKEN, verify_ssl=JIRA_VERIFY_SSL)
    await jira.authenticate()

    logger.info("Fetching %s ...", ticket_id)
    ticket_data = await jira.get_ticket_data(ticket_id)
    await jira.close()

    logger.info(
        "  → %d attachment(s)  |  themes: %d",
        len(ticket_data.get("attachments", [])),
        len(ticket_data.get("themes", [])),
    )

    # ── 2. Build download function ─────────────────────────────────────────
    # We already closed the Jira client above, so download attachments now
    # and cache the bytes so the pipeline doesn't need a live connection.
    attachments_with_bytes: list[dict] = []
    for att in ticket_data.get("attachments", []):
        try:
            jira2 = JiraValueStreamClient(JIRA_BASE_URL, JIRA_TOKEN, verify_ssl=JIRA_VERIFY_SSL)
            await jira2.authenticate()
            raw = await jira2.download_attachment(att)
            await jira2.close()
            attachments_with_bytes.append({**att, "file_bytes": raw})
            logger.info("  downloaded %s (%d bytes)", att.get("filename"), len(raw))
        except Exception as exc:
            logger.warning("  skip %s — %s", att.get("filename"), exc)
            attachments_with_bytes.append(att)

    ticket_data["attachments"] = attachments_with_bytes

    def download_fn(att: dict) -> bytes:
        return att.get("file_bytes") or b""

    # ── 3. Optional: OpenAI embedding + LLM ───────────────────────────────
    llm_client = None
    embedding_client = None

    if not no_embed and OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            llm_client = client
            embedding_client = client
            logger.info("OpenAI client ready (embeddings + LLM summaries enabled)")
        except ImportError:
            logger.warning("openai not installed — running without embeddings/LLM")
    else:
        logger.info("Running without embeddings/LLM (use --no-embed or unset OPENAI_API_KEY)")

    # ── 4. Run extraction pipeline ─────────────────────────────────────────
    logger.info("Running extraction pipeline ...")
    doc = assemble_document(
        ticket_data=ticket_data,
        download_fn=download_fn,
        llm_client=llm_client,
        embedding_client=embedding_client,
    )

    # ── 5. Print summary ───────────────────────────────────────────────────
    _print_summary(doc)

    # ── 6. Save to JSON ────────────────────────────────────────────────────
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc_out = _strip_embeddings(doc)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc_out, fh, indent=2, ensure_ascii=False)

    logger.info("Saved → %s  (%d bytes)", out_path, out_path.stat().st_size)
    print(f"\nOutput: {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticket_id", help="Jira ticket key, e.g. IDMT-1925")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON path (default: output/test_<ticket>.json)",
    )
    parser.add_argument(
        "--no-embed", action="store_true",
        help="Skip OpenAI embedding/LLM calls (faster, offline-safe)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging",
    )
    ns = parser.parse_args()

    output = ns.output or f"output/test_{ns.ticket_id}.json"
    asyncio.run(main(ns.ticket_id, output, ns.no_embed, ns.verbose))
