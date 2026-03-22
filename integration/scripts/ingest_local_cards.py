"""
Bulk-ingest all PPTX files already on disk in src/static/idea_cards/
without going through the Jira API.

Run from the IDP-VALUE-STREAMS root:
    python jira-ingestion/integration/scripts/ingest_local_cards.py
    python jira-ingestion/integration/scripts/ingest_local_cards.py --output-dir output/documents --fmt jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make jira-ingestion importable when run from IDP-VALUE-STREAMS root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # jira-ingestion/
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))  # IDP-VALUE-STREAMS/src


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cards-dir", default="src/static/idea_cards",
                        help="Directory containing .pptx files (default: src/static/idea_cards)")
    parser.add_argument("--output-dir", default="output/documents",
                        help="Where to save assembled documents (default: output/documents)")
    parser.add_argument("--fmt", default="json", choices=["json", "jsonl", "parquet"],
                        help="Output format (default: json)")
    parser.add_argument("--pattern", default="*.pptx",
                        help="Glob pattern for files to ingest (default: *.pptx)")
    ns = parser.parse_args()

    cards_dir = Path(ns.cards_dir)
    files = sorted(cards_dir.glob(ns.pattern))

    if not files:
        print(f"No files matching '{ns.pattern}' found in {cards_dir}")
        sys.exit(0)

    print(f"Found {len(files)} file(s) in {cards_dir}")

    from integration.jira_ingestion_adapter import ingest_local_pptx

    results = {"ok": [], "failed": []}
    for fpath in files:
        print(f"  → {fpath.name} ...", end=" ", flush=True)
        try:
            doc = ingest_local_pptx(
                file_path=str(fpath),
                ticket_key=fpath.stem,
                output_dir=ns.output_dir,
            )
            tier = doc["observed"]["quality_tier"]
            chunks = doc["observed"]["stats"]["chunk_count"]
            print(f"OK  [tier={tier} chunks={chunks}]")
            results["ok"].append(fpath.name)
        except Exception as exc:
            print(f"FAILED — {exc}")
            results["failed"].append((fpath.name, str(exc)))

    print(f"\nDone: {len(results['ok'])} succeeded, {len(results['failed'])} failed")
    if results["failed"]:
        for name, err in results["failed"]:
            print(f"  FAILED: {name} — {err}")


if __name__ == "__main__":
    main()
