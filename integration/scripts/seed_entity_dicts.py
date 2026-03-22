"""
Seed entity dictionaries from IDP-VALUE-STREAMS data sources.

Run once from the IDP-VALUE-STREAMS root directory:
    python jira-ingestion/integration/scripts/seed_entity_dicts.py

What it does:
  1. Reads value_stream_index.json → populates capabilities.json
  2. Reads src/static/idea_cards/*.pptx filenames → hints for products.json
  3. Existing products/systems/platforms/business_units.json are left as-is
     unless you pass --overwrite.

After running, re-ingest tickets to pick up the new entity matches.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths (relative to IDP-VALUE-STREAMS root)
# ---------------------------------------------------------------------------

VS_INDEX_PATH = Path("value_stream_index.json")
IDEA_CARDS_DIR = Path("src/static/idea_cards")
ENTITY_DICT_DIR = Path("jira-ingestion/data/entity_dicts")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | list:
    if not path.exists():
        print(f"  [skip] {path} not found")
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, data: list) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data)} entries → {path}")


def _slug_to_aliases(name: str) -> list[str]:
    """Generate lowercase and hyphenated aliases from a canonical name."""
    lower = name.lower()
    slugged = re.sub(r"\s+", "-", lower)
    words = lower.split()
    aliases = list({lower, slugged})
    # Add acronym if multi-word (e.g. "Customer Onboarding" → "CO")
    if len(words) > 1:
        acronym = "".join(w[0] for w in words).upper()
        aliases.append(acronym)
    return aliases


# ---------------------------------------------------------------------------
# Step 1: capabilities from value_stream_index.json
# ---------------------------------------------------------------------------

def seed_capabilities(overwrite: bool) -> None:
    out_path = ENTITY_DICT_DIR / "capabilities.json"
    if out_path.exists() and not overwrite:
        existing = json.loads(out_path.read_text())
        if existing:
            print(f"  [skip] capabilities.json already has {len(existing)} entries (use --overwrite)")
            return

    vs_index = load_json(VS_INDEX_PATH)
    if not vs_index:
        return

    capabilities: list[dict] = []

    # Try common schema shapes — adjust the key names to match your actual JSON
    items = (
        vs_index.get("value_streams")
        or vs_index.get("streams")
        or vs_index.get("stages")
        or (vs_index if isinstance(vs_index, list) else [])
    )

    for item in items:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = (
                item.get("name")
                or item.get("stage_name")
                or item.get("value_stream")
                or item.get("title")
                or ""
            )
        else:
            continue

        if not name:
            continue

        aliases = _slug_to_aliases(name)
        # Also add words from a description field if present
        if isinstance(item, dict):
            desc = item.get("description") or item.get("summary") or ""
            if desc:
                key_words = [
                    w for w in re.findall(r"[a-z]{4,}", desc.lower())
                    if w not in {"that", "this", "with", "from", "have", "been", "will"}
                ]
                aliases.extend(key_words[:4])

        capabilities.append({
            "canonical": name,
            "aliases": list(dict.fromkeys(aliases)),  # deduplicate, preserve order
        })

    save_json(out_path, capabilities)


# ---------------------------------------------------------------------------
# Step 2: product hints from idea card filenames
# ---------------------------------------------------------------------------

def seed_products_from_idea_cards(overwrite: bool) -> None:
    """
    Extracts product/initiative names from PPTX filenames in idea_cards/.
    These are added as hints — they won't override existing entries.
    """
    out_path = ENTITY_DICT_DIR / "products.json"
    existing: list[dict] = json.loads(out_path.read_text()) if out_path.exists() else []
    existing_canonicals = {e["canonical"].lower() for e in existing}

    if not IDEA_CARDS_DIR.exists():
        print(f"  [skip] {IDEA_CARDS_DIR} not found")
        return

    new_entries: list[dict] = []
    for pptx_path in sorted(IDEA_CARDS_DIR.glob("*.pptx")):
        stem = pptx_path.stem  # e.g. "IDMT-19761"
        # Strip ticket-key prefix (e.g. "IDMT-19761_Fraud_Detection" → "Fraud Detection")
        clean = re.sub(r"^[A-Z]+-\d+[_\-\s]*", "", stem).replace("_", " ").strip()
        if not clean or clean.lower() in existing_canonicals:
            continue
        new_entries.append({
            "canonical": clean,
            "aliases": _slug_to_aliases(clean),
        })
        existing_canonicals.add(clean.lower())

    if new_entries:
        merged = existing + new_entries
        save_json(out_path, merged)
    else:
        print(f"  [skip] No new products found from idea card filenames")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing non-empty dictionaries")
    ns = parser.parse_args()

    ENTITY_DICT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n── Seeding capabilities from value_stream_index.json ──")
    seed_capabilities(ns.overwrite)

    print("\n── Seeding product hints from idea card filenames ──")
    seed_products_from_idea_cards(ns.overwrite)

    print("\nDone. Re-run ingestion to pick up new entity matches.")


if __name__ == "__main__":
    main()
