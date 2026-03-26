"""
jira_ingestion.debug — Artifact persistence, debug reports, and inspection helpers.

This submodule exposes the storage layer and debug utilities that operators
use to inspect pipeline decisions.  Normal callers running ``ingest_ticket``
do not need to import from here directly — artifacts are written automatically
when ``storage_dir`` is passed to ``ingest_ticket``.

Direct imports are intended for:
  - Custom artifact persistence outside the main pipeline
  - Tooling that reads/inspects the numbered debug artifacts
  - Tests that need to verify what was written to disk

Usage
-----
Persist and reload documents:

    from jira_ingestion.debug import DocumentStore

    store = DocumentStore("output/documents")
    store.save(document)                       # → output/documents/IDEA-1234.json
    doc   = store.load("IDEA-1234")
    docs  = list(store.load_all())

Numbered pipeline debug artifacts (written inside ``_debug/<ticket_key>/``):

    store.save_pipeline_artifact("IDEA-1234", 1, "raw_ticket",           raw_payload)
    store.save_pipeline_artifact("IDEA-1234", 2, "attachment_contents",  att_list)
    store.save_pipeline_artifact("IDEA-1234", 3, "triage_output",        triage)
    store.save_pipeline_artifact("IDEA-1234", 4, "assembled_prechunk",   prechunk)
    store.save_pipeline_artifact("IDEA-1234", 5, "debug_report",         report)

Named stage artifacts (legacy / ad-hoc debugging):

    store.save_raw_ticket("IDEA-1234", raw_jira_payload)
    store.save_extracted_attachments("IDEA-1234", att_list)
    store.save_debug_stage("IDEA-1234", "triage", triage_dict)
"""

from jira_ingestion.ingestion.storage import (
    DocumentStore,
)

__all__ = [
    "DocumentStore",
]
