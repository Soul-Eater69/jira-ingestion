"""
Dead-letter queue (DLQ) for the Jira ingestion pipeline.

Failed ingestion attempts are written to a local JSONL file so they can be
inspected, replayed, or escalated without manual log scraping.

Usage:
    from jira_ingestion.dlq import DeadLetterQueue

    dlq = DeadLetterQueue(path="output/failed.jsonl")
    dlq.push(ticket_key="IDEA-1234", stage="attachment_download",
              error=exc, raw_payload=ticket_data)

    for entry in dlq.iter_all():
        print(entry["ticket_key"], entry["error_type"])

    # Replay all failed tickets
    failed_keys = [e["ticket_key"] for e in dlq.iter_all()]
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DLQ_PATH = "output/failed.jsonl"


class DeadLetterQueue:
    """
    Append-only JSONL dead-letter store.

    Each entry records:
      - ticket_key
      - stage         which pipeline stage failed
      - error_type    exception class name
      - error_message exception message
      - timestamp     ISO 8601 UTC
      - raw_payload   optional raw ticket data for replay (may be large)
      - extra         arbitrary additional context dict
    """

    def __init__(self, path: str = _DEFAULT_DLQ_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def push(
        self,
        ticket_key: str,
        stage: str,
        error: Exception,
        raw_payload: Optional[dict] = None,
        trigger: str = "unknown",
        **extra: Any,
    ) -> None:
        """
        Write a failed-ingestion entry to the DLQ.

        Args:
            ticket_key:  Jira ticket key.
            stage:       Pipeline stage name (e.g. 'attachment_download').
            error:       The exception that caused the failure.
            raw_payload: Optional raw ticket payload for replay purposes.
            trigger:     'webhook' | 'backfill' | 'manual'.
            **extra:     Any additional context (attachment filenames, etc.).
        """
        entry: dict[str, Any] = {
            "ticket_key": ticket_key,
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger": trigger,
        }
        if raw_payload is not None:
            # Store only lightweight ticket identity fields to avoid huge DLQ files.
            entry["raw_summary"] = {
                "key": raw_payload.get("key", ticket_key),
                "fields_keys": list(raw_payload.get("fields", {}).keys()),
                "attachment_count": len(raw_payload.get("attachments", [])),
            }
        entry.update(extra)

        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            logger.warning(
                "DLQ: wrote failed entry for %s at stage=%s (%s)",
                ticket_key,
                stage,
                type(error).__name__,
            )
        except OSError as write_err:
            # DLQ write failure must not mask the original error
            logger.error("DLQ: failed to write entry for %s: %s", ticket_key, write_err)

    def iter_all(self) -> Iterator[dict]:
        """Yield all DLQ entries in order of insertion."""
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("DLQ: skipping malformed entry: %s", exc)

    def count(self) -> int:
        """Return the number of DLQ entries."""
        return sum(1 for _ in self.iter_all())

    def keys(self) -> list[str]:
        """Return all failed ticket keys (may contain duplicates for retried tickets)."""
        return [e["ticket_key"] for e in self.iter_all()]

    def clear(self) -> None:
        """
        Remove all DLQ entries.

        Only call this after confirming all failed tickets have been
        successfully replayed.
        """
        if self.path.exists():
            self.path.unlink()
            logger.info("DLQ: cleared %s", self.path)

    def summary(self) -> dict:
        """Return a brief summary of DLQ state."""
        entries = list(self.iter_all())
        by_stage: dict[str, int] = {}
        by_error: dict[str, int] = {}
        for entry in entries:
            by_stage[entry.get("stage", "unknown")] = (
                by_stage.get(entry.get("stage", "unknown"), 0) + 1
            )
            by_error[entry.get("error_type", "unknown")] = (
                by_error.get(entry.get("error_type", "unknown"), 0) + 1
            )
        return {
            "path": str(self.path),
            "total_entries": len(entries),
            "by_stage": by_stage,
            "by_error_type": by_error,
        }
