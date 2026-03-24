"""
Structured telemetry for the Jira ingestion pipeline.

Provides:
  - Counters / timers via structured log events (zero external deps)
  - A IngestionEvent dataclass that can be shipped to any sink
  - Hooks for future OpenTelemetry integration

Usage:
    from jira_ingestion.telemetry import record_ingest, record_skip, record_failure

    # On successful ingest
    record_ingest(ticket_key="IDEA-1234", tier="A", chunk_count=12,
                  ingest_ms=1340, attachment_ms=980)

    # On skip (idempotency)
    record_skip(ticket_key="IDEA-1234")

    # On failure
    record_failure(ticket_key="IDEA-1234", stage="attachment_download",
                   error_type="AttachmentDownloadError")
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------

@dataclass
class IngestionEvent:
    """
    A structured telemetry event emitted by the pipeline.

    Can be shipped to any sink: structured logs, Prometheus, Datadog,
    Azure Monitor, or OpenTelemetry collectors.
    """
    event_type: str                      # ingest | skip | failure | stage_start | stage_end
    ticket_key: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    trigger: str = "unknown"             # webhook | backfill | manual
    tier: Optional[str] = None           # A | B | C | D
    chunk_count: int = 0
    section_count: int = 0
    ingest_ms: Optional[float] = None    # total ingestion latency
    attachment_ms: Optional[float] = None
    extraction_ms: Optional[float] = None
    embedding_ms: Optional[float] = None
    indexing_ms: Optional[float] = None
    stage: Optional[str] = None          # which stage emitted this event
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Emit helpers
# ---------------------------------------------------------------------------

def record_ingest(
    ticket_key: str,
    tier: str = "",
    chunk_count: int = 0,
    section_count: int = 0,
    ingest_ms: Optional[float] = None,
    attachment_ms: Optional[float] = None,
    extraction_ms: Optional[float] = None,
    embedding_ms: Optional[float] = None,
    indexing_ms: Optional[float] = None,
    trigger: str = "unknown",
    **extra: Any,
) -> None:
    """Record a successful ticket ingestion."""
    event = IngestionEvent(
        event_type="ingest",
        ticket_key=ticket_key,
        trigger=trigger,
        tier=tier,
        chunk_count=chunk_count,
        section_count=section_count,
        ingest_ms=ingest_ms,
        attachment_ms=attachment_ms,
        extraction_ms=extraction_ms,
        embedding_ms=embedding_ms,
        indexing_ms=indexing_ms,
        extra=extra,
    )
    _emit(event)


def record_skip(ticket_key: str, reason: str = "not_modified", trigger: str = "unknown") -> None:
    """Record a ticket skip (idempotency or filter)."""
    event = IngestionEvent(
        event_type="skip",
        ticket_key=ticket_key,
        trigger=trigger,
        extra={"reason": reason},
    )
    _emit(event)


def record_failure(
    ticket_key: str,
    stage: str,
    error_type: str,
    error_message: str = "",
    trigger: str = "unknown",
    **extra: Any,
) -> None:
    """Record a pipeline failure for a specific ticket and stage."""
    event = IngestionEvent(
        event_type="failure",
        ticket_key=ticket_key,
        trigger=trigger,
        stage=stage,
        error_type=error_type,
        error_message=error_message,
        extra=extra,
    )
    _emit(event)


# ---------------------------------------------------------------------------
# Stage timer context manager
# ---------------------------------------------------------------------------

@contextmanager
def timed_stage(stage_name: str) -> Generator[dict, None, None]:
    """
    Context manager that measures wall-clock time for a pipeline stage.

    Usage:
        with timed_stage("embedding") as t:
            embeddings = embed_batch(texts, client)
        ms = t["ms"]  # elapsed milliseconds
    """
    result: dict = {"ms": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["ms"] = (time.perf_counter() - start) * 1000


# ---------------------------------------------------------------------------
# Internal emit
# ---------------------------------------------------------------------------

def _emit(event: IngestionEvent) -> None:
    """
    Emit a telemetry event as a structured log line.

    Replace or augment this function to route events to an external sink
    (Prometheus, Datadog, Azure Monitor, OpenTelemetry, etc.).
    """
    log_data: Dict[str, Any] = {
        "telemetry": True,
        "event_type": event.event_type,
        "ticket_key": event.ticket_key,
        "trigger": event.trigger,
        "timestamp": event.timestamp,
    }
    if event.tier:
        log_data["tier"] = event.tier
    if event.chunk_count:
        log_data["chunk_count"] = event.chunk_count
    if event.section_count:
        log_data["section_count"] = event.section_count
    if event.ingest_ms is not None:
        log_data["ingest_ms"] = round(event.ingest_ms, 1)
    if event.attachment_ms is not None:
        log_data["attachment_ms"] = round(event.attachment_ms, 1)
    if event.extraction_ms is not None:
        log_data["extraction_ms"] = round(event.extraction_ms, 1)
    if event.embedding_ms is not None:
        log_data["embedding_ms"] = round(event.embedding_ms, 1)
    if event.indexing_ms is not None:
        log_data["indexing_ms"] = round(event.indexing_ms, 1)
    if event.stage:
        log_data["stage"] = event.stage
    if event.error_type:
        log_data["error_type"] = event.error_type
    if event.error_message:
        log_data["error_message"] = event.error_message
    log_data.update(event.extra)

    level = logging.ERROR if event.event_type == "failure" else logging.INFO
    logger.log(level, "telemetry %s", log_data)
