"""
jira_ingestion.platform — Operational and runtime utilities.

These are internal engineering concerns (retry, telemetry, dead-letter queue)
that are useful to maintainers and operators but are not part of the main
framework contract for normal callers.

Usage
-----
Retry with backoff:

    from jira_ingestion.platform import RetryPolicy, retry_async, retry_sync

    policy = RetryPolicy(max_attempts=4, backoff_base=2.0)
    result = await retry_async(my_async_fn, ticket_key, policy=policy)

Dead-letter queue (unprocessable tickets):

    from jira_ingestion.platform import DeadLetterQueue

    dlq = DeadLetterQueue("output/dlq.jsonl")
    dlq.push("IDEA-1234", "fetch", error="HTTP 429", raw_payload={}, trigger="webhook")
    print(dlq.summary())

Structured telemetry (OpenTelemetry-ready):

    from jira_ingestion.platform import record_ingest, record_skip, timed_stage

    with timed_stage("triage") as timer:
        primary, supp, quality = triage_attachments(...)
    record_ingest("IDEA-1234", stage="triage", elapsed_ms=timer.elapsed_ms)

Error taxonomy:

    from jira_ingestion.platform import (
        JiraIngestionError,
        TicketFetchError,
        TicketNotFoundError,
        AttachmentDownloadError,
        is_transient,
        is_permanent,
        TRANSIENT_ERRORS,
        PERMANENT_ERRORS,
    )
"""

from jira_ingestion.retries import (
    RetryPolicy,
    retry_async,
    retry_sync,
)
from jira_ingestion.telemetry import (
    IngestionEvent,
    record_ingest,
    record_skip,
    record_failure,
    timed_stage,
)
from jira_ingestion.dlq import DeadLetterQueue
from jira_ingestion.errors import (
    JiraIngestionError,
    AuthenticationError,
    TicketFetchError,
    TicketNotFoundError,
    AttachmentDownloadError,
    AttachmentExtractionError,
    MetadataExtractionError,
    SchemaValidationError,
    IndexingError,
    StorageError,
    is_transient,
    is_permanent,
    TRANSIENT_ERRORS,
    PERMANENT_ERRORS,
)

__all__ = [
    # Retry
    "RetryPolicy",
    "retry_async",
    "retry_sync",
    # Telemetry
    "IngestionEvent",
    "record_ingest",
    "record_skip",
    "record_failure",
    "timed_stage",
    # Dead-letter queue
    "DeadLetterQueue",
    # Error taxonomy
    "JiraIngestionError",
    "AuthenticationError",
    "TicketFetchError",
    "TicketNotFoundError",
    "AttachmentDownloadError",
    "AttachmentExtractionError",
    "MetadataExtractionError",
    "SchemaValidationError",
    "IndexingError",
    "StorageError",
    "is_transient",
    "is_permanent",
    "TRANSIENT_ERRORS",
    "PERMANENT_ERRORS",
]
