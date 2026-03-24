"""
Error taxonomy for the Jira ingestion pipeline.

Distinguishing error types enables:
  - targeted retry logic (transient vs permanent)
  - structured dead-letter handling
  - clear operator messaging

Usage:
    from jira_ingestion.errors import TicketFetchError, AttachmentDownloadError

    raise TicketFetchError("IDEA-1234", cause=exc)
"""

from __future__ import annotations

from typing import Optional


class JiraIngestionError(Exception):
    """Base class for all pipeline errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        base = super().__str__()
        if self.cause:
            return f"{base} — caused by {type(self.cause).__name__}: {self.cause}"
        return base


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class AuthenticationError(JiraIngestionError):
    """Jira authentication failed or session expired."""


# ---------------------------------------------------------------------------
# Ticket fetch
# ---------------------------------------------------------------------------

class TicketFetchError(JiraIngestionError):
    """Failed to retrieve a ticket from Jira."""

    def __init__(self, ticket_key: str, cause: Optional[Exception] = None) -> None:
        super().__init__(f"Failed to fetch ticket {ticket_key!r}", cause=cause)
        self.ticket_key = ticket_key


class TicketNotFoundError(TicketFetchError):
    """Ticket key does not exist in Jira (HTTP 404)."""


# ---------------------------------------------------------------------------
# Attachment handling
# ---------------------------------------------------------------------------

class AttachmentDownloadError(JiraIngestionError):
    """Failed to download an attachment from Jira."""

    def __init__(
        self,
        filename: str,
        ticket_key: str = "",
        cause: Optional[Exception] = None,
    ) -> None:
        msg = f"Failed to download attachment {filename!r}"
        if ticket_key:
            msg += f" for ticket {ticket_key!r}"
        super().__init__(msg, cause=cause)
        self.filename = filename
        self.ticket_key = ticket_key


class AttachmentExtractionError(JiraIngestionError):
    """Failed to extract text content from an attachment."""

    def __init__(
        self,
        filename: str,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(f"Text extraction failed for {filename!r}", cause=cause)
        self.filename = filename


# ---------------------------------------------------------------------------
# Metadata / schema
# ---------------------------------------------------------------------------

class MetadataExtractionError(JiraIngestionError):
    """Failed during metadata extraction from ticket fields."""

    def __init__(self, ticket_key: str, cause: Optional[Exception] = None) -> None:
        super().__init__(f"Metadata extraction failed for {ticket_key!r}", cause=cause)
        self.ticket_key = ticket_key


class SchemaValidationError(JiraIngestionError):
    """Ticket payload or document failed schema validation."""

    def __init__(self, detail: str, cause: Optional[Exception] = None) -> None:
        super().__init__(f"Schema validation failed: {detail}", cause=cause)
        self.detail = detail


# ---------------------------------------------------------------------------
# Indexing / storage
# ---------------------------------------------------------------------------

class IndexingError(JiraIngestionError):
    """Failed to write to a retrieval or supervision index."""

    def __init__(
        self,
        index_name: str,
        ticket_key: str = "",
        cause: Optional[Exception] = None,
    ) -> None:
        msg = f"Indexing failed for index {index_name!r}"
        if ticket_key:
            msg += f" (ticket {ticket_key!r})"
        super().__init__(msg, cause=cause)
        self.index_name = index_name
        self.ticket_key = ticket_key


class StorageError(JiraIngestionError):
    """Failed to persist a document or artifact to disk."""


# ---------------------------------------------------------------------------
# Retryability classification
# ---------------------------------------------------------------------------

#: Errors that are safe to retry without manual intervention.
TRANSIENT_ERRORS: tuple[type[JiraIngestionError], ...] = (
    TicketFetchError,
    AttachmentDownloadError,
    IndexingError,
)

#: Errors that will not resolve with retry — require operator action or data fix.
PERMANENT_ERRORS: tuple[type[JiraIngestionError], ...] = (
    AuthenticationError,
    TicketNotFoundError,
    SchemaValidationError,
)


def is_transient(exc: Exception) -> bool:
    """Return True if the exception is likely safe to retry."""
    return isinstance(exc, TRANSIENT_ERRORS)


def is_permanent(exc: Exception) -> bool:
    """Return True if the exception will not resolve without intervention."""
    return isinstance(exc, PERMANENT_ERRORS)
