"""
Tests for error taxonomy and retry utilities.

Covers:
  - Error hierarchy and is_transient / is_permanent classification
  - RetryPolicy.sleep_for backoff curve
  - RetryPolicy.should_retry classification
  - retry_sync: succeeds on retry
  - retry_sync: raises immediately on permanent error
  - retry_async: succeeds on retry
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock

from jira_ingestion.errors import (
    AuthenticationError,
    AttachmentDownloadError,
    TicketFetchError,
    TicketNotFoundError,
    SchemaValidationError,
    IndexingError,
    is_transient,
    is_permanent,
)
from jira_ingestion.retries import RetryPolicy, retry_sync, retry_async


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

class TestErrorTaxonomy:
    def test_ticket_fetch_error_is_transient(self):
        exc = TicketFetchError("IDEA-1")
        assert is_transient(exc)
        assert not is_permanent(exc)

    def test_attachment_download_error_is_transient(self):
        exc = AttachmentDownloadError("deck.pptx")
        assert is_transient(exc)

    def test_authentication_error_is_permanent(self):
        exc = AuthenticationError("bad credentials")
        assert is_permanent(exc)
        assert not is_transient(exc)

    def test_ticket_not_found_is_permanent(self):
        exc = TicketNotFoundError("IDEA-999")
        assert is_permanent(exc)

    def test_schema_validation_error_is_permanent(self):
        exc = SchemaValidationError("missing key")
        assert is_permanent(exc)

    def test_indexing_error_is_transient(self):
        exc = IndexingError("coarse", "IDEA-1")
        assert is_transient(exc)

    def test_str_includes_cause(self):
        cause = ValueError("root cause")
        exc = TicketFetchError("IDEA-1", cause=cause)
        assert "root cause" in str(exc)

    def test_generic_exception_not_transient(self):
        assert not is_transient(RuntimeError("something"))
        assert not is_permanent(RuntimeError("something"))


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------

class TestRetryPolicy:
    def test_backoff_increases(self):
        p = RetryPolicy(backoff_base=2.0, jitter=False)
        assert p.sleep_for(0) == 1.0
        assert p.sleep_for(1) == 2.0
        assert p.sleep_for(2) == 4.0

    def test_backoff_capped(self):
        p = RetryPolicy(backoff_base=10.0, backoff_max=5.0, jitter=False)
        assert p.sleep_for(5) == 5.0

    def test_should_retry_transient(self):
        p = RetryPolicy()
        assert p.should_retry(TicketFetchError("IDEA-1"))

    def test_should_not_retry_permanent(self):
        p = RetryPolicy()
        assert not p.should_retry(AuthenticationError("bad creds"))

    def test_custom_retryable_errors(self):
        p = RetryPolicy(retryable_errors=(ValueError,))
        assert p.should_retry(ValueError("recoverable"))
        assert not p.should_retry(RuntimeError("not retryable"))

    def test_non_retryable_overrides(self):
        p = RetryPolicy(non_retryable=(TicketFetchError,))
        assert not p.should_retry(TicketFetchError("IDEA-1"))


# ---------------------------------------------------------------------------
# retry_sync
# ---------------------------------------------------------------------------

class TestRetrySync:
    def test_success_on_first_try(self):
        fn = MagicMock(return_value=42)
        result = retry_sync(fn, policy=RetryPolicy(max_attempts=3))
        assert result == 42
        assert fn.call_count == 1

    def test_retries_on_transient_error(self):
        call_count = {"n": 0}

        def fn():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise TicketFetchError("IDEA-1")
            return "success"

        policy = RetryPolicy(max_attempts=3, backoff_base=0.001, jitter=False)
        result = retry_sync(fn, policy=policy)
        assert result == "success"
        assert call_count["n"] == 3

    def test_raises_after_all_attempts(self):
        def fn():
            raise TicketFetchError("IDEA-1")

        policy = RetryPolicy(max_attempts=2, backoff_base=0.001, jitter=False)
        with pytest.raises(TicketFetchError):
            retry_sync(fn, policy=policy)

    def test_does_not_retry_permanent_error(self):
        call_count = {"n": 0}

        def fn():
            call_count["n"] += 1
            raise AuthenticationError("bad creds")

        policy = RetryPolicy(max_attempts=3, backoff_base=0.001, jitter=False)
        with pytest.raises(AuthenticationError):
            retry_sync(fn, policy=policy)
        assert call_count["n"] == 1  # no retry


# ---------------------------------------------------------------------------
# retry_async
# ---------------------------------------------------------------------------

class TestRetryAsync:
    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        async def fn():
            return "ok"

        result = await retry_async(fn, policy=RetryPolicy(max_attempts=3))
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self):
        call_count = {"n": 0}

        async def fn():
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise TicketFetchError("IDEA-1")
            return "done"

        policy = RetryPolicy(max_attempts=3, backoff_base=0.001, jitter=False)
        result = await retry_async(fn, policy=policy)
        assert result == "done"
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_permanent_error(self):
        call_count = {"n": 0}

        async def fn():
            call_count["n"] += 1
            raise SchemaValidationError("bad schema")

        policy = RetryPolicy(max_attempts=3, backoff_base=0.001, jitter=False)
        with pytest.raises(SchemaValidationError):
            await retry_async(fn, policy=policy)
        assert call_count["n"] == 1
