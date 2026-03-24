"""
Retry / backoff utilities for the Jira ingestion pipeline.

Centralises retry logic so each module does not implement its own ad hoc
sleep-and-retry loop.  Separates retry policy from business logic.

Usage:
    from jira_ingestion.retries import retry_async, retry_sync, RetryPolicy

    policy = RetryPolicy(max_attempts=3, backoff_base=1.5)

    # Async
    result = await retry_async(my_async_fn, arg1, arg2, policy=policy)

    # Sync
    result = retry_sync(my_sync_fn, arg1, policy=policy)
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional, Sequence, Type, TypeVar

from jira_ingestion.errors import is_transient

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """
    Configures retry behaviour for a callable.

    Attributes:
        max_attempts:       Total attempts (1 = no retry).
        backoff_base:       Base seconds for exponential backoff.
        backoff_max:        Cap on sleep between retries.
        jitter:             If True, add random jitter to each sleep.
        retryable_errors:   Exception types to retry on. None = retry on any.
        non_retryable:      Exception types to NEVER retry (overrides retryable_errors).
    """
    max_attempts: int = 3
    backoff_base: float = 1.5
    backoff_max: float = 30.0
    jitter: bool = True
    retryable_errors: Optional[tuple[Type[Exception], ...]] = None
    non_retryable: tuple[Type[Exception], ...] = field(
        default_factory=lambda: ()
    )

    def sleep_for(self, attempt: int) -> float:
        """Compute sleep duration for attempt N (0-indexed)."""
        delay = min(self.backoff_base ** attempt, self.backoff_max)
        if self.jitter:
            delay *= (0.5 + random.random() * 0.5)
        return delay

    def should_retry(self, exc: Exception) -> bool:
        """Return True if this exception warrants a retry attempt."""
        if isinstance(exc, self.non_retryable):
            return False
        if self.retryable_errors is not None:
            return isinstance(exc, self.retryable_errors)
        # Default: retry on transient pipeline errors and generic transient signals
        return is_transient(exc) or _is_http_transient(exc)


async def retry_async(
    fn: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    policy: Optional[RetryPolicy] = None,
    **kwargs: Any,
) -> T:
    """
    Call an async callable with retry/backoff per the given policy.

    Args:
        fn:     Async callable to invoke.
        *args:  Positional arguments forwarded to fn.
        policy: RetryPolicy to use. Defaults to RetryPolicy() (3 attempts, 1.5× backoff).
        **kwargs: Keyword arguments forwarded to fn.

    Returns:
        Return value of fn on success.

    Raises:
        The last exception if all attempts fail.
    """
    p = policy or RetryPolicy()
    last_exc: Exception = RuntimeError("retry_async: no attempts made")

    for attempt in range(p.max_attempts):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= p.max_attempts or not p.should_retry(exc):
                logger.error(
                    "retry_async: giving up after %d attempt(s) on %s: %s",
                    attempt + 1,
                    getattr(fn, "__name__", repr(fn)),
                    exc,
                )
                raise

            sleep_s = p.sleep_for(attempt)
            logger.warning(
                "retry_async: attempt %d/%d failed (%s: %s) — retrying in %.1fs",
                attempt + 1,
                p.max_attempts,
                type(exc).__name__,
                exc,
                sleep_s,
            )
            await asyncio.sleep(sleep_s)

    raise last_exc  # unreachable but satisfies type checker


def retry_sync(
    fn: Callable[..., T],
    *args: Any,
    policy: Optional[RetryPolicy] = None,
    **kwargs: Any,
) -> T:
    """
    Call a sync callable with retry/backoff per the given policy.

    Args:
        fn:     Sync callable to invoke.
        *args:  Positional arguments forwarded to fn.
        policy: RetryPolicy to use. Defaults to RetryPolicy() (3 attempts, 1.5× backoff).
        **kwargs: Keyword arguments forwarded to fn.

    Returns:
        Return value of fn on success.

    Raises:
        The last exception if all attempts fail.
    """
    p = policy or RetryPolicy()
    last_exc: Exception = RuntimeError("retry_sync: no attempts made")

    for attempt in range(p.max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= p.max_attempts or not p.should_retry(exc):
                logger.error(
                    "retry_sync: giving up after %d attempt(s) on %s: %s",
                    attempt + 1,
                    getattr(fn, "__name__", repr(fn)),
                    exc,
                )
                raise

            sleep_s = p.sleep_for(attempt)
            logger.warning(
                "retry_sync: attempt %d/%d failed (%s: %s) — retrying in %.1fs",
                attempt + 1,
                p.max_attempts,
                type(exc).__name__,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)

    raise last_exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_http_transient(exc: Exception) -> bool:
    """Return True for HTTP status codes that are safe to retry."""
    try:
        import httpx  # type: ignore
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in (429, 500, 502, 503, 504)
    except ImportError:
        pass
    return False
