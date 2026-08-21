"""Shared classification of transient remote-API errors.

Used by grounded-facts extraction and evaluation so retry policy stays in sync.
"""

from __future__ import annotations

# Retryable: rate limits, gateway errors, timeouts. Not 4xx/500 (fail fast).
_TRANSIENT_ERROR_MARKERS: tuple[str, ...] = (
    "429",
    "rate limit",
    "resource exhausted",
    "high demand",
    "502",
    "503",
    "504",
    "bad gateway",
    "gateway timeout",
    "service unavailable",
    "unavailable",
    "timeout",
    "timed out",
    "connection error",
    "connection reset",
    "connection aborted",
    "remote end closed",
)


def is_retryable_api_error(exc: BaseException) -> bool:
    """True when ``exc`` looks like a transient remote failure worth retrying."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_ERROR_MARKERS)
