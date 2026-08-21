"""Shared classification of transient remote-API errors.

Both the grounded-facts extraction step (rag/extraction.py) and the evaluation
step (evaluation/evaluation.py) call the same Hugging Face router and need the
same notion of "this failed, but retrying may work". Keeping that logic here
means the two call sites cannot drift apart -- previously each carried its own
copy and they had already diverged (extraction handled timeouts, evaluation
did not, and neither handled 502/504).
"""

from __future__ import annotations

# Substrings that indicate a transient, worth-retrying failure.
#
# Included:
#   429 / rate limit / resource exhausted -- throttled, backoff helps.
#   502 / 503 / 504 and gateway wording   -- the router could not reach (or
#       got no timely answer from) the upstream model provider. These are the
#       single most common HF-router blip and are almost always transient.
#   timeouts / connection errors          -- network-level, retry is cheap.
#
# Deliberately NOT included:
#   500 Internal Server Error -- ambiguous. It can be transient, but it also
#       covers genuine malformed-request bugs, where retrying just adds delay
#       before the local-model fallback runs anyway. Left out so a real bug
#       fails fast rather than sleeping through the backoff schedule first.
#   401 / 403 / 404 / 400    -- auth, quota, and bad-request errors. Retrying
#       cannot fix these.
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
    """True when `exc` looks like a transient remote-API failure worth retrying.

    Matching is done on the lowercased string form of the exception, because
    the HF client surfaces status codes inside message text rather than as a
    consistently-typed attribute.
    """
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_ERROR_MARKERS)
