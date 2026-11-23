"""Detect AI/Ollama connectivity failures that must surface as service errors."""

from __future__ import annotations

import httpx

# ConnectionError: raised by the ollama package after httpx.ConnectError.
# httpx.RequestError: covers ConnectError, TimeoutException, and other transport failures.
AI_SERVICE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    httpx.RequestError,
)


def is_ai_service_error(exc: BaseException) -> bool:
    """True if this exception (or its cause chain) is an AI connectivity failure."""
    seen: set[int] = set()
    current: BaseException | None = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, AI_SERVICE_EXCEPTIONS):
            return True
        current = current.__cause__ or current.__context__

    return False
