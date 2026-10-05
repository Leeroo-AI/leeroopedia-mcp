"""
HTTP client for Leeroopedia API gateway.

Handles authentication, error mapping, and response parsing.
Uses async task-based API: POST /search creates a task,
then poll GET /search/task/{task_id} for results.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from . import __version__
from .config import Config

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API errors."""
    def __init__(self, message: str, code: str, status_code: int = 500):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class AuthenticationError(APIError):
    """Raised when API key is invalid or revoked."""
    def __init__(self, message: str = "Invalid or revoked API key"):
        super().__init__(message, "invalid_api_key", 401)


class InsufficientCreditsError(APIError):
    """Raised when user has no credits remaining."""
    def __init__(self, message: str = "No credits remaining. Purchase more at https://app.leeroopedia.com"):
        super().__init__(message, "insufficient_credits", 402)


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    def __init__(self, retry_after: int = 60):
        super().__init__(
            f"Rate limit exceeded. Retry after {retry_after} seconds.",
            "rate_limited",
            429
        )
        self.retry_after = retry_after


class TaskTimeoutError(APIError):
    """Raised when a search task does not complete in time."""
    def __init__(self, task_id: str, max_wait: int, last_error: Optional[str] = None):
        message = f"Search task {task_id} did not complete within {max_wait}s"
        if last_error:
            # Polling was still failing at the deadline - say why
            message += f" (last poll error: {last_error})"
        super().__init__(message, "task_timeout", 504)
        self.task_id = task_id


class _TransientPollError(Exception):
    """A single poll attempt failed in a way that is worth retrying."""
    def __init__(self, reason: str, retry_after: Optional[float] = None):
        super().__init__(reason)
        self.retry_after = retry_after


@dataclass
class SearchResponse:
    """Response from search API."""
    success: bool
    results: str
    latency_ms: int
    # None when the gateway did not report a balance
    credits_remaining: Optional[int]
    error: Optional[str] = None


def _parse_error_body(response: httpx.Response) -> Dict[str, Any]:
    """
    Extract error details from a gateway error response.

    The gateway wraps errors as {"error": ..., "message": <detail>}, where
    <detail> is a string or a dict such as
    {"error": "invalid_api_key", "message": "..."}. FastAPI's default
    {"detail": <detail>} envelope is accepted too. A proxy in front of the
    gateway can answer with a non-JSON body (e.g. an HTML 502 page), so the
    body is never assumed to parse.

    Returns:
        Dict with optional "message", "error" and "retry_after" keys
    """
    try:
        body = response.json()
    except ValueError:
        return {}
    if not isinstance(body, dict):
        return {}

    detail = body.get("detail", body.get("message"))
    if isinstance(detail, dict):
        return detail
    if isinstance(detail, str) and detail:
        return {"message": detail, "error": body.get("error")}
    return {"error": body.get("error")}


def _retry_after_seconds(
    response: httpx.Response,
    error_data: Dict[str, Any],
    default: int = 60,
) -> int:
    """Read the retry delay from the error body or the Retry-After header."""
    for raw in (error_data.get("retry_after"), response.headers.get("Retry-After")):
        try:
            seconds = int(float(raw))
        except (TypeError, ValueError, OverflowError):
            continue
        if seconds >= 0:
            return seconds
    return default


class LeeroopediaClient:
    """
    HTTP client for Leeroopedia API gateway.

    Uses the async task-based search API:
      1. POST /search -> returns task_id (queued in Celery)
      2. GET /search/task/{task_id} -> poll until success/failure
    """

    # Terminal task statuses that stop polling
    TERMINAL_STATUSES = {"success", "failure"}

    # HTTP statuses on a poll request that are worth retrying. The task keeps
    # running on the backend, so a failed status check must not discard it.
    RETRYABLE_POLL_STATUSES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        config: Config,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.config = config
        # Use a longer timeout for individual requests (polling is fast)
        self.client = httpx.AsyncClient(
            base_url=config.api_url,
            timeout=300.0,
            headers={
                "X-API-Key": config.api_key,
                "Content-Type": "application/json",
                "User-Agent": f"leeroopedia-mcp/{__version__}",
            },
            # Only set by tests, to mock the gateway without real HTTP
            transport=transport,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    def _handle_error_response(self, response: httpx.Response) -> None:
        """
        Check HTTP response for error status codes and raise typed exceptions.

        Args:
            response: The HTTP response to check

        Raises:
            AuthenticationError: 401 - invalid API key
            InsufficientCreditsError: 402 - no credits
            RateLimitError: 429 - rate limited
            APIError: Other 4xx/5xx errors
        """
        status = response.status_code
        if status < 400:
            return

        error_data = _parse_error_body(response)
        message = error_data.get("message")
        if not isinstance(message, str) or not message:
            message = None

        if status == 401:
            raise AuthenticationError(message or "Invalid API key")

        if status == 402:
            # Fall back to the exception's default text rather than passing None
            raise InsufficientCreditsError(message) if message else InsufficientCreditsError()

        if status == 429:
            raise RateLimitError(_retry_after_seconds(response, error_data))

        code = error_data.get("error")
        raise APIError(
            message or f"API error (HTTP {status})",
            code if isinstance(code, str) and code else "api_error",
            status,
        )

    async def _create_search_task(
        self,
        tool: str,
        arguments: Dict[str, Any],
    ) -> str:
        """
        Create a search task via POST /search.

        Returns immediately with a task_id. The actual search
        runs asynchronously in the Celery worker queue.

        Args:
            tool: Agentic tool name
            arguments: Tool-specific arguments dict

        Returns:
            task_id string for polling

        Raises:
            AuthenticationError, InsufficientCreditsError,
            RateLimitError, APIError on HTTP errors
        """
        payload = {
            "tool": tool,
            "arguments": arguments,
        }

        response = await self.client.post("/v1/search", json=payload)

        # Check for error responses (auth, credits, rate limit, etc.)
        self._handle_error_response(response)

        data = response.json()
        task_id = data.get("task_id")

        if not task_id:
            raise APIError("No task_id in response", "missing_task_id", 500)

        logger.info(f"Search task created: {task_id}")
        return task_id

    async def _poll_once(self, task_id: str) -> Dict[str, Any]:
        """
        Make a single status request for a task.

        Args:
            task_id: The task ID returned from _create_search_task

        Returns:
            The parsed task status payload

        Raises:
            _TransientPollError: Network error, retryable HTTP status, or an
                unreadable body. The task is still valid - poll again.
            AuthenticationError, InsufficientCreditsError, APIError:
                HTTP errors that retrying will not fix
        """
        try:
            response = await self.client.get(f"/v1/search/task/{task_id}")
        except httpx.TransportError as e:
            # Timeouts, connection resets, DNS failures, etc.
            reason = type(e).__name__ + (f": {e}" if str(e) else "")
            raise _TransientPollError(reason) from e

        if response.status_code in self.RETRYABLE_POLL_STATUSES:
            retry_after = None
            if response.status_code == 429:
                # Respect the gateway's requested delay when it gives one
                error_data = _parse_error_body(response)
                retry_after = _retry_after_seconds(response, error_data, default=0) or None
            raise _TransientPollError(f"HTTP {response.status_code}", retry_after)

        # Any other 4xx (401, 402, 404, ...) will not fix itself
        self._handle_error_response(response)

        try:
            data = response.json()
        except ValueError as e:
            raise _TransientPollError("poll response was not valid JSON") from e
        if not isinstance(data, dict):
            raise _TransientPollError("poll response was not a JSON object")
        return data

    async def _poll_search_task(self, task_id: str) -> SearchResponse:
        """
        Poll GET /search/task/{task_id} until the task completes.

        Uses exponential backoff: starts at config.poll_initial_interval,
        grows by 1.5x each iteration, capped at 5 seconds.

        A poll request that fails transiently (network error, 429, 5xx) is
        retried with the same task_id. The task is already running and paid
        for, so one failed status check must not throw its result away.

        Args:
            task_id: The task ID returned from _create_search_task

        Returns:
            SearchResponse with results on success

        Raises:
            TaskTimeoutError: If task doesn't complete within max_wait
            APIError: If the task fails or polling hits a non-retryable error
        """
        max_wait = self.config.poll_max_wait
        delay = self.config.poll_initial_interval
        max_delay = 5.0  # Cap backoff at 5 seconds
        start_time = time.monotonic()
        last_error: Optional[str] = None

        while time.monotonic() - start_time < max_wait:
            wait = delay

            try:
                data = await self._poll_once(task_id)
            except _TransientPollError as e:
                last_error = str(e)
                if e.retry_after:
                    wait = max(wait, e.retry_after)
                logger.warning(f"Poll for task {task_id} failed ({e}), retrying in {wait:.1f}s")
            else:
                last_error = None
                status = data.get("status") or ""

                if status == "success":
                    if data.get("success") is False:
                        # Task finished, but the backend reports the search itself failed
                        error_msg = data.get("error") or "Search task failed"
                        raise APIError(error_msg, "task_failure", 500)

                    # The API sends explicit nulls for unset fields, so use "or":
                    # data.get(key, default) would still return None for them.
                    return SearchResponse(
                        success=True,
                        results=data.get("results") or "",
                        latency_ms=data.get("latency_ms") or 0,
                        credits_remaining=data.get("credits_remaining"),
                    )

                if status == "failure":
                    # Task failed - credits are auto-refunded by the gateway
                    error_msg = data.get("error") or "Search task failed"
                    raise APIError(error_msg, "task_failure", 500)

                # Task still in progress (queued/pending/started) - wait and retry
                logger.debug(f"Task {task_id} status: {status}, polling again in {wait:.1f}s")

            # Never sleep past the overall deadline
            remaining = max_wait - (time.monotonic() - start_time)
            await asyncio.sleep(max(0.0, min(wait, remaining)))

            # Exponential backoff capped at max_delay
            delay = min(delay * 1.5, max_delay)

        # Timed out waiting for task completion
        raise TaskTimeoutError(task_id, max_wait, last_error)

    async def search(
        self,
        tool: str,
        arguments: Dict[str, Any],
    ) -> SearchResponse:
        """
        Execute an agentic search using the async task API.

        Creates a search task, then polls for the result with
        exponential backoff. The caller sees a simple request/response
        interface - the async polling is handled internally.

        Args:
            tool: Agentic tool name (one of 8 tools)
            arguments: Tool-specific arguments dict

        Returns:
            SearchResponse with results

        Raises:
            AuthenticationError: If API key is invalid
            InsufficientCreditsError: If no credits remaining
            RateLimitError: If rate limit exceeded
            TaskTimeoutError: If search doesn't complete in time
            APIError: For other API errors
        """
        try:
            # Step 1: Create the search task (returns immediately).
            # Not retried: a repeated POST could start and bill a second task.
            task_id = await self._create_search_task(
                tool=tool,
                arguments=arguments,
            )

            # Step 2: Poll for the result with exponential backoff.
            # Transient poll failures are retried inside _poll_search_task.
            return await self._poll_search_task(task_id)

        except (AuthenticationError, InsufficientCreditsError, RateLimitError, TaskTimeoutError):
            # Re-raise typed exceptions directly
            raise
        except httpx.TimeoutException:
            raise APIError("Request timed out", "timeout", 504)
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise APIError(f"Connection error: {e}", "connection_error", 503)

    async def __aenter__(self) -> "LeeroopediaClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
