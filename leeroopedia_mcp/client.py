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
from typing import Any, Dict, List, Optional

import httpx

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
    def __init__(self, task_id: str, max_wait: int):
        super().__init__(
            f"Search task {task_id} did not complete within {max_wait}s",
            "task_timeout",
            504
        )
        self.task_id = task_id


@dataclass
class SearchResponse:
    """Response from search API."""
    success: bool
    results: str
    results_count: int
    latency_ms: int
    credits_remaining: int
    error: Optional[str] = None


class LeeroopediaClient:
    """
    HTTP client for Leeroopedia API gateway.

    Uses the async task-based search API:
      1. POST /search -> returns task_id (queued in Celery)
      2. GET /search/task/{task_id} -> poll until success/failure
    """

    # Terminal task statuses that stop polling
    TERMINAL_STATUSES = {"success", "failure"}

    def __init__(self, config: Config):
        self.config = config
        # Use a longer timeout for individual requests (polling is fast)
        self.client = httpx.AsyncClient(
            base_url=config.api_url,
            timeout=60.0,
            headers={
                "X-API-Key": config.api_key,
                "Content-Type": "application/json",
                "User-Agent": "leeroopedia-mcp/0.1.0",
            },
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
        if response.status_code == 401:
            error_data = response.json().get("detail", {})
            raise AuthenticationError(error_data.get("message", "Invalid API key"))

        if response.status_code == 402:
            error_data = response.json().get("detail", {})
            raise InsufficientCreditsError(error_data.get("message"))

        if response.status_code == 429:
            error_data = response.json().get("detail", {})
            retry_after = error_data.get("retry_after", 60)
            raise RateLimitError(retry_after)

        if response.status_code >= 400:
            error_data = response.json().get("detail", {})
            raise APIError(
                error_data.get("message", "API error"),
                error_data.get("error", "api_error"),
                response.status_code,
            )

    async def _create_search_task(
        self,
        query: str,
        tool: str,
        top_k: int = 5,
        domains: Optional[List[str]] = None,
    ) -> str:
        """
        Create a search task via POST /search.

        Returns immediately with a task_id. The actual search
        runs asynchronously in the Celery worker queue.

        Args:
            query: Search query string
            tool: Tool name (wiki_idea_search or wiki_code_search)
            top_k: Number of results to return (max 20)
            domains: Optional domain filters

        Returns:
            task_id string for polling

        Raises:
            AuthenticationError, InsufficientCreditsError,
            RateLimitError, APIError on HTTP errors
        """
        payload: Dict[str, Any] = {
            "query": query,
            "tool": tool,
        }

        if top_k != 5:
            payload["top_k"] = min(top_k, 20)

        if domains:
            payload["domains"] = domains

        response = await self.client.post("/v1/search", json=payload)

        # Check for error responses (auth, credits, rate limit, etc.)
        self._handle_error_response(response)

        data = response.json()
        task_id = data.get("task_id")

        if not task_id:
            raise APIError("No task_id in response", "missing_task_id", 500)

        logger.info(f"Search task created: {task_id}")
        return task_id

    async def _poll_search_task(self, task_id: str) -> SearchResponse:
        """
        Poll GET /search/task/{task_id} until the task completes.

        Uses exponential backoff: starts at config.poll_initial_interval,
        grows by 1.5x each iteration, capped at 5 seconds.

        Args:
            task_id: The task ID returned from _create_search_task

        Returns:
            SearchResponse with results on success

        Raises:
            TaskTimeoutError: If task doesn't complete within max_wait
            APIError: If the task fails or polling encounters an error
        """
        max_wait = self.config.poll_max_wait
        delay = self.config.poll_initial_interval
        max_delay = 5.0  # Cap backoff at 5 seconds
        start_time = time.monotonic()

        while time.monotonic() - start_time < max_wait:
            response = await self.client.get(f"/v1/search/task/{task_id}")

            # Handle HTTP-level errors on the poll endpoint
            if response.status_code >= 400:
                self._handle_error_response(response)

            data = response.json()
            status = data.get("status", "")

            if status == "success":
                # Task completed - return results
                return SearchResponse(
                    success=data.get("success", True),
                    results=data.get("results", ""),
                    results_count=data.get("results_count", 0),
                    latency_ms=data.get("latency_ms", 0),
                    credits_remaining=data.get("credits_remaining", 0),
                )

            if status == "failure":
                # Task failed - credits are auto-refunded by the gateway
                error_msg = data.get("error", "Search task failed")
                raise APIError(error_msg, "task_failure", 500)

            # Task still in progress (queued/pending/started) - wait and retry
            logger.debug(f"Task {task_id} status: {status}, polling again in {delay:.1f}s")
            await asyncio.sleep(delay)

            # Exponential backoff capped at max_delay
            delay = min(delay * 1.5, max_delay)

        # Timed out waiting for task completion
        raise TaskTimeoutError(task_id, max_wait)

    async def search(
        self,
        query: str,
        tool: str = "wiki_idea_search",
        top_k: int = 5,
        domains: Optional[List[str]] = None,
    ) -> SearchResponse:
        """
        Execute a search query using the async task API.

        Creates a search task, then polls for the result with
        exponential backoff. The caller sees a simple request/response
        interface - the async polling is handled internally.

        Args:
            query: Search query string
            tool: Tool name (wiki_idea_search or wiki_code_search)
            top_k: Number of results to return (max 20)
            domains: Optional domain filters

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
            # Step 1: Create the search task (returns immediately)
            task_id = await self._create_search_task(
                query=query,
                tool=tool,
                top_k=top_k,
                domains=domains,
            )

            # Step 2: Poll for the result with exponential backoff
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
