"""
HTTP client for Leeroopedia API gateway.

Handles authentication, error mapping, and response parsing.
"""

import logging
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
    def __init__(self, message: str = "No credits remaining. Purchase more at https://leeroopedia.com/dashboard"):
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
    """HTTP client for Leeroopedia API gateway."""

    def __init__(self, config: Config):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.api_url,
            timeout=30.0,
            headers={
                "X-API-Key": config.api_key,
                "Content-Type": "application/json",
                "User-Agent": "leeroopedia-mcp/0.1.0",
            },
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def search(
        self,
        query: str,
        tool: str = "wiki_idea_search",
        top_k: int = 5,
        domains: Optional[List[str]] = None,
    ) -> SearchResponse:
        """
        Execute a search query.

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
            APIError: For other API errors
        """
        payload: Dict[str, Any] = {
            "query": query,
            "tool": tool,
        }

        if top_k != 5:
            payload["top_k"] = min(top_k, 20)

        if domains:
            payload["domains"] = domains

        try:
            response = await self.client.post("/search", json=payload)

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

            data = response.json()
            return SearchResponse(
                success=data.get("success", True),
                results=data.get("results", ""),
                results_count=data.get("results_count", 0),
                latency_ms=data.get("latency_ms", 0),
                credits_remaining=data.get("credits_remaining", 0),
            )

        except httpx.TimeoutException:
            raise APIError("Request timed out", "timeout", 504)
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise APIError(f"Connection error: {e}", "connection_error", 503)

    async def __aenter__(self) -> "LeeroopediaClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
