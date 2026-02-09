"""
Test 2: Test the LeeroopediaClient directly.

Tests the full MCP client flow: client.search() which internally
creates a task and polls for results.
"""

import asyncio
import os

# Set env vars before importing config
os.environ["LEEROOPEDIA_API_KEY"] = "kpsk_021a25ec_021a25ecdbc544405e284e7419239b05"
os.environ["LEEROOPEDIA_API_URL"] = "https://api.leeroopedia.com"

from leeroopedia_mcp.config import Config
from leeroopedia_mcp.client import LeeroopediaClient


async def test_idea_search():
    """Test wiki_idea_search via the client."""
    print("=" * 50)
    print("TEST 2a: wiki_idea_search via LeeroopediaClient")
    print("=" * 50)

    config = Config()
    async with LeeroopediaClient(config) as client:
        result = await client.search(
            query="machine learning",
            tool="wiki_idea_search",
            top_k=3,
        )

        print(f"  success: {result.success}")
        print(f"  results_count: {result.results_count}")
        print(f"  latency_ms: {result.latency_ms}")
        print(f"  credits_remaining: {result.credits_remaining}")
        preview = (result.results or "")[:300]
        print(f"  results preview: {preview}...")

        # Validate client flow worked (got a real response, not an error)
        assert result.credits_remaining is not None, "Expected credits_remaining in response"
        print("\n  PASS")
        return True


async def test_code_search():
    """Test wiki_code_search via the client."""
    print("\n" + "=" * 50)
    print("TEST 2b: wiki_code_search via LeeroopediaClient")
    print("=" * 50)

    config = Config()
    async with LeeroopediaClient(config) as client:
        result = await client.search(
            query="PyTorch training loop",
            tool="wiki_code_search",
            top_k=2,
        )

        print(f"  success: {result.success}")
        print(f"  results_count: {result.results_count}")
        print(f"  latency_ms: {result.latency_ms}")
        print(f"  credits_remaining: {result.credits_remaining}")
        preview = (result.results or "")[:300]
        print(f"  results preview: {preview}...")

        # Validate client flow worked (got a real response, not an error)
        assert result.credits_remaining is not None, "Expected credits_remaining in response"
        print("\n  PASS")
        return True


async def main():
    results = []

    try:
        results.append(await test_idea_search())
    except Exception as e:
        print(f"\n  FAIL: {e}")
        results.append(False)

    try:
        results.append(await test_code_search())
    except Exception as e:
        print(f"\n  FAIL: {e}")
        results.append(False)

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Test 2 result: {passed}/{total} passed")
    if all(results):
        print("ALL PASS")
    else:
        print("SOME FAILED")


if __name__ == "__main__":
    asyncio.run(main())
