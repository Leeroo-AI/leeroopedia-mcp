"""
Test 1: Standalone smoke test for async task-based search API.

Tests the raw HTTP flow: POST /v1/search -> poll GET /v1/search/task/{task_id}
without involving the MCP client layer.
"""

import asyncio
import time
import httpx

API_URL = "https://api.leeroopedia.com"
API_KEY = "kpsk_021a25ec_021a25ecdbc544405e284e7419239b05"


async def test_async_search():
    async with httpx.AsyncClient(
        base_url=API_URL,
        headers={
            "X-API-Key": API_KEY,
            "Content-Type": "application/json",
        },
        timeout=60.0,
    ) as client:

        # --- Step 1: Create task ---
        print("=" * 50)
        print("STEP 1: Creating search task...")
        resp = await client.post(
            "/v1/search",
            json={"query": "machine learning", "tool": "wiki_idea_search"},
        )
        print(f"  HTTP status: {resp.status_code}")
        data = resp.json()
        print(f"  Response: {data}")

        if resp.status_code != 200:
            print(f"\nFAILED: Task creation returned {resp.status_code}")
            return False

        task_id = data.get("task_id")
        if not task_id:
            print("\nFAILED: No task_id in response!")
            return False

        print(f"  Task ID: {task_id}")
        print(f"  Status: {data.get('status')}")

        # --- Step 2: Poll for result ---
        print("\n" + "=" * 50)
        print("STEP 2: Polling for result...")
        delay = 0.5
        max_delay = 5.0
        start = time.monotonic()
        max_wait = 60

        while time.monotonic() - start < max_wait:
            resp = await client.get(f"/v1/search/task/{task_id}")
            result = resp.json()
            status = result.get("status", "unknown")
            elapsed = time.monotonic() - start
            print(f"  [{elapsed:.1f}s] status={status}")

            if status == "success":
                print("\n" + "=" * 50)
                print("SUCCESS!")
                print(f"  results_count: {result.get('results_count')}")
                print(f"  latency_ms: {result.get('latency_ms')}")
                print(f"  credits_remaining: {result.get('credits_remaining')}")
                preview = (result.get("results") or "")[:300]
                print(f"  results preview: {preview}...")
                return True

            if status == "failure":
                print(f"\nFAILED: {result.get('error')}")
                return False

            await asyncio.sleep(delay)
            delay = min(delay * 1.5, max_delay)

        print(f"\nTIMEOUT after {max_wait}s!")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_async_search())
    print("\n" + "=" * 50)
    print(f"Test 1 result: {'PASS' if success else 'FAIL'}")
