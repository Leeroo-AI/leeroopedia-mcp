"""
Test doubles for the Leeroopedia API gateway.

Nothing here touches the network, so the suite spends no API credits.
Response shapes mirror what the real gateway returns.
"""

from typing import Any, Callable, Dict, List, Optional

import httpx

API_KEY = "kpsk_test_key"
TASK_ID = "task-123"

# A scripted gateway reply: called once per request, returns a response or raises
Step = Callable[[], httpx.Response]


def json_response(status: int, body: Any, headers: Optional[Dict[str, str]] = None) -> Step:
    """A reply with a JSON body."""
    return lambda: httpx.Response(status, json=body, headers=headers)


def html_response(status: int) -> Step:
    """A non-JSON reply, like a proxy error page in front of the gateway."""
    return lambda: httpx.Response(
        status,
        text="<html><body>Bad Gateway</body></html>",
        headers={"content-type": "text/html"},
    )


def network_error(exc: Exception) -> Step:
    """A request that never gets a reply (timeout, connection reset, ...)."""
    def _raise() -> httpx.Response:
        raise exc
    return _raise


def gateway_error(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    """Error body in the envelope the real gateway uses."""
    return {"error": "HTTPException", "message": {"error": code, "message": message, **extra}}


def task_payload(status: str, **fields: Any) -> Dict[str, Any]:
    """
    Poll payload for a task.

    The real API sends every field on every poll and uses explicit nulls
    until the task finishes, so unset fields default to None here too.
    """
    payload = {
        "task_id": TASK_ID,
        "status": status,
        "success": None,
        "results": None,
        "results_count": None,
        "latency_ms": None,
        "credits_remaining": None,
        "error": None,
    }
    payload.update(fields)
    return payload


PENDING = json_response(200, task_payload("pending"))
STARTED = json_response(200, task_payload("started"))
SUCCESS = json_response(200, task_payload(
    "success",
    success=True,
    results="# Answer\n\nUse rank 16 [Heuristic/LoRA_Rank]",
    results_count=1,
    latency_ms=42,
    credits_remaining=99,
))


class FakeGateway:
    """
    Scriptable stand-in for the gateway, used as an httpx mock transport.

    POST /v1/search answers with `create`. Polls consume `polls` in order,
    and the last entry repeats forever (handy for "never finishes" cases).
    """

    def __init__(self) -> None:
        self.create: Step = json_response(200, {"task_id": TASK_ID, "status": "queued"})
        self.polls: List[Step] = [SUCCESS]
        self.requests: List[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "POST":
            return self.create()
        step = self.polls[0] if len(self.polls) == 1 else self.polls.pop(0)
        return step()

    @property
    def create_requests(self) -> List[httpx.Request]:
        return [r for r in self.requests if r.method == "POST"]

    @property
    def poll_requests(self) -> List[httpx.Request]:
        return [r for r in self.requests if r.method == "GET"]


class FakeClock:
    """Virtual time: sleeping advances the clock instantly and is recorded."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: List[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
