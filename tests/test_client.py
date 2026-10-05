"""HTTP client: task creation, polling, and error mapping (client.py)."""

import json

import httpx
import pytest

from fakes import (
    API_KEY,
    PENDING,
    STARTED,
    SUCCESS,
    TASK_ID,
    gateway_error,
    html_response,
    json_response,
    network_error,
    task_payload,
)
from leeroopedia_mcp import __version__
from leeroopedia_mcp.client import (
    APIError,
    AuthenticationError,
    InsufficientCreditsError,
    RateLimitError,
    TaskTimeoutError,
)

# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_search_creates_a_task_then_polls_until_success(search, gateway):
    gateway.polls = [PENDING, STARTED, SUCCESS]

    response = search("build_plan", {"goal": "fine-tune", "constraints": "1 GPU"})

    assert response.success is True
    assert response.results == "# Answer\n\nUse rank 16 [Heuristic/LoRA_Rank]"
    assert response.latency_ms == 42
    assert response.credits_remaining == 99

    # One task is created, with the tool name and arguments passed through untouched
    assert len(gateway.create_requests) == 1
    create = gateway.create_requests[0]
    assert create.url.path == "/v1/search"
    assert json.loads(create.content) == {
        "tool": "build_plan",
        "arguments": {"goal": "fine-tune", "constraints": "1 GPU"},
    }

    # Every poll targets the task returned by the create call
    assert len(gateway.poll_requests) == 3
    assert {r.url.path for r in gateway.poll_requests} == {f"/v1/search/task/{TASK_ID}"}


def test_requests_are_authenticated_and_identify_the_client(search, gateway):
    search()

    for request in gateway.requests:
        assert request.headers["X-API-Key"] == API_KEY
        assert request.headers["User-Agent"] == f"leeroopedia-mcp/{__version__}"


def test_poll_interval_backs_off_and_is_capped_at_five_seconds(search, gateway, clock):
    gateway.polls = [PENDING] * 8 + [SUCCESS]

    search()

    assert clock.sleeps == pytest.approx(
        [0.5, 0.75, 1.125, 1.6875, 2.53125, 3.796875, 5.0, 5.0]
    )


def test_search_times_out_when_the_task_never_finishes(search, gateway, config, clock):
    config.poll_max_wait = 10
    gateway.polls = [PENDING]

    with pytest.raises(TaskTimeoutError) as exc:
        search()

    assert exc.value.task_id == TASK_ID
    assert exc.value.status_code == 504
    assert "did not complete within 10s" in str(exc.value)
    assert "last poll error" not in str(exc.value)
    # The final sleep is trimmed so the client never waits past the deadline
    assert clock.now == pytest.approx(10)


# --------------------------------------------------------------------------
# Errors when creating the task
# --------------------------------------------------------------------------


def test_invalid_api_key_raises_authentication_error(search, gateway):
    gateway.create = json_response(
        401, gateway_error("invalid_api_key", "Invalid or revoked API key")
    )

    with pytest.raises(AuthenticationError) as exc:
        search()

    assert str(exc.value) == "Invalid or revoked API key"
    assert gateway.poll_requests == []


def test_no_credits_raises_insufficient_credits_error(search, gateway):
    gateway.create = json_response(
        402, gateway_error("insufficient_credits", "You are out of credits")
    )

    with pytest.raises(InsufficientCreditsError) as exc:
        search()

    assert str(exc.value) == "You are out of credits"
    assert exc.value.status_code == 402


def test_no_credits_without_a_message_uses_the_default_text(search, gateway):
    gateway.create = json_response(402, {"error": "HTTPException"})

    with pytest.raises(InsufficientCreditsError) as exc:
        search()

    # Regression: a missing message used to surface as the literal "None"
    assert "None" not in str(exc.value)
    assert "No credits remaining" in str(exc.value)


@pytest.mark.parametrize(
    "create, expected",
    [
        (json_response(429, gateway_error("rate_limited", "slow down", retry_after=17)), 17),
        (json_response(429, {"error": "HTTPException"}, headers={"Retry-After": "30"}), 30),
        (json_response(429, {"error": "HTTPException"}), 60),
        (html_response(429), 60),
    ],
    ids=["body", "header", "default", "non-json"],
)
def test_rate_limit_reports_when_to_retry(search, gateway, create, expected):
    gateway.create = create

    with pytest.raises(RateLimitError) as exc:
        search()

    assert exc.value.retry_after == expected
    assert f"{expected} seconds" in str(exc.value)


def test_validation_error_envelope_is_reported(search, gateway):
    gateway.create = json_response(
        422, {"error": "validation_error", "details": [{"loc": ["body", "tool"]}]}
    )

    with pytest.raises(APIError) as exc:
        search()

    assert exc.value.status_code == 422
    assert exc.value.code == "validation_error"
    assert "422" in str(exc.value)


def test_fastapi_detail_envelope_is_understood(search, gateway):
    gateway.create = json_response(404, {"detail": "Not Found"})

    with pytest.raises(APIError) as exc:
        search()

    assert str(exc.value) == "Not Found"
    assert exc.value.status_code == 404


def test_non_json_error_page_becomes_an_api_error(search, gateway):
    gateway.create = html_response(502)

    # Regression: this used to escape as a raw JSONDecodeError
    with pytest.raises(APIError) as exc:
        search()

    assert exc.value.status_code == 502
    assert "502" in str(exc.value)


def test_create_response_without_task_id_is_an_error(search, gateway):
    gateway.create = json_response(200, {"status": "queued"})

    with pytest.raises(APIError) as exc:
        search()

    assert exc.value.code == "missing_task_id"
    assert gateway.poll_requests == []


@pytest.mark.parametrize(
    "error, code",
    [
        (httpx.ConnectError("connection refused"), "connection_error"),
        (httpx.ReadTimeout("timed out"), "timeout"),
    ],
    ids=["connect-error", "timeout"],
)
def test_network_failure_while_creating_is_not_retried(search, gateway, error, code):
    gateway.create = network_error(error)

    with pytest.raises(APIError) as exc:
        search()

    assert exc.value.code == code
    # Retrying the POST could start, and bill for, a second task
    assert len(gateway.create_requests) == 1


# --------------------------------------------------------------------------
# Polling survives transient failures (the task is already paid for)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        json_response(500, {"error": "InternalServerError"}),
        html_response(502),
        json_response(503, gateway_error("unavailable", "try later")),
        html_response(504),
        network_error(httpx.ConnectError("connection reset")),
        network_error(httpx.ReadTimeout("timed out")),
        network_error(httpx.RemoteProtocolError("server disconnected")),
        html_response(200),
        json_response(200, ["not", "an", "object"]),
    ],
    ids=[
        "500", "502-html", "503", "504-html",
        "connect-error", "read-timeout", "protocol-error",
        "200-non-json", "200-wrong-shape",
    ],
)
def test_transient_poll_failure_is_retried_with_the_same_task(search, gateway, failure):
    gateway.polls = [PENDING, failure, SUCCESS]

    response = search()

    assert response.credits_remaining == 99
    # The task is not recreated, so the user is not charged twice
    assert len(gateway.create_requests) == 1
    assert len(gateway.poll_requests) == 3
    assert {r.url.path for r in gateway.poll_requests} == {f"/v1/search/task/{TASK_ID}"}


def test_several_consecutive_poll_failures_still_recover(search, gateway):
    gateway.polls = [
        html_response(503),
        network_error(httpx.ConnectError("connection reset")),
        html_response(502),
        network_error(httpx.ReadTimeout("timed out")),
        SUCCESS,
    ]

    assert search().results.startswith("# Answer")
    assert len(gateway.create_requests) == 1


def test_failed_polls_keep_backing_off(search, gateway, clock):
    gateway.polls = [html_response(503)] * 3 + [SUCCESS]

    search()

    assert clock.sleeps == pytest.approx([0.5, 0.75, 1.125])


def test_rate_limited_poll_waits_for_the_requested_delay(search, gateway, clock):
    gateway.polls = [
        json_response(429, gateway_error("rate_limited", "slow down", retry_after=7)),
        SUCCESS,
    ]

    search()

    assert clock.sleeps == pytest.approx([7])


def test_rate_limited_poll_never_waits_past_the_deadline(search, gateway, config, clock):
    config.poll_max_wait = 10
    gateway.polls = [json_response(429, {"error": "HTTPException"}, headers={"Retry-After": "600"})]

    with pytest.raises(TaskTimeoutError):
        search()

    assert clock.now == pytest.approx(10)


def test_timeout_after_persistent_poll_failures_says_why(search, gateway, config):
    config.poll_max_wait = 10
    gateway.polls = [html_response(503)]

    with pytest.raises(TaskTimeoutError) as exc:
        search()

    assert exc.value.task_id == TASK_ID
    assert "last poll error: HTTP 503" in str(exc.value)


def test_timeout_does_not_blame_a_poll_failure_that_recovered(search, gateway, config):
    config.poll_max_wait = 10
    gateway.polls = [html_response(503), PENDING]

    with pytest.raises(TaskTimeoutError) as exc:
        search()

    assert "last poll error" not in str(exc.value)


@pytest.mark.parametrize(
    "failure, error_type",
    [
        (json_response(401, gateway_error("invalid_api_key", "Key revoked")), AuthenticationError),
        (json_response(402, gateway_error("insufficient_credits", "No credits")), InsufficientCreditsError),
        (json_response(404, {"error": "HTTPException", "message": "Not Found"}), APIError),
    ],
    ids=["401", "402", "404"],
)
def test_non_retryable_poll_error_fails_immediately(search, gateway, failure, error_type):
    gateway.polls = [failure, SUCCESS]

    with pytest.raises(error_type):
        search()

    assert len(gateway.poll_requests) == 1


# --------------------------------------------------------------------------
# Task outcomes
# --------------------------------------------------------------------------


def test_failed_task_raises_with_the_backend_error(search, gateway):
    gateway.polls = [json_response(200, task_payload("failure", success=False, error="agent crashed"))]

    with pytest.raises(APIError) as exc:
        search()

    assert str(exc.value) == "agent crashed"
    assert exc.value.code == "task_failure"


def test_failed_task_with_null_error_uses_a_default_message(search, gateway):
    gateway.polls = [json_response(200, task_payload("failure"))]

    with pytest.raises(APIError) as exc:
        search()

    # Regression: the explicit null used to surface as the literal "None"
    assert str(exc.value) == "Search task failed"


def test_finished_task_reporting_success_false_is_an_error(search, gateway):
    # Real gateway reply for a tool it does not know: the task "succeeds"
    # at the queue level but carries success=false and an error message.
    gateway.polls = [json_response(200, task_payload(
        "success",
        success=False,
        results="",
        results_count=0,
        latency_ms=0,
        credits_remaining=752,
        error="Unknown tool 'nope'.",
    ))]

    with pytest.raises(APIError) as exc:
        search()

    assert str(exc.value) == "Unknown tool 'nope'."
    assert exc.value.code == "task_failure"


def test_successful_task_with_null_fields_is_normalised(search, gateway):
    gateway.polls = [json_response(200, task_payload("success", success=True))]

    response = search()

    # results stays a string so callers can safely build text from it
    assert response.results == ""
    assert response.latency_ms == 0
    assert response.credits_remaining is None
