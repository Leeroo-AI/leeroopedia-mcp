"""Shared fixtures for the Leeroopedia MCP test suite."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

import leeroopedia_mcp.client as client_module
from fakes import API_KEY, FakeClock, FakeGateway
from leeroopedia_mcp.client import LeeroopediaClient
from leeroopedia_mcp.config import Config

ENV_VARS = (
    "LEEROOPEDIA_API_KEY",
    "LEEROOPEDIA_API_URL",
    "LEEROOPEDIA_POLL_MAX_WAIT",
    "LEEROOPEDIA_POLL_INTERVAL",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Start from an environment with no Leeroopedia settings."""
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def config(clean_env):
    """A valid config using defaults plus a test API key."""
    clean_env.setenv("LEEROOPEDIA_API_KEY", API_KEY)
    return Config()


@pytest.fixture
def clock(monkeypatch):
    """
    Replace the client's clock and sleep with virtual time.

    Only the names inside the client module are swapped, so the real
    asyncio event loop running the test is unaffected.
    """
    fake = FakeClock()
    monkeypatch.setattr(client_module, "time", SimpleNamespace(monotonic=fake.monotonic))
    monkeypatch.setattr(client_module, "asyncio", SimpleNamespace(sleep=fake.sleep))
    return fake


@pytest.fixture
def gateway():
    return FakeGateway()


@pytest.fixture
def search(config, gateway, clock):
    """Run one client.search() against the fake gateway and return its result."""
    def _search(tool="search_knowledge", arguments=None):
        async def _run():
            transport = httpx.MockTransport(gateway.handler)
            async with LeeroopediaClient(config, transport=transport) as client:
                return await client.search(tool, arguments or {"query": "q"})
        return asyncio.run(_run())
    return _search
