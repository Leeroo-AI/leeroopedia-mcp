"""The package version has a single source: pyproject.toml."""

import importlib
import importlib.metadata

import httpx

import leeroopedia_mcp
from leeroopedia_mcp.client import LeeroopediaClient


def test_version_matches_installed_package_metadata():
    assert leeroopedia_mcp.__version__ == importlib.metadata.version("leeroopedia-mcp")


def test_user_agent_reports_the_package_version(config):
    client = LeeroopediaClient(config, transport=httpx.MockTransport(lambda request: None))

    assert client.client.headers["User-Agent"] == f"leeroopedia-mcp/{leeroopedia_mcp.__version__}"


def test_version_falls_back_when_package_is_not_installed(monkeypatch):
    def not_installed(name):
        raise importlib.metadata.PackageNotFoundError(name)

    try:
        monkeypatch.setattr(importlib.metadata, "version", not_installed)
        importlib.reload(leeroopedia_mcp)

        assert leeroopedia_mcp.__version__ == "0.0.0+unknown"
    finally:
        # Restore the real version for the rest of the suite
        monkeypatch.undo()
        importlib.reload(leeroopedia_mcp)
