"""Configuration parsing and validation (config.py)."""

import pytest

from leeroopedia_mcp.config import (
    Config,
    ConfigError,
    get_config,
    validate_config_or_exit,
)


def test_defaults(clean_env):
    config = Config()

    assert config.api_key == ""
    assert config.api_url == "https://api.leeroopedia.com"
    assert config.poll_max_wait == 300
    assert config.poll_initial_interval == 0.5


def test_reads_environment_overrides(clean_env):
    clean_env.setenv("LEEROOPEDIA_API_KEY", "kpsk_abc")
    clean_env.setenv("LEEROOPEDIA_API_URL", "https://staging.example.com")
    clean_env.setenv("LEEROOPEDIA_POLL_MAX_WAIT", "45")
    clean_env.setenv("LEEROOPEDIA_POLL_INTERVAL", "0.25")

    config = Config()

    assert config.api_key == "kpsk_abc"
    assert config.api_url == "https://staging.example.com"
    assert config.poll_max_wait == 45
    assert config.poll_initial_interval == 0.25


def test_api_url_trailing_slash_is_stripped(clean_env):
    clean_env.setenv("LEEROOPEDIA_API_URL", "https://staging.example.com/")

    assert Config().api_url == "https://staging.example.com"


def test_missing_api_key_is_rejected(clean_env):
    with pytest.raises(ConfigError) as exc:
        get_config()

    # The message must tell the user which variable to set and where to get it
    assert "LEEROOPEDIA_API_KEY" in str(exc.value)
    assert "app.leeroopedia.com" in str(exc.value)


def test_get_config_returns_validated_config(config):
    assert get_config().api_key == config.api_key


def test_validate_or_exit_exits_with_readable_error(clean_env, capsys):
    with pytest.raises(SystemExit) as exc:
        validate_config_or_exit()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    # stdout carries the MCP protocol, so the error must go to stderr only
    assert captured.out == ""
    assert "Configuration error" in captured.err
    assert "LEEROOPEDIA_API_KEY" in captured.err


def test_validate_or_exit_returns_config_when_valid(config):
    assert validate_config_or_exit().api_key == config.api_key
