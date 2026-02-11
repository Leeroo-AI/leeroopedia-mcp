"""
Configuration for Leeroopedia MCP server.

Environment Variables:
    LEEROOPEDIA_API_KEY: Required. Your Leeroopedia API key.
    LEEROOPEDIA_API_URL: Optional. API URL (default: https://api.leeroopedia.com)
"""

import os
import sys


class ConfigError(Exception):
    """Raised when configuration is invalid."""
    pass


class Config:
    """Leeroopedia MCP configuration."""

    def __init__(self):
        self.api_key = os.getenv("LEEROOPEDIA_API_KEY", "")
        self.api_url = os.getenv("LEEROOPEDIA_API_URL", "https://api.leeroopedia.com").rstrip("/")

        # Polling config for async task API (Celery queue backend)
        # Max seconds to wait for a search task to complete
        self.poll_max_wait = int(os.getenv("LEEROOPEDIA_POLL_MAX_WAIT", "300"))
        # Initial poll interval in seconds (grows via exponential backoff)
        self.poll_initial_interval = float(os.getenv("LEEROOPEDIA_POLL_INTERVAL", "0.5"))

    def validate(self) -> None:
        """
        Validate configuration.

        Raises:
            ConfigError: If required configuration is missing
        """
        if not self.api_key:
            raise ConfigError(
                "LEEROOPEDIA_API_KEY environment variable is required.\n"
                "Get your API key at https://app.leeroopedia.com\n\n"
                "Set it in your MCP config:\n"
                '  "env": {"LEEROOPEDIA_API_KEY": "kpsk_..."}'
            )


def get_config() -> Config:
    """Get validated configuration."""
    config = Config()
    config.validate()
    return config


def validate_config_or_exit() -> Config:
    """Validate configuration or exit with error message."""
    try:
        return get_config()
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
