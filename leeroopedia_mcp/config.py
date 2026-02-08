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

    def validate(self) -> None:
        """
        Validate configuration.

        Raises:
            ConfigError: If required configuration is missing
        """
        if not self.api_key:
            raise ConfigError(
                "LEEROOPEDIA_API_KEY environment variable is required.\n"
                "Get your API key at https://leeroopedia.com/dashboard\n\n"
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
