import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from xyber_sdk.mcp_client.config import McpClientConfig


@pytest.fixture
def clean_env():
    """Fixture that provides a clean environment (no MCP_ variables)."""
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.fixture
def single_server_env():
    """Fixture with one server configured via env vars."""
    env_vars = {
        "MCP_SERVERS__GENERATE_IMAGE__URL": "http://localhost:8080/sse",
        "MCP_SERVERS__GENERATE_IMAGE__TRANSPORT": "sse",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def multiple_servers_env():
    """Fixture with multiple servers configured via env vars."""
    env_vars = {
        "MCP_SERVERS__GENERATE_IMAGE__URL": "http://localhost:8080/sse",
        "MCP_SERVERS__GENERATE_IMAGE__TRANSPORT": "sse",
        "MCP_SERVERS__QDRANT_MEMORY__URL": "http://localhost:8002/sse",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


@pytest.fixture
def required_fields_only_env():
    """Fixture with servers having only required fields (url)."""
    env_vars = {
        "MCP_SERVERS__TAVILY__URL": "http://localhost:8080",
        "MCP_SERVERS__QDRANT__URL": "http://localhost:8081",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        yield env_vars


# --- Happy Path Tests ---


def test_multiple_servers_with_nested_delimiter(multiple_servers_env):
    """McpClientConfig parses multiple servers from MCP_SERVERS__* env vars."""
    config = McpClientConfig(_env_file=None)

    assert len(config.servers) == 2

    assert "generate_image" in config.servers
    assert config.servers["generate_image"].url == "http://localhost:8080/sse"
    assert config.servers["generate_image"].transport == "sse"

    assert "qdrant_memory" in config.servers
    assert config.servers["qdrant_memory"].url == "http://localhost:8002/sse"
    assert config.servers["qdrant_memory"].transport == "streamable_http"  # default


def test_single_server_with_nested_delimiter(single_server_env):
    """McpClientConfig parses a single server from MCP_SERVERS__* env vars."""
    config = McpClientConfig(_env_file=None)

    assert len(config.servers) == 1
    assert "generate_image" in config.servers
    assert config.servers["generate_image"].url == "http://localhost:8080/sse"
    assert config.servers["generate_image"].transport == "sse"


def test_required_fields_only_uses_defaults(required_fields_only_env):
    """McpClientConfig uses default transport when only url is provided."""
    config = McpClientConfig(_env_file=None)

    assert len(config.servers) == 2

    assert config.servers["tavily"].url == "http://localhost:8080"
    assert config.servers["tavily"].transport == "streamable_http"

    assert config.servers["qdrant"].url == "http://localhost:8081"
    assert config.servers["qdrant"].transport == "streamable_http"


# --- Edge Case Tests ---


def test_no_env_vars_returns_empty_servers(clean_env):
    """McpClientConfig defaults to empty servers dict when no env vars set."""
    config = McpClientConfig(_env_file=None)

    assert config.servers == {}


def test_empty_env_var_values():
    """McpClientConfig handles env vars with empty string values."""
    env_vars = {
        "MCP_SERVERS__EMPTY_SERVER__URL": "",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        config = McpClientConfig(_env_file=None)

        assert "empty_server" in config.servers
        assert config.servers["empty_server"].url == ""
        assert config.servers["empty_server"].transport == "streamable_http"


# --- Bad Case Tests ---


def test_invalid_transport_value_raises_validation_error():
    """McpClientConfig raises ValidationError for invalid transport values."""
    env_vars = {
        "MCP_SERVERS__BAD_SERVER__URL": "http://localhost:8080",
        "MCP_SERVERS__BAD_SERVER__TRANSPORT": "invalid_transport",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        with pytest.raises(ValidationError) as exc_info:
            McpClientConfig(_env_file=None)

        assert "transport" in str(exc_info.value).lower()


def test_bad_delimiter_is_ignored(clean_env):
    """Env vars with wrong delimiters (single underscore) are ignored."""
    env_vars = {
        "MCP_SERVERS_BAD_SERVER_URL": "http://localhost:8080",  # single underscores
    }
    with patch.dict(os.environ, env_vars, clear=True):
        config = McpClientConfig(_env_file=None)

        # Should not parse - wrong delimiter format
        assert config.servers == {}


# --- JSON Format Tests ---


def test_json_format_parsing(clean_env):
    """McpClientConfig parses servers from MCP_SERVERS JSON string."""
    json_config = '{"web_search": {"url": "http://localhost:8080/sse", "transport": "sse"}, "database": {"url": "http://localhost:9000/mcp"}}'
    env_vars = {"MCP_SERVERS": json_config}

    with patch.dict(os.environ, env_vars, clear=True):
        config = McpClientConfig(_env_file=None)

        assert len(config.servers) == 2

        assert "web_search" in config.servers
        assert config.servers["web_search"].url == "http://localhost:8080/sse"
        assert config.servers["web_search"].transport == "sse"

        assert "database" in config.servers
        assert config.servers["database"].url == "http://localhost:9000/mcp"
        assert config.servers["database"].transport == "streamable_http"  # default


def test_json_format_with_headers(clean_env):
    """McpClientConfig parses headers from MCP_SERVERS JSON string."""
    json_config = """{
        "apify": {
            "url": "https://mcp.apify.com/sse",
            "transport": "sse",
            "headers": {"Authorization": "Bearer test-token"}
        },
        "tavily": {
            "url": "https://tavily.com/mcp",
            "headers": {"X-API-Key": "tavily-key", "X-Custom": "value"}
        }
    }"""
    env_vars = {"MCP_SERVERS": json_config}

    with patch.dict(os.environ, env_vars, clear=True):
        config = McpClientConfig(_env_file=None)

        assert len(config.servers) == 2

        # Check apify headers
        assert config.servers["apify"].headers == {"Authorization": "Bearer test-token"}

        # Check tavily headers (multiple)
        assert config.servers["tavily"].headers == {
            "X-API-Key": "tavily-key",
            "X-Custom": "value",
        }


def test_json_format_without_headers_defaults_to_none(clean_env):
    """McpClientConfig defaults headers to None when not specified."""
    json_config = '{"simple": {"url": "http://localhost:8080"}}'
    env_vars = {"MCP_SERVERS": json_config}

    with patch.dict(os.environ, env_vars, clear=True):
        config = McpClientConfig(_env_file=None)

        assert config.servers["simple"].headers is None


# --- Nested Format with Headers Tests ---


def test_nested_format_with_headers(clean_env):
    """McpClientConfig parses headers from nested env vars.

    Note: pydantic-settings lowercases env var keys, so AUTHORIZATION becomes authorization.
    HTTP headers are case-insensitive per RFC 7230, so this is acceptable.
    """
    env_vars = {
        "MCP_SERVERS__APIFY__URL": "https://mcp.apify.com/sse",
        "MCP_SERVERS__APIFY__TRANSPORT": "sse",
        "MCP_SERVERS__APIFY__HEADERS__AUTHORIZATION": "Bearer test-token",
    }

    with patch.dict(os.environ, env_vars, clear=True):
        config = McpClientConfig(_env_file=None)

        assert "apify" in config.servers
        assert config.servers["apify"].url == "https://mcp.apify.com/sse"
        assert config.servers["apify"].transport == "sse"
        # pydantic-settings lowercases keys; HTTP headers are case-insensitive anyway
        assert config.servers["apify"].headers == {"authorization": "Bearer test-token"}
