from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.tools import StructuredTool, ToolException

from xyber_sdk.mcp_client.client import (
    McpClient,
    get_mcp_client,
    get_mcp_client_config,
)
from xyber_sdk.mcp_client.config import (
    McpClientConfig,
    McpConfigError,
    McpServerConfig,
    McpServerConnectionError,
    UnknownToolError,
)

# --- Fixtures ---


@pytest.fixture
def sample_config():
    """Create a sample McpClientConfig with multiple servers."""
    return McpClientConfig(
        servers={
            "tavily": McpServerConfig(url="http://localhost:8080/sse", transport="sse"),
            "qdrant": McpServerConfig(url="http://localhost:8081", transport="streamable_http"),
            "local": McpServerConfig(url="stdio://local", transport="stdio"),
        },
        _env_file=None,
    )


@pytest.fixture
def single_server_config():
    """Create a config with a single server."""
    return McpClientConfig(
        servers={
            "tavily": McpServerConfig(url="http://localhost:8080"),
        },
        _env_file=None,
    )


@pytest.fixture
def empty_config():
    """Create an empty McpClientConfig."""
    return McpClientConfig(servers={}, _env_file=None)


@pytest.fixture
def mock_tool():
    """Create a mock StructuredTool."""
    tool = MagicMock(spec=StructuredTool)
    tool.name = "web_search"
    return tool


@pytest.fixture
def mock_tools():
    """Create a list of mock StructuredTools."""
    tools = []
    for name in ["web_search", "get_content", "summarize"]:
        tool = MagicMock(spec=StructuredTool)
        tool.name = name
        tools.append(tool)
    return tools


@pytest.fixture
def mock_invokable_tool():
    """Create a mock StructuredTool with async invocation capability."""
    tool = MagicMock(spec=StructuredTool)
    tool.name = "invokable_tool"
    tool.ainvoke = AsyncMock(return_value={"result": "success"})
    return tool


# --- McpClient Initialization Tests ---


def test_mcp_client_init_with_config(sample_config):
    """McpClient initializes with configuration."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_adapter:
        client = McpClient(config=sample_config)

        assert client.config == sample_config
        assert client.httpx_client_factory is None
        mock_adapter.assert_called_once()


def test_mcp_client_init_creates_empty_tools_dict(sample_config):
    """McpClient creates empty tools dict for each server on init."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient"):
        client = McpClient(config=sample_config)

        assert "tavily" in client.tools
        assert "qdrant" in client.tools
        assert "local" in client.tools
        assert client.tools["tavily"] == {}
        assert client.tools["qdrant"] == {}
        assert client.tools["local"] == {}


def test_mcp_client_from_config_classmethod(sample_config):
    """McpClient.from_config creates instance correctly."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient"):
        client = McpClient.from_config(config=sample_config)

        assert isinstance(client, McpClient)
        assert client.config == sample_config


def test_mcp_client_with_httpx_factory(sample_config):
    """McpClient accepts custom httpx_client_factory."""
    mock_factory = MagicMock()

    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient"):
        client = McpClient(config=sample_config, httpx_client_factory=mock_factory)

        assert client.httpx_client_factory == mock_factory


def test_mcp_client_passes_factory_to_http_transports(sample_config):
    """McpClient passes httpx_client_factory to sse and streamable_http transports."""
    mock_factory = MagicMock()

    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_adapter:
        McpClient(config=sample_config, httpx_client_factory=mock_factory)

        call_args = mock_adapter.call_args[0][0]

        # SSE transport should have factory
        assert call_args["tavily"]["httpx_client_factory"] == mock_factory

        # streamable_http transport should have factory
        assert call_args["qdrant"]["httpx_client_factory"] == mock_factory

        # stdio transport should NOT have factory
        assert "httpx_client_factory" not in call_args["local"]


def test_mcp_client_no_factory_for_stdio_transport():
    """McpClient does not pass factory to stdio transport."""
    config = McpClientConfig(
        servers={
            "local": McpServerConfig(url="stdio://local", transport="stdio"),
        },
        _env_file=None,
    )
    mock_factory = MagicMock()

    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_adapter:
        McpClient(config=config, httpx_client_factory=mock_factory)

        call_args = mock_adapter.call_args[0][0]
        assert "httpx_client_factory" not in call_args["local"]


def test_mcp_client_empty_config_creates_empty_adapter(empty_config):
    """McpClient with empty config creates adapter with empty dict."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_adapter:
        client = McpClient(config=empty_config)

        mock_adapter.assert_called_once_with({})
        assert client.tools == {}


# --- Server Validation Tests ---


def test_validate_server_name_success(sample_config):
    """_validate_server_name passes for existing server."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient"):
        client = McpClient(config=sample_config)

        # Should not raise
        client._validate_server_name("tavily")
        client._validate_server_name("qdrant")
        client._validate_server_name("local")


def test_validate_server_name_raises_for_unknown_server(sample_config):
    """_validate_server_name raises McpConfigError for unknown server."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient"):
        client = McpClient(config=sample_config)

        with pytest.raises(McpConfigError) as exc_info:
            client._validate_server_name("unknown_server")

        assert "unknown_server" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()


def test_validate_server_name_error_suggests_env_check(sample_config):
    """_validate_server_name error message suggests checking env vars."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient"):
        client = McpClient(config=sample_config)

        with pytest.raises(McpConfigError) as exc_info:
            client._validate_server_name("missing")

        assert "MCP_SERVERS" in str(exc_info.value)


def test_validate_server_name_empty_string(sample_config):
    """_validate_server_name raises for empty server name."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient"):
        client = McpClient(config=sample_config)

        with pytest.raises(McpConfigError):
            client._validate_server_name("")


# --- Tool Loading Tests ---


@pytest.mark.asyncio
async def test_load_tools_from_server_success(single_server_config, mock_tools):
    """_load_tools_from_server loads tools into internal dict."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=mock_tools)
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)
        await client._load_tools_from_server("tavily")

        assert len(client.tools["tavily"]) == 3
        assert "web_search" in client.tools["tavily"]
        assert "get_content" in client.tools["tavily"]
        assert "summarize" in client.tools["tavily"]


@pytest.mark.asyncio
async def test_load_tools_from_server_empty_tools(single_server_config):
    """_load_tools_from_server handles server with no tools."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=[])
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)
        await client._load_tools_from_server("tavily")

        assert client.tools["tavily"] == {}


@pytest.mark.asyncio
async def test_load_tools_from_server_connection_error(single_server_config):
    """_load_tools_from_server raises McpServerConnectionError on failure."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(side_effect=ConnectionError("Network error"))
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)

        with pytest.raises(McpServerConnectionError) as exc_info:
            await client._load_tools_from_server("tavily")

        assert "tavily" in str(exc_info.value)
        assert "Network error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_load_tools_from_server_wraps_any_exception(single_server_config):
    """_load_tools_from_server wraps any exception as McpServerConnectionError."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(side_effect=ValueError("Unexpected error"))
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)

        with pytest.raises(McpServerConnectionError) as exc_info:
            await client._load_tools_from_server("tavily")

        assert "Unexpected error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_load_tools_from_server_preserves_original_exception(single_server_config):
    """_load_tools_from_server preserves original exception via __cause__.

    When langchain_mcp_adapter.get_tools() raises any exception, it should be
    wrapped in McpServerConnectionError with the original exception accessible
    via __cause__ for debugging purposes.
    """
    original_error = RuntimeError("Original connection failure")

    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(side_effect=original_error)
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)

        with pytest.raises(McpServerConnectionError) as exc_info:
            await client._load_tools_from_server("tavily")

        # Verify the original exception is preserved as __cause__
        assert exc_info.value.__cause__ is original_error
        assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_load_tools_from_server_wraps_httpx_connect_error(single_server_config):
    """_load_tools_from_server wraps httpx.ConnectError as McpServerConnectionError.

    This tests a common real-world scenario where the MCP server is unreachable.
    """
    original_error = httpx.ConnectError("Connection refused")

    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(side_effect=original_error)
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)

        with pytest.raises(McpServerConnectionError) as exc_info:
            await client._load_tools_from_server("tavily")

        assert exc_info.value.__cause__ is original_error
        assert "Connection refused" in str(exc_info.value)


# --- get_tool Tests ---


@pytest.mark.asyncio
async def test_get_tool_success(single_server_config, mock_tool):
    """get_tool returns requested tool when server and tool exist."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=[mock_tool])
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)
        tool = await client.get_tool("tavily", "web_search")

        assert tool == mock_tool


@pytest.mark.asyncio
async def test_get_tool_loads_on_first_access(single_server_config, mock_tool):
    """get_tool loads tools from server on first access."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=[mock_tool])
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)

        # Tools not loaded yet
        assert client.tools["tavily"] == {}

        await client.get_tool("tavily", "web_search")

        # Now loaded
        assert "web_search" in client.tools["tavily"]
        mock_adapter.get_tools.assert_called_once_with(server_name="tavily")


@pytest.mark.asyncio
async def test_get_tool_uses_cache(single_server_config, mock_tool):
    """get_tool uses cached tools on subsequent calls."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=[mock_tool])
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)

        # First call loads
        await client.get_tool("tavily", "web_search")
        # Second call uses cache
        await client.get_tool("tavily", "web_search")

        # Only called once
        mock_adapter.get_tools.assert_called_once()


@pytest.mark.asyncio
async def test_get_tool_unknown_server_raises_config_error(single_server_config):
    """get_tool raises McpConfigError for unknown server with meaningful message."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient"):
        client = McpClient(config=single_server_config)

        with pytest.raises(McpConfigError) as exc_info:
            await client.get_tool("unknown_server", "web_search")

        # Error message should mention the server name
        assert "unknown_server" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_tool_unknown_tool_raises_unknown_tool_error(single_server_config, mock_tool):
    """get_tool raises UnknownToolError for unknown tool with meaningful message."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=[mock_tool])
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)

        with pytest.raises(UnknownToolError) as exc_info:
            await client.get_tool("tavily", "nonexistent_tool")

        # Error message should mention the tool name and server
        assert "nonexistent_tool" in str(exc_info.value)
        assert "tavily" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_tool_unknown_tool_lists_available_tools(single_server_config, mock_tools):
    """UnknownToolError includes list of available tools from the server."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=mock_tools)
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)

        with pytest.raises(UnknownToolError) as exc_info:
            await client.get_tool("tavily", "nonexistent")

        error_msg = str(exc_info.value)
        # Should list available tools
        assert "web_search" in error_msg
        assert "get_content" in error_msg
        assert "summarize" in error_msg


@pytest.mark.asyncio
async def test_get_tool_reloads_if_tool_not_found_initially(single_server_config):
    """get_tool reloads tools if requested tool not in cache."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()

        # First tool in cache
        tool1 = MagicMock(spec=StructuredTool)
        tool1.name = "tool_one"

        # Second tool added later
        tool2 = MagicMock(spec=StructuredTool)
        tool2.name = "tool_two"

        mock_adapter.get_tools = AsyncMock(return_value=[tool1, tool2])
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)

        # Pre-populate cache with only tool_one
        client.tools["tavily"] = {"tool_one": tool1}

        # Request tool_two which is not in cache
        result = await client.get_tool("tavily", "tool_two")

        assert result == tool2
        mock_adapter.get_tools.assert_called_once()


# --- get_all_tools Tests ---


@pytest.mark.asyncio
async def test_get_all_tools_returns_tools_from_all_servers(sample_config):
    """get_all_tools returns all tools from all configured servers."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()

        tool1 = MagicMock(spec=StructuredTool)
        tool1.name = "tavily_search"
        tool2 = MagicMock(spec=StructuredTool)
        tool2.name = "qdrant_query"
        tool3 = MagicMock(spec=StructuredTool)
        tool3.name = "local_tool"

        # Return different tools for each server
        async def get_tools_side_effect(server_name):
            if server_name == "tavily":
                return [tool1]
            elif server_name == "qdrant":
                return [tool2]
            elif server_name == "local":
                return [tool3]
            return []

        mock_adapter.get_tools = AsyncMock(side_effect=get_tools_side_effect)
        mock_cls.return_value = mock_adapter

        client = McpClient(config=sample_config)
        all_tools = await client.get_all_tools()

        assert len(all_tools) == 3
        assert tool1 in all_tools
        assert tool2 in all_tools
        assert tool3 in all_tools


@pytest.mark.asyncio
async def test_get_all_tools_empty_config_returns_empty_list(empty_config):
    """get_all_tools returns empty list when no servers are configured."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient"):
        client = McpClient(config=empty_config)
        all_tools = await client.get_all_tools()

        assert all_tools == []


@pytest.mark.asyncio
async def test_get_all_tools_uses_cache(sample_config, mock_tool):
    """get_all_tools uses cached tools."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=[mock_tool])
        mock_cls.return_value = mock_adapter

        client = McpClient(config=sample_config)

        # Pre-populate cache
        client.tools["tavily"] = {"web_search": mock_tool}
        client.tools["qdrant"] = {"query": mock_tool}
        client.tools["local"] = {"local_tool": mock_tool}

        await client.get_all_tools()

        # Should not call get_tools since all servers have cached tools
        mock_adapter.get_tools.assert_not_called()


@pytest.mark.asyncio
async def test_get_all_tools_loads_missing_servers(sample_config, mock_tool):
    """get_all_tools loads tools only for servers without cached tools."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=[mock_tool])
        mock_cls.return_value = mock_adapter

        client = McpClient(config=sample_config)

        # Pre-populate only tavily
        client.tools["tavily"] = {"web_search": mock_tool}

        await client.get_all_tools()

        # Should load qdrant and local (empty dicts)
        assert mock_adapter.get_tools.call_count == 2


# --- get_all_tools_from_server Tests ---


@pytest.mark.asyncio
async def test_get_all_tools_from_server_success(single_server_config, mock_tools):
    """get_all_tools_from_server returns all tools from specified server."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=mock_tools)
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)
        tools = await client.get_all_tools_from_server("tavily")

        assert len(tools) == 3


@pytest.mark.asyncio
async def test_get_all_tools_from_server_empty_returns_empty_list(single_server_config):
    """get_all_tools_from_server returns empty list for server with no tools."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=[])
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)
        tools = await client.get_all_tools_from_server("tavily")

        assert tools == []


@pytest.mark.asyncio
async def test_get_all_tools_from_server_unknown_raises_config_error(
    single_server_config,
):
    """get_all_tools_from_server raises McpConfigError for unknown server."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient"):
        client = McpClient(config=single_server_config)

        with pytest.raises(McpConfigError) as exc_info:
            await client.get_all_tools_from_server("unknown")

        assert "unknown" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_all_tools_from_server_uses_cache(single_server_config, mock_tool):
    """get_all_tools_from_server uses cached tools."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=[mock_tool])
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)

        # Pre-populate cache
        client.tools["tavily"] = {"web_search": mock_tool}

        await client.get_all_tools_from_server("tavily")

        mock_adapter.get_tools.assert_not_called()


# --- Tool Invocation Error Tests ---


@pytest.mark.asyncio
async def test_tool_invocation_tool_exception_propagates(single_server_config):
    """When tool.ainvoke() raises ToolException, it propagates without wrapping.

    CURRENT BEHAVIOR DOCUMENTATION:
    When an MCP server returns an error response (isError=True), langchain_mcp_adapters
    raises ToolException. This exception propagates directly to the caller without
    any wrapping or retry logic in McpClient.

    This is intentional - ToolException indicates a tool-level error (e.g., invalid
    input, tool execution failure) rather than a connection error. The caller
    should handle this appropriately.
    """
    tool = MagicMock(spec=StructuredTool)
    tool.name = "failing_tool"
    tool_error = ToolException("Tool execution failed: invalid input")
    tool.ainvoke = AsyncMock(side_effect=tool_error)

    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=[tool])
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)

        # Get the tool first
        retrieved_tool = await client.get_tool("tavily", "failing_tool")

        # Invoke the tool - ToolException should propagate directly
        with pytest.raises(ToolException) as exc_info:
            await retrieved_tool.ainvoke({"query": "test"})

        assert "invalid input" in str(exc_info.value)
        # Verify it's the exact same exception, not wrapped
        assert exc_info.value is tool_error


@pytest.mark.asyncio
async def test_tool_invocation_connection_error_propagates(single_server_config):
    """When tool.ainvoke() raises connection error during invocation, it propagates.

    CURRENT BEHAVIOR DOCUMENTATION:
    When a connection error occurs during tool invocation (after the tool has been
    successfully loaded), the raw exception (e.g., httpx.ConnectError) propagates
    directly to the caller.

    This is different from connection errors during _load_tools_from_server(),
    which ARE wrapped in McpServerConnectionError. The distinction:
    - Loading tools: Connection errors are wrapped (configuration/setup phase)
    - Invoking tools: Connection errors propagate raw (runtime phase)

    Future enhancement: Consider adding retry logic or wrapping these errors.
    """
    tool = MagicMock(spec=StructuredTool)
    tool.name = "network_tool"
    connection_error = httpx.ConnectError("Connection refused")
    tool.ainvoke = AsyncMock(side_effect=connection_error)

    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=[tool])
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)

        # Get the tool first (this should succeed)
        retrieved_tool = await client.get_tool("tavily", "network_tool")

        # Invoke the tool - connection error should propagate directly
        with pytest.raises(httpx.ConnectError) as exc_info:
            await retrieved_tool.ainvoke({"query": "test"})

        assert "Connection refused" in str(exc_info.value)
        # Verify it's the exact same exception, not wrapped
        assert exc_info.value is connection_error


@pytest.mark.asyncio
async def test_tool_invocation_timeout_error_propagates(single_server_config):
    """When tool.ainvoke() raises timeout error during invocation, it propagates.

    CURRENT BEHAVIOR DOCUMENTATION:
    Timeout errors during tool invocation propagate directly without wrapping.
    No retry logic is implemented.
    """
    tool = MagicMock(spec=StructuredTool)
    tool.name = "slow_tool"
    timeout_error = httpx.TimeoutException("Request timed out")
    tool.ainvoke = AsyncMock(side_effect=timeout_error)

    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=[tool])
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)

        retrieved_tool = await client.get_tool("tavily", "slow_tool")

        with pytest.raises(httpx.TimeoutException) as exc_info:
            await retrieved_tool.ainvoke({"query": "test"})

        assert exc_info.value is timeout_error


@pytest.mark.asyncio
async def test_tool_invocation_success(single_server_config, mock_invokable_tool):
    """Tool invocation succeeds and returns expected result."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=[mock_invokable_tool])
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)

        retrieved_tool = await client.get_tool("tavily", "invokable_tool")
        result = await retrieved_tool.ainvoke({"query": "test"})

        assert result == {"result": "success"}
        mock_invokable_tool.ainvoke.assert_called_once_with({"query": "test"})


# --- Helper Function Tests ---


def test_get_mcp_client_config_returns_new_instance():
    """get_mcp_client_config returns new McpClientConfig instance."""
    with patch.dict("os.environ", {}, clear=True):
        config = get_mcp_client_config()

        assert isinstance(config, McpClientConfig)


def test_get_mcp_client_config_reads_env():
    """get_mcp_client_config reads from environment."""
    env_vars = {
        "MCP_SERVERS__TEST__URL": "http://test:8080",
    }

    with patch.dict("os.environ", env_vars, clear=True):
        config = get_mcp_client_config()

        assert "test" in config.servers


def test_get_mcp_client_returns_client_instance(single_server_config):
    """get_mcp_client returns McpClient instance."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient"):
        client = get_mcp_client(config=single_server_config)

        assert isinstance(client, McpClient)


def test_get_mcp_client_passes_factory(single_server_config):
    """get_mcp_client passes httpx_client_factory to client."""
    mock_factory = MagicMock()

    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient"):
        client = get_mcp_client(config=single_server_config, httpx_client_factory=mock_factory)

        assert client.httpx_client_factory == mock_factory


def test_get_mcp_client_uses_from_config(single_server_config):
    """get_mcp_client internally uses McpClient.from_config."""
    with patch("xyber_sdk.mcp_client.client.McpClient.from_config") as mock_from_config:
        mock_from_config.return_value = MagicMock()

        get_mcp_client(config=single_server_config)

        mock_from_config.assert_called_once_with(
            config=single_server_config, httpx_client_factory=None
        )


# --- Edge Cases ---


def test_mcp_client_server_name_case_sensitivity(sample_config):
    """McpClient server names are case-sensitive."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient"):
        client = McpClient(config=sample_config)

        # Lowercase works
        client._validate_server_name("tavily")

        # Uppercase fails
        with pytest.raises(McpConfigError):
            client._validate_server_name("TAVILY")


@pytest.mark.asyncio
async def test_get_tool_concurrent_access(single_server_config, mock_tool):
    """get_tool handles concurrent access correctly."""
    import asyncio

    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()
        mock_adapter.get_tools = AsyncMock(return_value=[mock_tool])
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)

        # Simulate concurrent requests
        results = await asyncio.gather(
            client.get_tool("tavily", "web_search"),
            client.get_tool("tavily", "web_search"),
            client.get_tool("tavily", "web_search"),
        )

        assert all(r == mock_tool for r in results)


@pytest.mark.asyncio
async def test_get_tool_with_special_chars_in_name(single_server_config):
    """get_tool handles tool names with special characters."""
    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_cls:
        mock_adapter = AsyncMock()

        special_tool = MagicMock(spec=StructuredTool)
        special_tool.name = "my-tool_v2.0"

        mock_adapter.get_tools = AsyncMock(return_value=[special_tool])
        mock_cls.return_value = mock_adapter

        client = McpClient(config=single_server_config)
        result = await client.get_tool("tavily", "my-tool_v2.0")

        assert result == special_tool


# --- httpx_client_factory Tests ---


def test_client_factory_passed_to_http_transports():
    """httpx_client_factory is passed to HTTP-based transports but not stdio."""
    config = McpClientConfig(
        servers={
            "http_server": McpServerConfig(
                url="http://localhost:8080", transport="streamable_http"
            ),
            "sse_server": McpServerConfig(url="http://localhost:8081", transport="sse"),
            "stdio_server": McpServerConfig(url="stdio://test", transport="stdio"),
        },
        _env_file=None,
    )

    mock_factory = MagicMock()

    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_adapter_cls:
        McpClient.from_config(config, httpx_client_factory=mock_factory)

        mock_adapter_cls.assert_called_once()
        args, _ = mock_adapter_cls.call_args
        server_configs = args[0]

        # HTTP-based transports should have the factory
        assert "http_server" in server_configs
        assert server_configs["http_server"]["httpx_client_factory"] == mock_factory

        assert "sse_server" in server_configs
        assert server_configs["sse_server"]["httpx_client_factory"] == mock_factory

        # stdio transport should NOT have the factory
        assert "stdio_server" in server_configs
        assert "httpx_client_factory" not in server_configs["stdio_server"]


def test_headers_passed_to_http_transports():
    """Headers are passed to HTTP-based transports but not stdio."""
    config = McpClientConfig(
        servers={
            "apify": McpServerConfig(
                url="https://mcp.apify.com/sse",
                transport="sse",
                headers={"Authorization": "Bearer apify-token"},
            ),
            "tavily": McpServerConfig(
                url="https://tavily.com/mcp",
                transport="streamable_http",
                headers={"X-API-Key": "tavily-key"},
            ),
            "local": McpServerConfig(
                url="stdio://test",
                transport="stdio",
                headers={"Should": "Be-Ignored"},  # headers on stdio should be ignored
            ),
        },
        _env_file=None,
    )

    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_adapter_cls:
        McpClient.from_config(config)

        mock_adapter_cls.assert_called_once()
        args, _ = mock_adapter_cls.call_args
        server_configs = args[0]

        # SSE transport should have headers
        assert "apify" in server_configs
        assert server_configs["apify"]["headers"] == {"Authorization": "Bearer apify-token"}

        # streamable_http transport should have headers
        assert "tavily" in server_configs
        assert server_configs["tavily"]["headers"] == {"X-API-Key": "tavily-key"}

        # stdio transport should NOT have headers
        assert "local" in server_configs
        assert "headers" not in server_configs["local"]


def test_headers_not_passed_when_none():
    """Headers key is not added when headers is None."""
    config = McpClientConfig(
        servers={
            "no_headers": McpServerConfig(
                url="https://example.com/mcp",
                transport="sse",
                headers=None,
            ),
        },
        _env_file=None,
    )

    with patch("xyber_sdk.mcp_client.client.MultiServerMCPClient") as mock_adapter_cls:
        McpClient.from_config(config)

        mock_adapter_cls.assert_called_once()
        args, _ = mock_adapter_cls.call_args
        server_configs = args[0]

        # Headers key should not exist when None
        assert "no_headers" in server_configs
        assert "headers" not in server_configs["no_headers"]
