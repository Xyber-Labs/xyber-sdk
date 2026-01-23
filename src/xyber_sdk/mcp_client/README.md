# MCP Client

Connect to Model Context Protocol (MCP) servers and access their tools.

## Prerequisites

- [langchain-mcp-adapters](https://pypi.org/project/langchain-mcp-adapters/) - MCP protocol adapters for LangChain

## Public API

```python
from xyber_sdk.mcp_client import (
    McpClient,
    get_mcp_client,
    get_mcp_client_config,
    McpClientConfig,
    McpServerConfig,
    # Exceptions
    McpClientError,
    McpServerConnectionError,
    UnknownToolError,
)
```

## Configuration

### Environment Variables

**JSON format (recommended):**
```bash
MCP_SERVERS='{"web_search": {"url": "http://localhost:8080/sse", "transport": "sse"}}'
```

**Nested format:**
```bash
MCP_SERVERS__WEB_SEARCH__URL=http://localhost:8080/sse
MCP_SERVERS__WEB_SEARCH__TRANSPORT=sse
MCP_SERVERS__DATABASE__URL=http://localhost:9000/mcp
MCP_SERVERS__DATABASE__TRANSPORT=streamable_http
```

### Manual Configuration

```python
from xyber_sdk.mcp_client import McpClientConfig, McpServerConfig

config = McpClientConfig(
    servers={
        "web_search": McpServerConfig(
            url="http://localhost:8080/sse",
            transport="sse"
        ),
        "database": McpServerConfig(
            url="http://localhost:9000/mcp",
            transport="streamable_http"
        ),
    }
)
client = McpClient.from_config(config)
```

### Transport Types

| Transport | Use Case |
|-----------|----------|
| `sse` | Server-Sent Events (most common) |
| `streamable_http` | HTTP with streaming support |
| `stdio` | Local process communication |

## Usage Scenarios

### Scenario 1: Basic Setup

```python
from xyber_sdk.mcp_client import McpClient, get_mcp_client_config

# Load config from environment
config = get_mcp_client_config()

# Create client
client = McpClient.from_config(config)
```

### Scenario 2: Getting and Using Tools

```python
# Get a specific tool from a specific server
tool = await client.get_tool("web_search", "search")
result = await tool.ainvoke({"query": "LangGraph tutorials"})

# Get all tools from one server
search_tools = await client.get_all_tools_from_server("web_search")

# Get all tools from all configured servers
all_tools = await client.get_all_tools()
```

### Scenario 3: Using Tools with LangGraph

```python
from langgraph.prebuilt import create_react_agent
from xyber_sdk.model_registry import get_model, SupportedModels
from xyber_sdk.mcp_client import McpClient, get_mcp_client_config

# Setup
llm = get_model(SupportedModels.GEMINI_2_0_FLASH)
mcp_client = McpClient.from_config(get_mcp_client_config())

# Get tools and bind to agent
tools = await mcp_client.get_all_tools()
agent = create_react_agent(llm, tools)
```

## Error Handling

```python
from xyber_sdk.mcp_client import (
    McpClientError,
    McpServerConnectionError,
    UnknownToolError,
)

try:
    tool = await client.get_tool("server", "tool")
except McpServerConnectionError as e:
    print(f"Cannot connect to server: {e}")
except UnknownToolError as e:
    print(f"Tool not found: {e}")
except McpClientError as e:
    print(f"General MCP error: {e}")
```
