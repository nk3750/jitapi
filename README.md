# JitAPI

**Just-in-Time API Orchestration for LLMs**

<!-- mcp-name: io.github.nk3750/jitapi -->

JitAPI is an MCP server that enables LLMs (like Claude) to interact with ANY API by dynamically discovering relevant endpoints from OpenAPI specifications. Instead of loading entire API specs into context, JitAPI uses semantic search and dependency graphs to find only the endpoints needed for each task.

## Features

- **Semantic Search**: Find relevant API endpoints using natural language queries
- **Dependency Graph**: Automatically detects endpoint dependencies (e.g., "POST /orders needs product_id from GET /products")
- **LLM-Powered Workflow Planning**: Uses GPT-4o-mini to extract parameters from queries and plan multi-step workflows
- **Generic Execution**: Executes workflows with automatic parameter passing between steps
- **Multi-API Support**: Register and query multiple APIs simultaneously
- **MCP Integration**: Native integration with Claude Code via Model Context Protocol

## Quick Start

### Installation

```bash
pip install jitapi
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uvx jitapi
```

> **Note for macOS users**: If you get an "externally-managed-environment" error with pip, use a virtual environment:
> ```bash
> python3 -m venv .venv
> source .venv/bin/activate
> pip install jitapi
> ```
> Or use `uvx jitapi` which handles this automatically.

### Setup with Claude Desktop

1. Locate your Claude Desktop config file:

   | OS | Path |
   |----|------|
   | macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
   | Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
   | Linux | `~/.config/Claude/claude_desktop_config.json` |

2. Add the JitAPI MCP server configuration:

```json
{
  "mcpServers": {
    "jitapi": {
      "command": "uvx",
      "args": ["jitapi"],
      "env": {
        "OPENAI_API_KEY": "sk-proj-your-key-here"
      }
    }
  }
}
```

3. Restart Claude Desktop
4. Look for "jitapi" in the MCP servers list (hammer icon)

### Setup with Claude Code

**Option A: Project-level config (Recommended)**

Create a `.mcp.json` file in your project directory:

```json
{
  "mcpServers": {
    "jitapi": {
      "command": "uvx",
      "args": ["jitapi"],
      "env": {
        "OPENAI_API_KEY": "sk-proj-your-key-here"
      }
    }
  }
}
```

Then start Claude Code from that directory. JitAPI will be available only in that project.

**Option B: Global config**

Edit `~/.claude.json` to make JitAPI available in all projects:

```json
{
  "mcpServers": {
    "jitapi": {
      "command": "uvx",
      "args": ["jitapi"],
      "env": {
        "OPENAI_API_KEY": "sk-proj-your-key-here"
      }
    }
  }
}
```

Restart Claude Code and verify by asking: "List available JitAPI tools"

**Alternative: Using pip-installed package**

If you installed via pip instead of using uvx:

```json
{
  "mcpServers": {
    "jitapi": {
      "command": "python",
      "args": ["-m", "jitapi"],
      "env": {
        "OPENAI_API_KEY": "sk-proj-your-key-here"
      }
    }
  }
}
```

**Alternative: From source (development)**

```json
{
  "mcpServers": {
    "jitapi": {
      "command": "python",
      "args": ["-m", "jitapi"],
      "cwd": "/path/to/jitapi",
      "env": {
        "OPENAI_API_KEY": "sk-proj-your-key-here",
        "PYTHONPATH": "/path/to/jitapi/src"
      }
    }
  }
}
```

### Usage with Claude Code

Once configured, you can use natural language to work with APIs:

```
User: "Register the Petstore API from https://petstore.swagger.io/v2/swagger.json"
Claude: [calls jitapi:register_api tool]
✓ Registered Swagger Petstore with 20 endpoints

User: "Find a pet and get its details"
Claude: [calls jitapi:get_workflow tool]
Workflow planned:
1. GET /pet/findByStatus - Find pets by status
2. GET /pet/{petId} - Get pet details
Parameters extracted: status="available" (from context)

User: "Execute that workflow"
Claude: [calls jitapi:set_api_auth, then jitapi:execute_workflow]
Step 1: Found 3 available pets
Step 2: Retrieved details for pet "Max"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                           │
│                                                                 │
│  OpenAPI Spec → Parser → Graph Builder → Embedder → Storage    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    RUNTIME PIPELINE                             │
│                                                                 │
│  Query → Vector Search → Graph Expansion → LLM Rerank →        │
│  Parameter Extraction → Workflow Execution                      │
└─────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Register an API**: Parse OpenAPI spec, build dependency graph, create embeddings
2. **Query**: User asks "get weather in Tokyo"
3. **Search**: Find semantically relevant endpoints
4. **Expand**: Add required dependencies (geocoding endpoint)
5. **Plan**: LLM extracts "Tokyo" as city parameter, maps data flow
6. **Execute**: Run workflow, passing coordinates from geocoding to weather API

## MCP Tools

| Tool | Description |
|------|-------------|
| `register_api` | Register an OpenAPI spec from URL |
| `list_apis` | List all registered APIs |
| `search_endpoints` | Semantic search for endpoints |
| `get_workflow` | Plan a workflow with parameter extraction |
| `execute_workflow` | Execute a planned workflow |
| `get_endpoint_schema` | Get detailed schema for an endpoint |
| `call_api` | Execute a single API call |
| `set_api_auth` | Configure API authentication |

## Configuration

### Setting the OpenAI API Key

JitAPI requires an OpenAI API key for embeddings and LLM-based workflow planning. You can set it in several ways:

**Option 1: In MCP Configuration (Recommended)**
```json
{
  "mcpServers": {
    "jitapi": {
      "command": "uvx",
      "args": ["jitapi"],
      "env": {
        "OPENAI_API_KEY": "sk-proj-your-key-here"
      }
    }
  }
}
```

**Option 2: Environment Variable**
```bash
# Add to ~/.zshrc or ~/.bashrc
export OPENAI_API_KEY="sk-proj-your-key-here"
```

**Option 3: .env File**

Create a `.env` file in one of these locations (checked in order):
1. Current working directory
2. `~/.jitapi/.env`
3. `~/.env`

```bash
# ~/.jitapi/.env
OPENAI_API_KEY=sk-proj-your-key-here
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for embeddings and reranking |
| `JITAPI_STORAGE_DIR` | No | Data storage directory (default: `~/.jitapi`) |
| `JITAPI_LOG_LEVEL` | No | Log level: DEBUG, INFO, WARNING, ERROR (default: INFO) |
| `JITAPI_LOG_FILE` | No | File path for logs (default: stderr) |

## Example: Weather API Workflow

```
User: "What's the weather in San Francisco?"

JitAPI plans the workflow:
1. GET /geo/1.0/direct
   - Parameters: q="San Francisco" (from user query)
   - Output mapping: lat=$[0].lat, lon=$[0].lon

2. GET /data/2.5/weather
   - Parameters: lat=step_1.lat, lon=step_1.lon
   - Returns: Current weather data

The LLM extracts "San Francisco" from the query and maps coordinates
between steps - no hardcoded logic required.
```

## Development

```bash
# Clone the repository
git clone https://github.com/nk3750/jitapi.git
cd jitapi

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/
```

## Project Structure

```
jitapi/
├── src/jitapi/
│   ├── main.py          # Entry point
│   ├── ingestion/       # OpenAPI parsing, graph building, embedding
│   ├── retrieval/       # Search, expansion, reranking
│   ├── execution/       # HTTP execution, workflow execution
│   ├── mcp/             # MCP server, tools, resources
│   └── stores/          # Data persistence
├── tests/               # Unit tests
└── pyproject.toml
```

## API Reference

### Registering an API

```python
# Via MCP tool
{
  "tool": "register_api",
  "arguments": {
    "api_id": "weather",
    "spec_url": "https://example.com/openapi.yaml"
  }
}
```

### Getting a Workflow

```python
# Via MCP tool
{
  "tool": "get_workflow",
  "arguments": {
    "query": "get the weather in Tokyo",
    "api_id": "weather"
  }
}

# Returns:
{
  "workflow_id": "abc123",
  "steps": [
    {
      "endpoint_id": "GET /geo/1.0/direct",
      "parameters": {
        "q": {"value": "Tokyo", "source": "user_query"}
      },
      "output_mapping": {
        "lat": "$[0].lat",
        "lon": "$[0].lon"
      }
    },
    ...
  ]
}
```

### Executing a Workflow

```python
# Via MCP tool
{
  "tool": "execute_workflow",
  "arguments": {
    "workflow_id": "abc123",
    "api_id": "weather"
  }
}
```

## Supported Authentication

- **API Key (Header)**: `X-API-Key`, or custom header name
- **API Key (Query)**: `?apikey=...`, or custom param name
- **Bearer Token**: `Authorization: Bearer ...`

## License

MIT

## Contributing

Contributions are welcome! Please read the contributing guidelines first.
