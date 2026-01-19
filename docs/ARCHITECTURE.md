# JitAPI: Architecture & Workflow Documentation

> **JitAPI** (Sanskrit: "conversation") - A Just-in-Time API Orchestration System for LLMs

---

## Executive Summary

JitAPI is an MCP (Model Context Protocol) server that enables Claude to discover, understand, and execute workflows across any REST API. It transforms natural language queries into multi-step API calls by combining:

- **Semantic Search** - Find relevant endpoints using vector embeddings
- **Dependency Analysis** - Understand data flow between endpoints
- **LLM Planning** - Intelligent workflow orchestration with parameter extraction
- **Generic Execution** - Run any workflow without API-specific code

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Directory Structure](#2-directory-structure)
3. [Core Components](#3-core-components)
4. [Data Flow Pipelines](#4-data-flow-pipelines)
5. [Storage Layer](#5-storage-layer)
6. [MCP Interface](#6-mcp-interface)
7. [Workflow Execution](#7-workflow-execution)
8. [Configuration](#8-configuration)
9. [Example Walkthrough](#9-example-walkthrough)

---

## 1. System Architecture

### High-Level Overview

```
                                    JITAPI ARCHITECTURE

    +------------------+     +--------------------------------------------------+
    |   Claude Code    |     |                  JITAPI MCP SERVER              |
    |                  |     |                                                  |
    |  User: "Get      |     |  +------------+  +------------+  +------------+  |
    |   weather in     |────>|  |   MCP      |  | Retrieval  |  | Execution  |  |
    |   Tokyo"         |     |  |  Tools     |  |  Pipeline  |  |  Pipeline  |  |
    |                  |     |  +-----+------+  +-----+------+  +-----+------+  |
    +------------------+     |        |              |              |           |
                             |        v              v              v           |
                             |  +------------+  +------------+  +------------+  |
                             |  | Ingestion  |  |  Storage   |  |   HTTP     |  |
                             |  |  Pipeline  |  |   Layer    |  |  Executor  |  |
                             |  +------------+  +------------+  +------------+  |
                             +--------------------------------------------------+
                                       |              |              |
                                       v              v              v
                             +------------+  +------------+  +------------+
                             |  OpenAI    |  |  ChromaDB  |  |  Target    |
                             |  API       |  |  + JSON    |  |  APIs      |
                             +------------+  +------------+  +------------+
```

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **API-Agnostic** | Works with any OpenAPI 3.x or Swagger 2.0 spec |
| **LLM-Powered** | Uses GPT-4o-mini for intelligent workflow planning |
| **Semantic Search** | ChromaDB + OpenAI embeddings for endpoint discovery |
| **Dependency-Aware** | NetworkX graphs track parameter flow between endpoints |
| **Generic Execution** | JSONPath-based data flow, no hardcoded logic |

---

## 2. Directory Structure

### Source Code Layout

```
jitapi/
├── pyproject.toml                    # Package metadata, dependencies, entry points
├── README.md                         # User documentation
│
├── src/jitapi/
│   ├── __init__.py
│   ├── __main__.py                   # Entry point: python -m jitapi
│   ├── main.py                       # Server initialization, CLI entry
│   │
│   ├── mcp/                          # MCP Server Layer
│   │   ├── server.py                 # JitAPIServer - main MCP server class
│   │   ├── tools.py                  # ToolRegistry - 9 MCP tools
│   │   ├── models.py                 # Pydantic input validation models
│   │   └── resources.py              # MCP resources (API specs, endpoints)
│   │
│   ├── ingestion/                    # API Ingestion Pipeline
│   │   ├── parser.py                 # OpenAPIParser - spec parsing
│   │   ├── indexer.py                # APIIndexer - orchestrates ingestion
│   │   ├── embedder.py               # EndpointEmbedder - OpenAI embeddings
│   │   └── graph_builder.py          # DependencyGraphBuilder - NetworkX graphs
│   │
│   ├── stores/                       # Storage Layer
│   │   ├── spec_store.py             # SpecStore - parsed specs & metadata
│   │   ├── vector_store.py           # VectorStore - ChromaDB embeddings
│   │   └── graph_store.py            # GraphStore - dependency graphs
│   │
│   ├── retrieval/                    # Retrieval Pipeline
│   │   ├── vector_search.py          # VectorSearcher - semantic search
│   │   ├── graph_expander.py         # GraphExpander - dependency expansion
│   │   └── reranker.py               # LLMReranker - workflow planning
│   │
│   └── execution/                    # Execution Layer
│       ├── http_executor.py          # HTTPExecutor - makes API calls
│       ├── auth_handler.py           # AuthHandler - credential management
│       ├── schema_formatter.py       # SchemaFormatter - LLM-friendly schemas
│       └── workflow_executor.py      # WorkflowExecutor - multi-step execution
│
├── tests/                            # Test suite
│   ├── test_parser.py
│   ├── test_graph_builder.py
│   ├── test_retrieval.py
│   └── test_mcp_server.py
│
└── docs/                             # Documentation
```

### Data Storage Structure

```
~/.jitapi/                           # Default storage root
├── specs/{api_id}.json               # Raw OpenAPI specs
├── endpoints/{api_id}.json           # Parsed endpoint data
├── graphs/{api_id}.json              # Dependency graphs (NetworkX)
├── chroma/                           # ChromaDB vector database
├── apis.json                         # API metadata index
└── auth.json                         # Authentication credentials
```

---

## 3. Core Components

### 3.1 MCP Server (`mcp/server.py`)

**Class:** `JitAPIServer`

The main entry point that initializes all components and registers MCP handlers.

```python
class JitAPIServer:
    def __init__(self, storage_dir: Path):
        # Initialize all stores
        self.spec_store = SpecStore(storage_dir)
        self.vector_store = VectorStore(storage_dir)
        self.graph_store = GraphStore(storage_dir)
        self.auth_handler = AuthHandler(storage_dir)

        # Initialize pipelines
        self.indexer = APIIndexer(...)
        self.searcher = VectorSearcher(...)
        self.reranker = LLMReranker(...)
        self.executor = WorkflowExecutor(...)

        # Register MCP tools
        self.tool_registry = ToolRegistry(...)
```

### 3.2 OpenAPI Parser (`ingestion/parser.py`)

**Class:** `OpenAPIParser`

Parses OpenAPI 3.x and Swagger 2.0 specifications into normalized `Endpoint` objects.

**Key Data Classes:**

```python
@dataclass
class Endpoint:
    endpoint_id: str          # "GET /users/{id}"
    path: str                 # "/users/{id}"
    method: str               # "GET"
    summary: str              # "Get user by ID"
    description: str
    parameters: list[Parameter]
    request_body: RequestBody | None
    responses: dict[str, Response]
    tags: list[str]
    operation_id: str | None
    deprecated: bool
    servers: list[str]
    required_params: list[str]    # For dependency detection
    returned_fields: list[str]    # For dependency detection

@dataclass
class Parameter:
    name: str
    location: str             # "path", "query", "header", "cookie"
    required: bool
    schema: dict
    description: str
```

**Capabilities:**
- Handles `$ref` resolution with caching
- Extracts `required_params` and `returned_fields` for dependency analysis
- Normalizes base URLs across spec versions

### 3.3 Dependency Graph Builder (`ingestion/graph_builder.py`)

**Class:** `DependencyGraphBuilder`

Analyzes parameter flows between endpoints to build a directed dependency graph.

**Dependency Detection Strategies:**

| Strategy | Description | Confidence |
|----------|-------------|------------|
| Direct Field Match | Parameter name matches a returned field | 0.8 |
| Entity Mapping | `user_id` → looks for `/users` endpoints | 0.7 |
| Tag-Based Match | Parameter entity matches endpoint tags | 0.5 |

**Graph Structure:**
- **Nodes:** Endpoint IDs (e.g., `"POST /orders"`)
- **Edges:** Dependencies with metadata:
  ```json
  {
    "parameter": "product_id",
    "type": "body_param",
    "confidence": 0.8
  }
  ```

### 3.4 Endpoint Embedder (`ingestion/embedder.py`)

**Class:** `EndpointEmbedder`

Generates semantic embeddings for endpoints using OpenAI's `text-embedding-3-small`.

**Embedding Text Format:**
```
{METHOD} {path} | {summary} | {description[:500]} | {operation_id_words} |
Category: {tags} | Parameters: {param_names} | Requires: {required_params}
```

**Example:**
```
POST /orders | Create order | Category: orders | Parameters: product_id, quantity | Requires: product_id, user_id
```

### 3.5 Vector Searcher (`retrieval/vector_search.py`)

**Class:** `VectorSearcher`

Performs semantic search over endpoint embeddings.

**Features:**
- Cosine similarity search via ChromaDB
- Query expansion (synonyms: "create" → "add", "get" → "fetch")
- Filtering by API, method, tags, deprecated status
- Minimum score thresholds

### 3.6 Graph Expander (`retrieval/graph_expander.py`)

**Class:** `GraphExpander`

Augments search results with dependent endpoints from the dependency graph.

**Process:**
1. Take vector search results (top-10)
2. For each endpoint, find dependencies in graph
3. Add dependencies to result set (up to max_depth=3)
4. Adjust scores: `dependency_score = base_score * 0.8 * confidence`

### 3.7 LLM Reranker (`retrieval/reranker.py`)

**Class:** `LLMReranker`

The intelligence center - uses GPT-4o-mini to plan workflows.

**Key Innovation: Parameter Extraction**

The reranker extracts actual parameter values from the user query:

```json
{
  "steps": [
    {
      "endpoint_id": "GET /geo/1.0/direct",
      "purpose": "Get coordinates for the city",
      "parameters": {
        "q": {"value": "Tokyo", "source": "user_query"}
      },
      "output_mapping": {
        "lat": "$[0].lat",
        "lon": "$[0].lon"
      }
    },
    {
      "endpoint_id": "GET /data/2.5/weather",
      "parameters": {
        "lat": {"value": null, "source": "step_1.lat"},
        "lon": {"value": null, "source": "step_1.lon"}
      }
    }
  ]
}
```

**Parameter Sources:**

| Source | Description | Example |
|--------|-------------|---------|
| `user_query` | Extracted from natural language | `"Tokyo"` from "weather in Tokyo" |
| `step_N.field` | From previous step output via JSONPath | `step_1.lat` |
| `literal` | Default/hardcoded value | `"en"` for language |

### 3.8 Workflow Executor (`execution/workflow_executor.py`)

**Class:** `WorkflowExecutor`

Executes multi-step workflows with automatic parameter resolution.

**Execution Loop:**
```python
for step in workflow.steps:
    # 1. Resolve parameters
    params = resolve_parameters(step, previous_outputs)

    # 2. Categorize into path/query/body
    path_params, query_params, body = categorize_params(params, endpoint)

    # 3. Execute HTTP call
    result = http_executor.call_endpoint(endpoint, path_params, query_params, body)

    # 4. Extract output data via JSONPath
    outputs[step_num] = extract_outputs(result, step.output_mapping)
```

### 3.9 HTTP Executor (`execution/http_executor.py`)

**Class:** `HTTPExecutor`

Makes authenticated HTTP requests to target APIs.

**Features:**
- Path parameter substitution (`/users/{id}` → `/users/123`)
- Query parameter encoding
- JSON request body handling
- Authentication injection via `AuthHandler`
- Error extraction from responses
- Async execution via `httpx`

### 3.10 Auth Handler (`execution/auth_handler.py`)

**Class:** `AuthHandler`

Manages API credentials and injects them into requests.

**Supported Auth Types:**

| Type | Header/Param | Example |
|------|--------------|---------|
| `api_key` | Custom header | `X-API-Key: abc123` |
| `api_key_query` | Query parameter | `?api_key=abc123` |
| `bearer` | Authorization header | `Bearer token123` |
| `basic` | Authorization header | `Basic base64(user:pass)` |
| `custom_header` | Any headers | `{"X-Custom": "value"}` |

---

## 4. Data Flow Pipelines

### 4.1 Ingestion Pipeline (API Registration)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                               │
│                                                                          │
│   OpenAPI Spec URL                                                       │
│         │                                                                │
│         ▼                                                                │
│   ┌─────────────────┐                                                   │
│   │  OpenAPIParser  │ ─── Fetch spec, parse endpoints, resolve $refs    │
│   └────────┬────────┘                                                   │
│            │                                                             │
│            ▼                                                             │
│   ┌─────────────────┐                                                   │
│   │   SpecStore     │ ─── Store raw spec + parsed endpoints as JSON     │
│   └────────┬────────┘                                                   │
│            │                                                             │
│            ▼                                                             │
│   ┌─────────────────┐                                                   │
│   │ GraphBuilder    │ ─── Analyze dependencies, build NetworkX graph    │
│   └────────┬────────┘                                                   │
│            │                                                             │
│            ▼                                                             │
│   ┌─────────────────┐                                                   │
│   │  GraphStore     │ ─── Persist graph as JSON                         │
│   └────────┬────────┘                                                   │
│            │                                                             │
│            ▼                                                             │
│   ┌─────────────────┐                                                   │
│   │ EndpointEmbedder│ ─── Generate embeddings via OpenAI                │
│   └────────┬────────┘                                                   │
│            │                                                             │
│            ▼                                                             │
│   ┌─────────────────┐                                                   │
│   │  VectorStore    │ ─── Store in ChromaDB                             │
│   └─────────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Retrieval Pipeline (Query Processing)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL PIPELINE                               │
│                                                                          │
│   User Query: "Get weather in Tokyo"                                     │
│         │                                                                │
│         ▼                                                                │
│   ┌─────────────────┐                                                   │
│   │ VectorSearcher  │ ─── Embed query, search ChromaDB (top-10)         │
│   └────────┬────────┘                                                   │
│            │                                                             │
│            ▼                                                             │
│   ┌─────────────────┐                                                   │
│   │ GraphExpander   │ ─── Add dependent endpoints from graph            │
│   └────────┬────────┘                                                   │
│            │                                                             │
│            ▼                                                             │
│   ┌─────────────────┐                                                   │
│   │  LLMReranker    │ ─── GPT-4o-mini orders steps, extracts params     │
│   └────────┬────────┘                                                   │
│            │                                                             │
│            ▼                                                             │
│   ┌─────────────────┐                                                   │
│   │ SchemaFormatter │ ─── Format schemas for execution                  │
│   └────────┬────────┘                                                   │
│            │                                                             │
│            ▼                                                             │
│   RerankedWorkflow {workflow_id, steps[], reasoning}                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Execution Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXECUTION PIPELINE                               │
│                                                                          │
│   RerankedWorkflow                                                       │
│         │                                                                │
│         ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    WorkflowExecutor                              │   │
│   │                                                                  │   │
│   │   For each step:                                                 │   │
│   │   ┌─────────────────┐                                           │   │
│   │   │ Resolve Params  │ ─── user_query → direct value             │   │
│   │   │                 │     step_N.field → JSONPath extraction    │   │
│   │   └────────┬────────┘                                           │   │
│   │            │                                                     │   │
│   │            ▼                                                     │   │
│   │   ┌─────────────────┐                                           │   │
│   │   │ Categorize      │ ─── path_params, query_params, body       │   │
│   │   └────────┬────────┘                                           │   │
│   │            │                                                     │   │
│   │            ▼                                                     │   │
│   │   ┌─────────────────┐     ┌─────────────────┐                   │   │
│   │   │ HTTPExecutor    │◄────│  AuthHandler    │                   │   │
│   │   │                 │     │ (inject creds)  │                   │   │
│   │   └────────┬────────┘     └─────────────────┘                   │   │
│   │            │                                                     │   │
│   │            ▼                                                     │   │
│   │   ┌─────────────────┐                                           │   │
│   │   │ Extract Outputs │ ─── JSONPath on response                  │   │
│   │   └────────┬────────┘                                           │   │
│   │            │                                                     │   │
│   │            ▼                                                     │   │
│   │   Store for next step                                            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│            │                                                             │
│            ▼                                                             │
│   WorkflowResult {steps[], final_result}                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Storage Layer

### 5.1 SpecStore

**Purpose:** Persists raw OpenAPI specs and parsed endpoint data

**Files:**
- `specs/{api_id}.json` - Raw OpenAPI spec
- `endpoints/{api_id}.json` - Parsed `Endpoint` objects
- `apis.json` - API metadata index

**Key Methods:**

| Method | Description |
|--------|-------------|
| `store_spec()` | Save parsed spec with metadata |
| `get_endpoints()` | Retrieve all endpoints for an API |
| `get_endpoint()` | Get specific endpoint by ID |
| `list_apis()` | List all registered APIs |
| `delete_api()` | Clean up API data |

### 5.2 VectorStore

**Purpose:** Semantic search over endpoint embeddings

**Backend:** ChromaDB (persistent)

**Collection:** `"endpoints"` with cosine similarity

**Document ID Format:** `"{api_id}::{endpoint_id}"`

**Metadata per Embedding:**
```json
{
  "api_id": "openweather",
  "endpoint_id": "GET /data/2.5/weather",
  "path": "/data/2.5/weather",
  "method": "GET",
  "summary": "Get current weather",
  "tags": "weather,current",
  "operation_id": "getCurrentWeather",
  "deprecated": false,
  "param_count": 3,
  "has_request_body": false
}
```

### 5.3 GraphStore

**Purpose:** Endpoint dependency graph storage

**Backend:** NetworkX DiGraph serialized to JSON (`node_link_data` format)

**Key Methods:**

| Method | Description |
|--------|-------------|
| `store_graph()` | Persist dependency graph |
| `get_graph()` | Retrieve with in-memory caching |
| `get_dependencies()` | Find what an endpoint depends on |
| `get_dependents()` | Find what depends on an endpoint |
| `find_providers_for_param()` | Find endpoints providing a parameter |
| `get_full_dependency_chain()` | DFS traversal (max 3 levels) |

### 5.4 AuthHandler

**Purpose:** Credential storage and injection

**File:** `auth.json`

**Storage Format:**
```json
{
  "openweather": {
    "auth_type": "api_key_query",
    "api_key": "abc123...",
    "api_key_param": "appid"
  },
  "stripe": {
    "auth_type": "bearer",
    "access_token": "sk_live_..."
  }
}
```

---

## 6. MCP Interface

### 6.1 Available Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `register_api` | Ingest OpenAPI spec | `api_id`, `spec_url` |
| `list_apis` | Show registered APIs | - |
| `search_endpoints` | Semantic endpoint search | `query`, `api_id`, `top_k` |
| `get_workflow` | Plan multi-step workflow | `query`, `api_id`, `max_steps` |
| `get_endpoint_schema` | Get detailed endpoint info | `api_id`, `endpoint_id` |
| `call_api` | Execute single endpoint | `api_id`, `endpoint_id`, params |
| `set_api_auth` | Configure authentication | `api_id`, `auth_type`, `credential` |
| `execute_workflow` | Run planned workflow | `workflow_id`, `api_id` |
| `delete_api` | Remove API and all data | `api_id` |

### 6.2 Tool Flow Example

```
User: "What's the weather in Paris?"

Claude: [calls get_workflow]
        → Returns workflow_id="abc12345" with 2 steps

Claude: [calls set_api_auth]
        → Configures API key

Claude: [calls execute_workflow]
        → Runs both steps, returns weather data

Claude: "The current weather in Paris is 18°C with clear skies."
```

### 6.3 Workflow Caching

The `get_workflow` tool caches planned workflows:
- Returns `workflow_id` (UUID first 8 chars)
- Stores `RerankedWorkflow` in memory
- `execute_workflow` retrieves by ID

This enables: **Plan → Inspect → Execute** interaction pattern.

---

## 7. Workflow Execution

### 7.1 Parameter Resolution

**Source Types:**

| Source | Resolution |
|--------|------------|
| `"user_query"` | Use pre-extracted value directly |
| `"literal"` | Use the literal value |
| `"step_N.field"` | Extract from step N's output via JSONPath |

**JSONPath Evaluation:**
- Handles: `.field`, `[N]`, `[N].field`
- Example: `$[0].lat` extracts first array element's `lat` field

### 7.2 Parameter Categorization

Based on endpoint definition, resolved parameters are categorized:

| Category | Location | Example |
|----------|----------|---------|
| Path | URL path | `/users/{id}` → `/users/123` |
| Query | URL params | `?lat=35.67&lon=139.65` |
| Body | Request body | `{"name": "Max", "status": "available"}` |

### 7.3 Error Handling

- Step failures stop execution immediately
- Error messages extracted from response body
- Common fields checked: `error`, `message`, `detail`, `error_description`

---

## 8. Configuration

### 8.1 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Required for embeddings & reranking | - |
| `JITAPI_STORAGE_DIR` | Data directory | `~/.jitapi` |
| `JITAPI_LOG_LEVEL` | Log level | `INFO` |
| `JITAPI_LOG_FILE` | Log file path | stderr |

### 8.2 MCP Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "jitapi": {
      "command": "uvx",
      "args": ["jitapi"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### 8.3 Dependencies

**Core:**
- `mcp>=1.0.0` - Model Context Protocol
- `openai>=1.0.0` - Embeddings & reranking
- `chromadb>=0.4.0` - Vector database
- `networkx>=3.0` - Dependency graphs
- `httpx>=0.25.0` - Async HTTP client
- `pydantic>=2.0.0` - Input validation

**Parsing:**
- `pyyaml` - YAML spec support

---

## 9. Example Walkthrough

### Scenario: "Get weather in Tokyo"

**Step 1: API Registration**
```
Tool: register_api(api_id="openweather", spec_url="https://...")

Processing:
1. Parser fetches and parses OpenAPI spec
2. Extracts 5 endpoints including:
   - GET /geo/1.0/direct (geocoding)
   - GET /data/2.5/weather (current weather)
3. GraphBuilder detects dependency:
   - weather endpoint needs lat/lon
   - geocoding endpoint provides lat/lon
4. Embedder generates vectors for all endpoints
5. All data stored in ChromaDB, JSON files
```

**Step 2: Authentication Setup**
```
Tool: set_api_auth(
  api_id="openweather",
  auth_type="api_key_query",
  credential="abc123...",
  param_name="appid"
)

Processing:
1. AuthHandler stores credential
2. Will inject as ?appid=abc123... on all calls
```

**Step 3: Workflow Planning**
```
Tool: get_workflow(
  query="Get weather in Tokyo",
  api_id="openweather",
  max_steps=5
)

Processing:
1. VectorSearcher embeds "Get weather in Tokyo"
2. Finds: GET /data/2.5/weather (score: 0.89)
3. GraphExpander adds: GET /geo/1.0/direct (dependency)
4. LLMReranker (GPT-4o-mini):
   - Analyzes endpoints
   - Extracts "Tokyo" from query
   - Orders steps: geocoding → weather
   - Maps data flow: lat/lon from step 1 to step 2

Returns:
{
  "workflow_id": "abc12345",
  "reasoning": "Need to geocode Tokyo first...",
  "steps": [
    {
      "step": 1,
      "endpoint_id": "GET /geo/1.0/direct",
      "purpose": "Get coordinates for Tokyo",
      "parameters": {
        "q": {"value": "Tokyo", "source": "user_query"}
      },
      "output_mapping": {
        "lat": "$[0].lat",
        "lon": "$[0].lon"
      }
    },
    {
      "step": 2,
      "endpoint_id": "GET /data/2.5/weather",
      "purpose": "Get weather at coordinates",
      "parameters": {
        "lat": {"value": null, "source": "step_1.lat"},
        "lon": {"value": null, "source": "step_1.lon"}
      }
    }
  ]
}
```

**Step 4: Workflow Execution**
```
Tool: execute_workflow(workflow_id="abc12345", api_id="openweather")

Processing:

Step 1 Execution:
- Resolve params: q="Tokyo" (from user_query)
- Build URL: /geo/1.0/direct?q=Tokyo&appid=abc123
- HTTP GET → Response: [{"lat": 35.6762, "lon": 139.6503, ...}]
- Extract outputs: lat=35.6762, lon=139.6503

Step 2 Execution:
- Resolve params: lat=35.6762 (from step_1.lat), lon=139.6503 (from step_1.lon)
- Build URL: /data/2.5/weather?lat=35.6762&lon=139.6503&appid=abc123
- HTTP GET → Response: {"weather": [...], "main": {"temp": 22}}

Returns:
{
  "success": true,
  "steps": [...],
  "final_result": {"weather": [...], "main": {"temp": 22}}
}
```

---

## 10. Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          COMPONENT INTERACTIONS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐                                                          │
│   │ToolRegistry  │ ◄─── MCP Tool calls from Claude                          │
│   └──────┬───────┘                                                          │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│   │  APIIndexer  │────►│  SpecStore   │────►│  GraphStore  │               │
│   │  (ingestion) │     │  (storage)   │     │  (storage)   │               │
│   └──────┬───────┘     └──────────────┘     └──────┬───────┘               │
│          │                                          │                        │
│          ▼                                          │                        │
│   ┌──────────────┐                                 │                        │
│   │EndpointEmbed.│                                 │                        │
│   └──────┬───────┘                                 │                        │
│          │                                          │                        │
│          ▼                                          │                        │
│   ┌──────────────┐                                 │                        │
│   │ VectorStore  │◄────────────────────────────────┘                        │
│   │  (ChromaDB)  │                                                          │
│   └──────┬───────┘                                                          │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│   │VectorSearcher│────►│GraphExpander │────►│ LLMReranker  │               │
│   │  (retrieval) │     │  (retrieval) │     │  (retrieval) │               │
│   └──────────────┘     └──────────────┘     └──────┬───────┘               │
│                                                     │                        │
│                                                     ▼                        │
│                                              ┌──────────────┐               │
│                                              │WorkflowExec. │               │
│                                              │  (execution) │               │
│                                              └──────┬───────┘               │
│                                                     │                        │
│                                                     ▼                        │
│   ┌──────────────┐                          ┌──────────────┐               │
│   │ AuthHandler  │─────────────────────────►│ HTTPExecutor │               │
│   │  (execution) │                          │  (execution) │               │
│   └──────────────┘                          └──────────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

JitAPI transforms natural language into executable API workflows through:

1. **Semantic Understanding** - Vector embeddings find relevant endpoints
2. **Dependency Awareness** - Graph analysis ensures proper ordering
3. **Intelligent Planning** - LLM extracts parameters and maps data flow
4. **Generic Execution** - JSONPath-based parameter resolution

This architecture enables any OpenAPI-compliant API to be orchestrated without API-specific code.
