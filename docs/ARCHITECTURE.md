# JitAPI: Architecture

> **JitAPI** — a just-in-time API orchestration layer for LLMs. Register any OpenAPI spec; the host model (Claude) discovers the right endpoints and calls them.

---

## Design in one paragraph

JitAPI is an MCP server. You register an OpenAPI spec once; JitAPI parses it, builds a lightweight endpoint **dependency graph**, and creates **local semantic embeddings** for every endpoint. At query time it does the retrieval — semantic search plus dependency-graph expansion — and returns the relevant endpoint schemas to the host. **The host LLM (Claude) does the planning and executes the calls** through a generic `call_api` tool. JitAPI itself runs no LLM and needs no API key by default.

This is deliberately a *thin retrieval + execution* layer. An earlier design had JitAPI plan and run multi-step workflows itself with a GPT-4o-mini reranker; that responsibility now lives in the host model, which plans better from schemas than a heuristic planner did. JitAPI's job is to make sure the model sees the *right* endpoints without dumping a 1,000-endpoint spec into context.

---

## Component map

```
Claude (host) ──tools──▶  JitAPI MCP server
                          │
            ┌─────────────┼──────────────┬───────────────┐
            ▼             ▼              ▼               ▼
        Ingestion     Retrieval       Stores         Execution
        ─────────     ─────────       ──────         ─────────
        parser        vector_search   spec_store     http_executor
        indexer       graph_expander  vector_store   auth_handler
        embedder                      graph_store    schema_formatter
        graph_builder
```

| Layer | Module(s) | Responsibility |
|-------|-----------|----------------|
| **MCP** | `mcp/server.py` (`JitAPIServer`), `mcp/tools.py` (`ToolRegistry`), `mcp/models.py`, `mcp/resources.py` | Expose 8 tools + resources/prompts over stdio; validate inputs with Pydantic |
| **Ingestion** | `ingestion/parser.py`, `indexer.py`, `embedder.py`, `graph_builder.py` | Parse OpenAPI 3.x / Swagger 2.0; build dependency graph; embed endpoints |
| **Retrieval** | `retrieval/vector_search.py`, `graph_expander.py` | Semantic search; expand with prerequisite endpoints from the graph |
| **Stores** | `stores/spec_store.py`, `vector_store.py`, `graph_store.py` | Persist specs/endpoints (JSON), embeddings (numpy), graphs (NetworkX) |
| **Execution** | `execution/http_executor.py`, `auth_handler.py`, `schema_formatter.py` | Make authenticated HTTP calls; inject credentials; format schemas for the model |

---

## Embeddings

Provider is auto-detected from environment variables; **the default is local** (`fastembed`, `BAAI/bge-small-en-v1.5`, 384-dim) so JitAPI works with **zero configuration and no API key**.

| Priority | Provider | Trigger | Dim |
|----------|----------|---------|-----|
| 1 | Voyage AI | `VOYAGE_API_KEY` | 1024 |
| 2 | OpenAI | `OPENAI_API_KEY` | 1536 |
| 3 | Cohere | `COHERE_API_KEY` | 1024 |
| 4 (default) | Local fastembed | none | 384 |

Force a provider with `JITAPI_EMBEDDING_PROVIDER=voyage|openai|cohere|local`. Embeddings are cached by content hash. The vector store is a small **numpy cosine-similarity** index (not a separate vector DB) — right-sized for the tens-to-hundreds of endpoints a single API exposes.

> Note: a store is built with one provider's dimension. Switching embedding providers requires re-registering affected APIs.

---

## Pipelines

### Registration (`register_api`)

```
spec URL ──▶ parser ──▶ spec_store (specs + parsed endpoints)
                 │
                 ├──▶ graph_builder ──▶ graph_store (dependency graph)
                 │
                 └──▶ embedder ──▶ vector_store (numpy embeddings)
```

The dependency graph records, per edge, the parameter that links a consumer endpoint to a provider endpoint (e.g. `POST /orders` needs `product_id`, which `GET /products` returns), with a heuristic confidence.

### Retrieval (`search_endpoints`, `get_workflow`)

```
query ──▶ vector_search (semantic top-k)
              │
              └──▶ graph_expander (adds prerequisite endpoints the search missed)
                        │
                        └──▶ schema_formatter ──▶ endpoint schemas returned to Claude
```

`search_endpoints` returns ranked endpoints. `get_workflow` additionally expands with dependencies and returns full call schemas, ordered prerequisites-first, so Claude can plan and then drive `call_api` for each step.

### Execution (`call_api`)

The host model calls `call_api` per endpoint. `http_executor` substitutes path params, injects auth via `auth_handler`, makes the request with `httpx`, and returns status + body + headers to the model. Outbound requests are validated against an SSRF guard (private/loopback/link-local/metadata hosts are blocked unless `JITAPI_ALLOW_PRIVATE_HOSTS=1`).

---

## MCP tools

| Tool | Purpose |
|------|---------|
| `register_api` | Ingest an OpenAPI spec from a URL |
| `list_apis` | List registered APIs |
| `search_endpoints` | Semantic endpoint search |
| `get_workflow` | Relevant endpoints + dependency expansion + schemas |
| `get_endpoint_schema` | Full schema for one endpoint |
| `call_api` | Execute a single endpoint call |
| `set_api_auth` | Configure auth (`bearer`, `api_key`, `api_key_query`; `env_var` for at-rest-safe secrets) |
| `delete_api` | Remove an API and all its data |

There is **no** server-side workflow-execution tool: planning and step sequencing are the host model's job.

---

## Storage layout (`~/.jitapi`, override with `JITAPI_STORAGE_DIR`)

```
~/.jitapi/
├── specs/{api_id}.json       # raw spec + parsed endpoints
├── graphs/{api_id}.json      # dependency graph (NetworkX node_link)
├── vector_store.json         # endpoint embeddings (numpy, serialized)
├── apis.json                 # API catalog/metadata
└── auth.json                 # credentials (0600); only env-var *names* when env_var auth is used
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `JITAPI_STORAGE_DIR` | `~/.jitapi` | Data directory |
| `JITAPI_LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `JITAPI_LOG_FILE` | stderr | Log file path |
| `JITAPI_EMBEDDING_PROVIDER` | auto | Force `voyage`/`openai`/`cohere`/`local` |
| `VOYAGE_API_KEY` / `OPENAI_API_KEY` / `COHERE_API_KEY` | — | Optional cloud embedding providers |
| `JITAPI_ALLOW_PRIVATE_HOSTS` | unset | Set to `1` to allow calls to private/loopback hosts (local dev) |
| `JITAPI_MAX_RESPONSE_BYTES` | `10485760` | Max response/spec body size in bytes before truncation |

No `OPENAI_API_KEY` is required — it is only one optional embedding provider among several.
