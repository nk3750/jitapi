#!/usr/bin/env python3
"""
Samvaad CLI - Register APIs and query them from the terminal.

Usage:
  # Register a new API
  python samvaad_cli.py register <api_id> <spec_path_or_url> [--key <api_key>] [--key-param <param_name>] [--key-header <header_name>]

  # Query a registered API
  python samvaad_cli.py query <api_id> "<your question>"

  # Query and execute (actually call the API)
  python samvaad_cli.py run <api_id> "<your question>"

  # List registered APIs
  python samvaad_cli.py list

  # Set API key for an existing API
  python samvaad_cli.py auth <api_id> <api_key> [--param <name>] [--header <name>]

Examples:
  python samvaad_cli.py register openweathermap ./specs/openweathermap.yaml --key abc123 --key-param appid
  python samvaad_cli.py query openweathermap "get weather in Tokyo"
  python samvaad_cli.py run openweathermap "get weather in Tokyo"
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from samvaad.ingestion.indexer import APIIndexer
from samvaad.ingestion.embedder import EndpointEmbedder
from samvaad.retrieval.vector_search import VectorSearcher
from samvaad.retrieval.graph_expander import GraphExpander
from samvaad.retrieval.reranker import LLMReranker
from samvaad.stores.spec_store import SpecStore
from samvaad.stores.vector_store import VectorStore
from samvaad.stores.graph_store import GraphStore
from samvaad.execution.auth_handler import AuthHandler
from samvaad.execution.http_executor import HTTPExecutor


STORAGE = Path.home() / ".samvaad"


def cmd_register(args):
    """Register a new API."""
    api_id = args.api_id
    spec = args.spec

    print(f"Registering API '{api_id}'...")

    indexer = APIIndexer(STORAGE)

    # Check if URL or file
    if spec.startswith("http://") or spec.startswith("https://"):
        result = asyncio.run(indexer.index_from_url(api_id, spec))
    else:
        if not Path(spec).exists():
            print(f"Error: File not found: {spec}")
            sys.exit(1)
        result = indexer.index_from_file(api_id, spec)

    if result.success:
        print(f"✓ Registered: {result.title} v{result.version}")
        print(f"  Endpoints: {result.endpoint_count}")
        print(f"  Dependencies: {result.dependency_count}")

        # Set up auth if provided
        if args.key:
            auth_handler = AuthHandler(STORAGE)
            if args.key_param:
                auth_handler.set_api_key_query_param(api_id, args.key, args.key_param)
                print(f"  Auth: query param '{args.key_param}'")
            elif args.key_header:
                auth_handler.set_api_key(api_id, args.key, args.key_header)
                print(f"  Auth: header '{args.key_header}'")
            else:
                # Default to header
                auth_handler.set_api_key(api_id, args.key, "X-API-Key")
                print(f"  Auth: header 'X-API-Key'")
    else:
        print(f"✗ Failed: {result.error_message}")
        sys.exit(1)


def cmd_auth(args):
    """Set authentication for an API."""
    auth_handler = AuthHandler(STORAGE)

    if args.param:
        auth_handler.set_api_key_query_param(args.api_id, args.key, args.param)
        print(f"✓ Auth set for '{args.api_id}': query param '{args.param}'")
    elif args.header:
        auth_handler.set_api_key(args.api_id, args.key, args.header)
        print(f"✓ Auth set for '{args.api_id}': header '{args.header}'")
    else:
        auth_handler.set_api_key(args.api_id, args.key, "X-API-Key")
        print(f"✓ Auth set for '{args.api_id}': header 'X-API-Key'")


def cmd_list(args):
    """List registered APIs."""
    spec_store = SpecStore(STORAGE)
    auth_handler = AuthHandler(STORAGE)
    apis = spec_store.list_apis()

    if not apis:
        print("No APIs registered yet.")
        print("Use: python samvaad_cli.py register <api_id> <spec>")
        return

    print(f"Registered APIs ({len(apis)}):\n")
    for api in apis:
        has_auth = "✓" if auth_handler.has_auth(api.api_id) else "✗"
        print(f"  {api.api_id}")
        print(f"    Title: {api.title} v{api.version}")
        print(f"    Endpoints: {api.endpoint_count}")
        print(f"    Auth configured: {has_auth}")
        print()


def cmd_query(args):
    """Query an API (search + workflow, no execution)."""
    api_id = args.api_id
    query = args.query

    spec_store = SpecStore(STORAGE)
    if not spec_store.api_exists(api_id):
        print(f"Error: API '{api_id}' not found. Register it first.")
        sys.exit(1)

    vector_store = VectorStore(STORAGE)
    graph_store = GraphStore(STORAGE)
    embedder = EndpointEmbedder()

    print(f"Query: {query}")
    print("=" * 60)

    # Search
    searcher = VectorSearcher(vector_store, embedder)
    results = searcher.search(query, api_id=api_id, top_k=5)

    if not results:
        print("No matching endpoints found.")
        return

    print(f"\nMatching endpoints:")
    for r in results:
        print(f"  [{r.score:.2f}] {r.endpoint_id} - {r.summary}")

    # Expand
    expander = GraphExpander(graph_store, spec_store)
    expansion = expander.expand(results, api_id, max_depth=2)

    # Rerank
    reranker = LLMReranker()
    workflow = reranker.rerank(query, expansion.endpoints, max_steps=5)

    print(f"\nWorkflow:")
    print(f"  {workflow.reasoning}\n")

    for step in workflow.steps:
        print(f"  Step {step.step_number}: {step.endpoint_id}")
        print(f"           {step.purpose}")
        if step.requires:
            print(f"           Requires: {step.requires}")
        if step.provides:
            print(f"           Provides: {step.provides}")
        print()

    # Show schemas
    print("Endpoint details:")
    for step in workflow.steps:
        endpoint = spec_store.get_endpoint(api_id, step.endpoint_id)
        if endpoint:
            print(f"\n  {endpoint.endpoint_id}")
            print(f"    Path: {endpoint.path}")
            if endpoint.parameters:
                params = [f"{p.name}{'*' if p.required else ''}" for p in endpoint.parameters]
                print(f"    Params: {', '.join(params)}")


def cmd_run(args):
    """Query and execute API calls."""
    api_id = args.api_id
    query = args.query

    spec_store = SpecStore(STORAGE)
    if not spec_store.api_exists(api_id):
        print(f"Error: API '{api_id}' not found. Register it first.")
        sys.exit(1)

    auth_handler = AuthHandler(STORAGE)
    if not auth_handler.has_auth(api_id):
        print(f"Warning: No auth configured for '{api_id}'. API calls may fail.")
        print(f"Use: python samvaad_cli.py auth {api_id} <your_key> --param <param_name>")
        print()

    vector_store = VectorStore(STORAGE)
    graph_store = GraphStore(STORAGE)
    embedder = EndpointEmbedder()
    http_executor = HTTPExecutor(auth_handler)

    print(f"Query: {query}")
    print("=" * 60)

    # Search → Expand → Rerank
    searcher = VectorSearcher(vector_store, embedder)
    results = searcher.search(query, api_id=api_id, top_k=5)

    if not results:
        print("No matching endpoints found.")
        return

    expander = GraphExpander(graph_store, spec_store)
    expansion = expander.expand(results, api_id, max_depth=2)

    reranker = LLMReranker()
    workflow = reranker.rerank(query, expansion.endpoints, max_steps=5)

    print(f"\nPlan: {workflow.reasoning}\n")
    print("Executing...")
    print("-" * 40)

    # Execute workflow
    context = {}  # Store data between steps

    async def execute():
        # Check if we need geocoding first (weather APIs need lat/lon)
        needs_coords = any(
            any(p.name.lower() in ["lat", "lon", "latitude", "longitude"]
                for p in spec_store.get_endpoint(api_id, step.endpoint_id).parameters)
            for step in workflow.steps
            if spec_store.get_endpoint(api_id, step.endpoint_id)
        )

        # If we need coordinates but don't have geocoding in the plan, add it first
        geo_endpoint = spec_store.get_endpoint(api_id, "GET /geo/1.0/direct")
        if needs_coords and geo_endpoint and "lat" not in context:
            print(f"\n>> GET /geo/1.0/direct (auto-added)")
            print(f"   Getting coordinates for location...")

            search_term = extract_search_term(query)
            result = await http_executor.call_endpoint(
                endpoint=geo_endpoint,
                api_id=api_id,
                query_params={"q": search_term, "limit": "1"},
            )

            if result.success and isinstance(result.body, list) and result.body:
                loc = result.body[0]
                context["lat"] = loc.get("lat")
                context["lon"] = loc.get("lon")
                print(f"   ✓ Found: {loc.get('name')}, {loc.get('country')} ({context['lat']}, {context['lon']})")
            else:
                print(f"   ✗ Could not geocode '{search_term}'")
                return

        for step in workflow.steps:
            endpoint = spec_store.get_endpoint(api_id, step.endpoint_id)
            if not endpoint:
                continue

            # Skip geocoding if we already did it
            if "geo" in endpoint.path and "lat" in context:
                continue

            print(f"\n>> {step.endpoint_id}")
            print(f"   {step.purpose}")

            # Build params from context and query
            query_params = {}
            path_params = {}

            for param in endpoint.parameters:
                param_lower = param.name.lower()

                # Check if we have this in context
                if param.name in context:
                    query_params[param.name] = context[param.name]
                elif param_lower in context:
                    query_params[param.name] = context[param_lower]

                # Special handling for common patterns
                elif param_lower == "q" or param_lower == "query":
                    # Extract search term from user query
                    query_params[param.name] = extract_search_term(query)
                elif param_lower == "limit":
                    query_params[param.name] = "5"
                elif param_lower == "units":
                    query_params[param.name] = "metric"

            result = await http_executor.call_endpoint(
                endpoint=endpoint,
                api_id=api_id,
                query_params=query_params,
                path_params=path_params,
            )

            if result.success:
                print(f"   ✓ {result.status_code} OK")

                # Extract useful data to context
                if isinstance(result.body, list) and result.body:
                    item = result.body[0]
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if isinstance(value, (str, int, float)):
                                context[key] = value
                        # Pretty print first result
                        print(f"   Result: {item}")
                elif isinstance(result.body, dict):
                    # Store relevant fields
                    for key in ["id", "lat", "lon", "name", "temp", "description"]:
                        if key in result.body:
                            context[key] = result.body[key]

                    # Pretty print key info
                    if "main" in result.body:  # Weather response
                        main = result.body["main"]
                        weather = result.body.get("weather", [{}])[0]
                        print(f"   Weather: {weather.get('description', 'N/A')}")
                        print(f"   Temp: {main.get('temp')}°C (feels like {main.get('feels_like')}°C)")
                        print(f"   Humidity: {main.get('humidity')}%")
                    elif "name" in result.body:
                        print(f"   Found: {result.body.get('name')}")
            else:
                print(f"   ✗ Error: {result.error_message}")

    asyncio.run(execute())
    print("\n" + "=" * 60)
    print("Done!")


def extract_search_term(query: str) -> str:
    """Extract a likely search term from a natural language query."""
    # Simple heuristic: look for "in <place>" or use last noun-like words
    import re

    # Try to find "in <location>" pattern (case-insensitive)
    match = re.search(r'\b(?:in|for|of)\s+([a-zA-Z][a-zA-Z\s]+?)(?:\?|$|,|\.|!)', query, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try simpler pattern: "in <words at end>"
    match = re.search(r'\b(?:in|for|of)\s+(.+?)(?:\?|$)', query, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fall back to last capitalized words
    words = query.split()
    for word in reversed(words):
        if word and word[0].isupper():
            return word

    # Last resort: return the whole query cleaned up
    return query.rstrip('?').strip()


def main():
    parser = argparse.ArgumentParser(
        description="Samvaad CLI - Query APIs with natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # register
    p_reg = subparsers.add_parser("register", help="Register a new API")
    p_reg.add_argument("api_id", help="Unique ID for this API")
    p_reg.add_argument("spec", help="Path or URL to OpenAPI spec")
    p_reg.add_argument("--key", help="API key")
    p_reg.add_argument("--key-param", help="Query parameter name for API key")
    p_reg.add_argument("--key-header", help="Header name for API key")

    # auth
    p_auth = subparsers.add_parser("auth", help="Set API authentication")
    p_auth.add_argument("api_id", help="API ID")
    p_auth.add_argument("key", help="API key")
    p_auth.add_argument("--param", help="Query parameter name")
    p_auth.add_argument("--header", help="Header name")

    # list
    subparsers.add_parser("list", help="List registered APIs")

    # query
    p_query = subparsers.add_parser("query", help="Query an API (no execution)")
    p_query.add_argument("api_id", help="API ID")
    p_query.add_argument("query", help="Your question")

    # run
    p_run = subparsers.add_parser("run", help="Query and execute API calls")
    p_run.add_argument("api_id", help="API ID")
    p_run.add_argument("query", help="Your question")

    args = parser.parse_args()

    if args.command == "register":
        cmd_register(args)
    elif args.command == "auth":
        cmd_auth(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
