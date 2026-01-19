#!/usr/bin/env python3
"""Example: Query a registered API with natural language.

This script demonstrates how to search for endpoints and get
workflows using natural language queries.

Usage:
    python query_api.py <api_id> "<query>"

Example:
    python query_api.py petstore "create a new pet"
    python query_api.py weather "get the weather in Tokyo"
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from samvaad.ingestion.embedder import EndpointEmbedder
from samvaad.ingestion.indexer import APIIndexer
from samvaad.retrieval.graph_expander import GraphExpander
from samvaad.retrieval.reranker import LLMReranker
from samvaad.retrieval.vector_search import VectorSearcher
from samvaad.stores.graph_store import GraphStore
from samvaad.stores.spec_store import SpecStore
from samvaad.stores.vector_store import VectorStore


async def main():
    if len(sys.argv) < 3:
        print("Usage: python query_api.py <api_id> \"<query>\"")
        print("")
        print("Examples:")
        print("  python query_api.py petstore \"create a new pet\"")
        print("  python query_api.py weather \"get weather in Tokyo\"")
        sys.exit(1)

    api_id = sys.argv[1]
    query = sys.argv[2]

    # Get configuration from environment
    storage_dir = os.environ.get("SAMVAAD_STORAGE_DIR", str(Path.home() / ".samvaad"))
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    storage_path = Path(storage_dir)

    # Initialize components
    print(f"Loading API '{api_id}'...")

    spec_store = SpecStore(storage_path)
    vector_store = VectorStore(storage_path)
    graph_store = GraphStore(storage_path)
    embedder = EndpointEmbedder(api_key=openai_api_key)

    # Check if API exists
    if not spec_store.api_exists(api_id):
        print(f"Error: API '{api_id}' not found")
        print("Register it first with: python register_spec.py <api_id> <spec_url>")
        sys.exit(1)

    # Get API info
    metadata = spec_store.get_metadata(api_id)
    print(f"API: {metadata.title} v{metadata.version}")
    print(f"Endpoints: {metadata.endpoint_count}")
    print("")

    # Initialize searcher and expander
    searcher = VectorSearcher(vector_store, embedder)
    expander = GraphExpander(graph_store, spec_store)

    # Step 1: Semantic search
    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)
    print("")
    print("Step 1: Semantic Search")
    print("-" * 40)

    search_results = searcher.search(query, api_id=api_id, top_k=5)

    if not search_results:
        print("No matching endpoints found.")
        sys.exit(0)

    print(f"Found {len(search_results)} matching endpoints:")
    for result in search_results:
        print(f"  [{result.score:.2f}] {result.endpoint_id}")
        if result.summary:
            print(f"          {result.summary}")

    # Step 2: Graph expansion
    print("")
    print("Step 2: Dependency Expansion")
    print("-" * 40)

    expansion = expander.expand(search_results, api_id, max_depth=2, max_total=8)

    print(f"Original: {expansion.original_count} endpoints")
    print(f"Added:    {expansion.expanded_count} dependencies")

    if expansion.expanded_count > 0:
        print("")
        print("Expanded endpoint set:")
        for ep in expansion.endpoints:
            marker = "[DEP]" if ep.is_dependency else "[SRC]"
            print(f"  {marker} {ep.endpoint_id}")
            if ep.depends_on:
                print(f"        depends on: {', '.join(ep.depends_on)}")

    # Step 3: LLM Reranking
    print("")
    print("Step 3: LLM Reranking")
    print("-" * 40)

    reranker = LLMReranker(api_key=openai_api_key)
    workflow = reranker.rerank(query, expansion.endpoints, max_steps=5)

    print(f"Reasoning: {workflow.reasoning}")
    print("")
    print("Recommended Workflow:")
    for step in workflow.steps:
        print(f"  Step {step.step_number}: {step.endpoint_id}")
        print(f"           Purpose: {step.purpose}")
        if step.requires:
            print(f"           Requires: {', '.join(step.requires)}")
        if step.provides:
            print(f"           Provides: {', '.join(step.provides)}")
        print("")

    # Step 4: Show endpoint schemas
    print("")
    print("Step 4: Endpoint Schemas")
    print("-" * 40)

    # Show schema for top result
    top_endpoint = search_results[0]
    endpoint = spec_store.get_endpoint(api_id, top_endpoint.endpoint_id)

    if endpoint:
        print(f"Schema for {endpoint.endpoint_id}:")
        print(f"  Path: {endpoint.path}")
        print(f"  Method: {endpoint.method}")
        print(f"  Summary: {endpoint.summary}")

        if endpoint.parameters:
            print("  Parameters:")
            for param in endpoint.parameters:
                req = "*" if param.required else ""
                print(f"    - {param.name}{req} ({param.location}): {param.schema_type}")

        if endpoint.request_body:
            print(f"  Request Body: {endpoint.request_body.content_type}")
            if endpoint.request_body.schema.get("properties"):
                print("    Properties:")
                for name, prop in endpoint.request_body.schema["properties"].items():
                    print(f"      - {name}: {prop.get('type', 'any')}")

    print("")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
