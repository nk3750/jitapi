#!/usr/bin/env python3
"""Test OpenWeatherMap API with actual API calls."""

import os
import sys
import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from samvaad.ingestion.embedder import EndpointEmbedder
from samvaad.retrieval.vector_search import VectorSearcher
from samvaad.retrieval.graph_expander import GraphExpander
from samvaad.retrieval.reranker import LLMReranker
from samvaad.stores.spec_store import SpecStore
from samvaad.stores.vector_store import VectorStore
from samvaad.stores.graph_store import GraphStore
from samvaad.execution.auth_handler import AuthHandler
from samvaad.execution.http_executor import HTTPExecutor


async def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "San Francisco"

    storage = Path.home() / ".samvaad"
    api_id = "openweathermap"
    query = f"what is the weather in {city}"

    # Initialize components
    spec_store = SpecStore(storage)
    vector_store = VectorStore(storage)
    graph_store = GraphStore(storage)
    embedder = EndpointEmbedder()
    auth_handler = AuthHandler(storage)
    http_executor = HTTPExecutor(auth_handler)

    # Set up auth
    owm_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    if not owm_key:
        print("Error: OPENWEATHERMAP_API_KEY not set in .env")
        sys.exit(1)

    auth_handler.set_api_key_query_param(api_id, owm_key, "appid")

    print(f"Query: {query}")
    print("=" * 60)

    # Step 1: Search
    searcher = VectorSearcher(vector_store, embedder)
    results = searcher.search(query, api_id=api_id, top_k=5)

    # Step 2: Expand
    expander = GraphExpander(graph_store, spec_store)
    expansion = expander.expand(results, api_id, max_depth=2)

    # Step 3: Rerank
    reranker = LLMReranker()
    workflow = reranker.rerank(query, expansion.endpoints, max_steps=5)

    print(f"\nWorkflow: {workflow.reasoning}\n")

    # Step 4: Execute the workflow
    print("Executing API calls...")
    print("-" * 40)

    context = {}  # Store results between steps

    for step in workflow.steps:
        endpoint = spec_store.get_endpoint(api_id, step.endpoint_id)
        if not endpoint:
            continue

        print(f"\nStep {step.step_number}: {step.endpoint_id}")
        print(f"  Purpose: {step.purpose}")

        # Build parameters
        query_params = {}

        if "geo" in endpoint.path:
            # Geocoding endpoint
            query_params["q"] = city
            query_params["limit"] = "1"
        elif "weather" in endpoint.path or "forecast" in endpoint.path:
            # Weather endpoints need lat/lon from previous step
            if "lat" in context and "lon" in context:
                query_params["lat"] = context["lat"]
                query_params["lon"] = context["lon"]
                query_params["units"] = "metric"
            else:
                print("  Skipping - no coordinates available")
                continue

        # Make the call
        result = await http_executor.call_endpoint(
            endpoint=endpoint,
            api_id=api_id,
            query_params=query_params,
        )

        if result.success:
            print(f"  Status: {result.status_code} OK")

            # Extract and display relevant data
            if "geo" in endpoint.path and isinstance(result.body, list) and result.body:
                loc = result.body[0]
                context["lat"] = loc.get("lat")
                context["lon"] = loc.get("lon")
                print(f"  Found: {loc.get('name')}, {loc.get('country')}")
                print(f"  Coordinates: {context['lat']}, {context['lon']}")

            elif "weather" in endpoint.path and isinstance(result.body, dict):
                main = result.body.get("main", {})
                weather = result.body.get("weather", [{}])[0]
                print(f"  Weather: {weather.get('description', 'N/A')}")
                print(f"  Temperature: {main.get('temp')}°C")
                print(f"  Feels like: {main.get('feels_like')}°C")
                print(f"  Humidity: {main.get('humidity')}%")
        else:
            print(f"  Error: {result.error_message}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
