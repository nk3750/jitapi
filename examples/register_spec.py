#!/usr/bin/env python3
"""Example: Register an OpenAPI specification with Samvaad.

This script demonstrates how to programmatically register an API
with Samvaad, which parses the spec, builds a dependency graph,
and creates searchable embeddings.

Usage:
    python register_spec.py <api_id> <spec_url_or_path>

Example:
    python register_spec.py petstore https://petstore.swagger.io/v2/swagger.json
    python register_spec.py weather ./tests/fixtures/weather_api.yaml
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from samvaad.ingestion.indexer import APIIndexer


async def main():
    if len(sys.argv) < 3:
        print("Usage: python register_spec.py <api_id> <spec_url_or_path>")
        print("")
        print("Examples:")
        print("  python register_spec.py petstore https://petstore.swagger.io/v2/swagger.json")
        print("  python register_spec.py weather ./tests/fixtures/weather_api.yaml")
        sys.exit(1)

    api_id = sys.argv[1]
    spec_source = sys.argv[2]

    # Get storage directory from env or use default
    storage_dir = os.environ.get("SAMVAAD_STORAGE_DIR", str(Path.home() / ".samvaad"))
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Set it with: export OPENAI_API_KEY=your-key")
        sys.exit(1)

    print(f"Initializing Samvaad indexer...")
    print(f"Storage directory: {storage_dir}")

    indexer = APIIndexer(storage_dir, openai_api_key=openai_api_key)

    # Determine if source is URL or file
    if spec_source.startswith("http://") or spec_source.startswith("https://"):
        print(f"Registering API '{api_id}' from URL: {spec_source}")
        result = await indexer.index_from_url(api_id, spec_source)
    else:
        # Local file
        spec_path = Path(spec_source)
        if not spec_path.exists():
            print(f"Error: File not found: {spec_path}")
            sys.exit(1)

        print(f"Registering API '{api_id}' from file: {spec_path}")
        result = indexer.index_from_file(api_id, spec_path)

    if result.success:
        print("")
        print("=" * 50)
        print("API registered successfully!")
        print("=" * 50)
        print(f"  API ID:           {result.api_id}")
        print(f"  Title:            {result.title}")
        print(f"  Version:          {result.version}")
        print(f"  Endpoints:        {result.endpoint_count}")
        print(f"  Dependencies:     {result.dependency_count}")
        print("")
        print("You can now use this API with the Samvaad MCP server.")
    else:
        print("")
        print("Error registering API:")
        print(f"  {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
