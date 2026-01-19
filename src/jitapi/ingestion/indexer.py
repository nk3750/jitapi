"""API Indexer - orchestrates the ingestion pipeline.

Coordinates parsing, graph building, embedding, and storage
for a complete API ingestion workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..stores.graph_store import GraphStore
from ..stores.spec_store import SpecStore
from ..stores.vector_store import VectorStore
from .embedder import EndpointEmbedder
from .graph_builder import DependencyGraphBuilder
from .parser import OpenAPIParser, ParsedSpec


@dataclass
class IndexingResult:
    """Result of indexing an API."""

    api_id: str
    title: str
    version: str
    endpoint_count: int
    dependency_count: int
    success: bool
    error_message: str | None = None


class APIIndexer:
    """Orchestrates the complete API ingestion pipeline.

    Steps:
    1. Parse OpenAPI spec
    2. Build dependency graph
    3. Generate embeddings
    4. Store all data
    """

    def __init__(
        self,
        storage_dir: str | Path,
        openai_api_key: str | None = None,
    ):
        """Initialize the API indexer.

        Args:
            storage_dir: Base directory for all storage.
            openai_api_key: OpenAI API key for embeddings.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.parser = OpenAPIParser()
        self.graph_builder = DependencyGraphBuilder()
        self.embedder = EndpointEmbedder(api_key=openai_api_key)

        # Initialize stores
        self.spec_store = SpecStore(self.storage_dir)
        self.graph_store = GraphStore(self.storage_dir)
        self.vector_store = VectorStore(self.storage_dir)

    async def index_from_url(
        self,
        api_id: str,
        spec_url: str,
    ) -> IndexingResult:
        """Index an API from a URL.

        Args:
            api_id: Unique identifier for this API.
            spec_url: URL to the OpenAPI specification.

        Returns:
            IndexingResult with status and statistics.
        """
        try:
            # Parse the spec
            parsed_spec = await self.parser.parse_from_url(spec_url)
            return self._index_parsed_spec(api_id, parsed_spec, spec_url)

        except Exception as e:
            return IndexingResult(
                api_id=api_id,
                title="",
                version="",
                endpoint_count=0,
                dependency_count=0,
                success=False,
                error_message=str(e),
            )

    def index_from_file(
        self,
        api_id: str,
        file_path: str | Path,
    ) -> IndexingResult:
        """Index an API from a local file.

        Args:
            api_id: Unique identifier for this API.
            file_path: Path to the OpenAPI specification file.

        Returns:
            IndexingResult with status and statistics.
        """
        try:
            # Parse the spec
            parsed_spec = self.parser.parse_from_file(str(file_path))
            return self._index_parsed_spec(api_id, parsed_spec, str(file_path))

        except Exception as e:
            return IndexingResult(
                api_id=api_id,
                title="",
                version="",
                endpoint_count=0,
                dependency_count=0,
                success=False,
                error_message=str(e),
            )

    def _index_parsed_spec(
        self,
        api_id: str,
        parsed_spec: ParsedSpec,
        source: str,
    ) -> IndexingResult:
        """Index a parsed specification.

        Args:
            api_id: The API identifier.
            parsed_spec: The parsed OpenAPI spec.
            source: Source URL or file path.

        Returns:
            IndexingResult with status.
        """
        # Store the spec
        self.spec_store.store_spec(api_id, parsed_spec, source)

        # Build dependency graph
        graph = self.graph_builder.build(parsed_spec)
        self.graph_store.store_graph(api_id, graph)
        dependency_count = graph.number_of_edges()

        # Generate and store embeddings
        embeddings = self.embedder.embed_endpoints(parsed_spec.endpoints, api_id)

        # Store in vector store
        endpoint_data = [
            {
                "endpoint_id": emb.endpoint_id,
                "api_id": emb.api_id,
                "embedding": emb.embedding,
                "metadata": emb.metadata,
                "document": emb.text,
            }
            for emb in embeddings
        ]
        self.vector_store.add_endpoints_batch(endpoint_data)

        return IndexingResult(
            api_id=api_id,
            title=parsed_spec.title,
            version=parsed_spec.version,
            endpoint_count=len(parsed_spec.endpoints),
            dependency_count=dependency_count,
            success=True,
        )

    def delete_api(self, api_id: str) -> bool:
        """Delete an API and all its indexed data.

        Args:
            api_id: The API identifier.

        Returns:
            True if deleted successfully.
        """
        # Delete from all stores
        spec_deleted = self.spec_store.delete_api(api_id)
        self.graph_store.delete_graph(api_id)
        self.vector_store.delete_api(api_id)

        return spec_deleted

    def list_apis(self) -> list[dict[str, Any]]:
        """List all indexed APIs.

        Returns:
            List of API metadata dicts.
        """
        apis = self.spec_store.list_apis()
        return [
            {
                "api_id": api.api_id,
                "title": api.title,
                "version": api.version,
                "description": api.description,
                "endpoint_count": api.endpoint_count,
                "source_url": api.source_url,
            }
            for api in apis
        ]

    def get_api_info(self, api_id: str) -> dict[str, Any] | None:
        """Get detailed information about an API.

        Args:
            api_id: The API identifier.

        Returns:
            Dict with API info, or None if not found.
        """
        metadata = self.spec_store.get_metadata(api_id)
        if not metadata:
            return None

        graph_stats = self.graph_store.get_graph_stats(api_id)

        return {
            "api_id": api_id,
            "title": metadata.title,
            "version": metadata.version,
            "description": metadata.description,
            "base_url": metadata.base_url,
            "endpoint_count": metadata.endpoint_count,
            "source_url": metadata.source_url,
            "graph_stats": graph_stats,
        }

    def api_exists(self, api_id: str) -> bool:
        """Check if an API is indexed.

        Args:
            api_id: The API identifier.

        Returns:
            True if the API exists.
        """
        return self.spec_store.api_exists(api_id)
