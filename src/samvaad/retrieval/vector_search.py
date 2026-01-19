"""Vector search for semantically similar endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ingestion.embedder import EndpointEmbedder
from ..stores.vector_store import SearchResult, VectorStore


@dataclass
class VectorSearchResult:
    """Enhanced search result with additional context."""

    endpoint_id: str
    api_id: str
    score: float
    path: str
    method: str
    summary: str
    tags: list[str]
    metadata: dict[str, Any]


class VectorSearcher:
    """Performs semantic search over API endpoints.

    Uses embeddings to find endpoints that semantically match a query.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EndpointEmbedder,
    ):
        """Initialize the vector searcher.

        Args:
            vector_store: The vector store containing endpoint embeddings.
            embedder: The embedder for generating query embeddings.
        """
        self.vector_store = vector_store
        self.embedder = embedder

    def search(
        self,
        query: str,
        api_id: str | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search for endpoints matching a natural language query.

        Args:
            query: The search query (natural language).
            api_id: Optional filter to search within a specific API.
            top_k: Maximum number of results to return.
            min_score: Minimum similarity score threshold.

        Returns:
            List of VectorSearchResult objects, sorted by relevance.
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            api_id=api_id,
            top_k=top_k,
            filter_deprecated=True,
        )

        # Convert and filter results
        search_results = []
        for result in results:
            if result.score >= min_score:
                search_results.append(self._enhance_result(result))

        return search_results

    def search_with_expansion(
        self,
        query: str,
        api_id: str | None = None,
        top_k: int = 10,
        expand_query: bool = True,
    ) -> list[VectorSearchResult]:
        """Search with query expansion for better recall.

        Generates multiple query variants to capture different phrasings.

        Args:
            query: The original search query.
            api_id: Optional filter to search within a specific API.
            top_k: Maximum number of results to return.
            expand_query: Whether to expand the query.

        Returns:
            List of VectorSearchResult objects.
        """
        queries = [query]

        if expand_query:
            # Add query variants
            queries.extend(self._expand_query(query))

        # Search with all queries and combine results
        all_results: dict[str, VectorSearchResult] = {}

        for q in queries:
            results = self.search(q, api_id=api_id, top_k=top_k)
            for result in results:
                key = f"{result.api_id}::{result.endpoint_id}"
                if key not in all_results:
                    all_results[key] = result
                else:
                    # Keep the highest score
                    if result.score > all_results[key].score:
                        all_results[key] = result

        # Sort by score and return top_k
        sorted_results = sorted(
            all_results.values(), key=lambda x: x.score, reverse=True
        )
        return sorted_results[:top_k]

    def search_by_method(
        self,
        query: str,
        method: str,
        api_id: str | None = None,
        top_k: int = 10,
    ) -> list[VectorSearchResult]:
        """Search with a method filter (GET, POST, etc).

        Args:
            query: The search query.
            method: HTTP method to filter by.
            api_id: Optional API filter.
            top_k: Maximum results.

        Returns:
            Filtered search results.
        """
        # First, do a broader search
        results = self.search(query, api_id=api_id, top_k=top_k * 2)

        # Filter by method
        filtered = [r for r in results if r.method.upper() == method.upper()]

        return filtered[:top_k]

    def search_by_tag(
        self,
        query: str,
        tag: str,
        api_id: str | None = None,
        top_k: int = 10,
    ) -> list[VectorSearchResult]:
        """Search with a tag filter.

        Args:
            query: The search query.
            tag: Tag to filter by.
            api_id: Optional API filter.
            top_k: Maximum results.

        Returns:
            Filtered search results.
        """
        results = self.search(query, api_id=api_id, top_k=top_k * 2)

        # Filter by tag
        tag_lower = tag.lower()
        filtered = [
            r for r in results if any(t.lower() == tag_lower for t in r.tags)
        ]

        return filtered[:top_k]

    def find_similar_endpoints(
        self,
        endpoint_id: str,
        api_id: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """Find endpoints similar to a given endpoint.

        Args:
            endpoint_id: The reference endpoint.
            api_id: The API containing the endpoint.
            top_k: Maximum results.

        Returns:
            List of similar endpoints.
        """
        # Get the endpoint's embedding
        endpoint_data = self.vector_store.get_endpoint(api_id, endpoint_id)
        if not endpoint_data or not endpoint_data.get("embedding"):
            return []

        # Search for similar
        results = self.vector_store.search(
            query_embedding=endpoint_data["embedding"],
            api_id=api_id,
            top_k=top_k + 1,  # +1 to exclude self
        )

        # Filter out the query endpoint itself
        similar = []
        for result in results:
            if result.endpoint_id != endpoint_id:
                similar.append(self._enhance_result(result))

        return similar[:top_k]

    def _enhance_result(self, result: SearchResult) -> VectorSearchResult:
        """Enhance a raw search result with additional context."""
        metadata = result.metadata

        # Parse tags from comma-separated string if needed
        tags = metadata.get("tags", "")
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        return VectorSearchResult(
            endpoint_id=result.endpoint_id,
            api_id=result.api_id,
            score=result.score,
            path=metadata.get("path", ""),
            method=metadata.get("method", ""),
            summary=metadata.get("summary", ""),
            tags=tags,
            metadata=metadata,
        )

    def _expand_query(self, query: str) -> list[str]:
        """Generate query variants for better recall.

        Simple heuristic-based expansion without LLM calls.
        """
        variants = []
        query_lower = query.lower()

        # Add action-based variants
        action_mappings = {
            "create": ["add", "new", "post", "make"],
            "get": ["fetch", "retrieve", "read", "list"],
            "update": ["modify", "edit", "change", "patch"],
            "delete": ["remove", "destroy", "drop"],
            "search": ["find", "query", "lookup", "filter"],
            "list": ["get all", "fetch all", "retrieve all"],
        }

        for action, synonyms in action_mappings.items():
            if action in query_lower:
                for synonym in synonyms[:2]:  # Limit variants
                    variant = query_lower.replace(action, synonym)
                    if variant != query_lower:
                        variants.append(variant)

        # Add noun variants
        noun_mappings = {
            "user": ["account", "member"],
            "product": ["item", "good"],
            "order": ["purchase", "transaction"],
            "customer": ["client", "buyer"],
        }

        for noun, synonyms in noun_mappings.items():
            if noun in query_lower:
                for synonym in synonyms[:1]:
                    variant = query_lower.replace(noun, synonym)
                    if variant != query_lower:
                        variants.append(variant)

        return variants[:3]  # Limit total variants
