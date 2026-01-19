"""Embedding generator for API endpoints.

Creates vector embeddings from endpoint metadata for semantic search.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .parser import Endpoint


@dataclass
class EndpointEmbedding:
    """An endpoint with its embedding vector."""

    endpoint_id: str
    api_id: str
    embedding: list[float]
    text: str  # The text that was embedded
    metadata: dict[str, Any]


class EndpointEmbedder:
    """Generates embeddings for API endpoints using OpenAI.

    Uses text-embedding-3-small for cost-effective, high-quality embeddings.
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS = 1536

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        """Initialize the embedder.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: The embedding model to use.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._cache: dict[str, list[float]] = {}

    def embed_endpoint(self, endpoint: Endpoint, api_id: str) -> EndpointEmbedding:
        """Generate an embedding for a single endpoint.

        Args:
            endpoint: The endpoint to embed.
            api_id: The API this endpoint belongs to.

        Returns:
            EndpointEmbedding with the vector and metadata.
        """
        text = self._build_embedding_text(endpoint)
        embedding = self._get_embedding(text)

        return EndpointEmbedding(
            endpoint_id=endpoint.endpoint_id,
            api_id=api_id,
            embedding=embedding,
            text=text,
            metadata=self._build_metadata(endpoint, api_id),
        )

    def embed_endpoints(
        self, endpoints: list[Endpoint], api_id: str
    ) -> list[EndpointEmbedding]:
        """Generate embeddings for multiple endpoints in batch.

        Args:
            endpoints: List of endpoints to embed.
            api_id: The API these endpoints belong to.

        Returns:
            List of EndpointEmbedding objects.
        """
        # Prepare texts for batch embedding
        texts = [self._build_embedding_text(ep) for ep in endpoints]

        # Batch embed
        embeddings = self._get_embeddings_batch(texts)

        # Build results
        results = []
        for endpoint, text, embedding in zip(endpoints, texts, embeddings):
            results.append(
                EndpointEmbedding(
                    endpoint_id=endpoint.endpoint_id,
                    api_id=api_id,
                    embedding=embedding,
                    text=text,
                    metadata=self._build_metadata(endpoint, api_id),
                )
            )

        return results

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a search query.

        Args:
            query: The search query text.

        Returns:
            The embedding vector.
        """
        return self._get_embedding(query)

    def _build_embedding_text(self, endpoint: Endpoint) -> str:
        """Build the text representation of an endpoint for embedding.

        Combines multiple fields to create a rich text representation
        that captures the endpoint's purpose and functionality.
        """
        parts = []

        # Method and path (most important)
        parts.append(f"{endpoint.method} {endpoint.path}")

        # Summary and description
        if endpoint.summary:
            parts.append(endpoint.summary)
        if endpoint.description and endpoint.description != endpoint.summary:
            # Truncate long descriptions
            desc = endpoint.description[:500]
            parts.append(desc)

        # Operation ID (often descriptive)
        if endpoint.operation_id:
            # Convert camelCase/snake_case to words
            op_words = self._split_identifier(endpoint.operation_id)
            parts.append(op_words)

        # Tags
        if endpoint.tags:
            parts.append(f"Category: {', '.join(endpoint.tags)}")

        # Parameter names (useful for semantic matching)
        param_names = [p.name for p in endpoint.parameters]
        if param_names:
            parts.append(f"Parameters: {', '.join(param_names)}")

        # Required parameters specifically
        if endpoint.required_params:
            parts.append(f"Requires: {', '.join(endpoint.required_params)}")

        return " | ".join(filter(None, parts))

    def _build_metadata(self, endpoint: Endpoint, api_id: str) -> dict[str, Any]:
        """Build metadata dict for vector store storage."""
        return {
            "api_id": api_id,
            "endpoint_id": endpoint.endpoint_id,
            "path": endpoint.path,
            "method": endpoint.method,
            "summary": endpoint.summary or "",
            "tags": endpoint.tags,
            "operation_id": endpoint.operation_id or "",
            "deprecated": endpoint.deprecated,
            "param_count": len(endpoint.parameters),
            "has_request_body": endpoint.request_body is not None,
        }

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text, with caching."""
        cache_key = self._cache_key(text)

        if cache_key in self._cache:
            return self._cache[cache_key]

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )

        embedding = response.data[0].embedding
        self._cache[cache_key] = embedding

        return embedding

    def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts in a single API call."""
        # Check cache for already-embedded texts
        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # Embed remaining texts
        if texts_to_embed:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts_to_embed,
            )

            for j, embedding_data in enumerate(response.data):
                idx = indices_to_embed[j]
                embedding = embedding_data.embedding
                results[idx] = embedding

                # Cache
                cache_key = self._cache_key(texts_to_embed[j])
                self._cache[cache_key] = embedding

        return results

    def _cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _split_identifier(self, identifier: str) -> str:
        """Split a camelCase or snake_case identifier into words."""
        import re

        # Handle snake_case
        if "_" in identifier:
            return identifier.replace("_", " ")

        # Handle camelCase
        words = re.sub(r"([a-z])([A-Z])", r"\1 \2", identifier)
        return words.lower()

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
