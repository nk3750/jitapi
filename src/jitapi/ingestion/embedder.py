"""Embedding generator for API endpoints.

Creates vector embeddings from endpoint metadata for semantic search.
Supports multiple embedding providers with automatic detection.

Provider selection:
1. JITAPI_EMBEDDING_PROVIDER env var (explicit override)
2. VOYAGE_API_KEY set → Voyage AI
3. OPENAI_API_KEY set → OpenAI
4. COHERE_API_KEY set → Cohere
5. None set → local fastembed (zero-config fallback)
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .parser import Endpoint

logger = logging.getLogger(__name__)


@dataclass
class EndpointEmbedding:
    """An endpoint with its embedding vector."""

    endpoint_id: str
    api_id: str
    embedding: list[float]
    text: str  # The text that was embedded
    metadata: dict[str, Any]


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this provider."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.
        """

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """


class VoyageProvider(EmbeddingProvider):
    """Voyage AI embedding provider (Anthropic's recommended choice)."""

    MODEL = "voyage-2"
    DIMENSION = 1024

    def __init__(self, api_key: str | None = None):
        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "voyageai package required for Voyage embeddings. "
                "Install with: pip install jitapi[voyage]"
            )
        self._client = voyageai.Client(api_key=api_key or os.environ.get("VOYAGE_API_KEY"))

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    @property
    def name(self) -> str:
        return "voyage"

    def embed(self, text: str) -> list[float]:
        result = self._client.embed([text], model=self.MODEL)
        return result.embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, model=self.MODEL)
        return result.embeddings


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    MODEL = "text-embedding-3-small"
    DIMENSION = 1536

    def __init__(self, api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for OpenAI embeddings. "
                "Install with: pip install jitapi[openai]"
            )
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    @property
    def name(self) -> str:
        return "openai"

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.MODEL, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.MODEL, input=texts)
        return [item.embedding for item in response.data]


class CohereProvider(EmbeddingProvider):
    """Cohere embedding provider."""

    MODEL = "embed-english-v3.0"
    DIMENSION = 1024

    def __init__(self, api_key: str | None = None):
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere package required for Cohere embeddings. "
                "Install with: pip install jitapi[cohere]"
            )
        self._client = cohere.Client(api_key=api_key or os.environ.get("COHERE_API_KEY"))

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    @property
    def name(self) -> str:
        return "cohere"

    def embed(self, text: str) -> list[float]:
        response = self._client.embed(
            texts=[text], model=self.MODEL, input_type="search_document"
        )
        return response.embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(
            texts=texts, model=self.MODEL, input_type="search_document"
        )
        return response.embeddings


class LocalProvider(EmbeddingProvider):
    """Local embedding provider using fastembed (zero-config fallback)."""

    MODEL = "BAAI/bge-small-en-v1.5"
    DIMENSION = 384

    def __init__(self):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "fastembed package required for local embeddings. "
                "Install with: pip install fastembed"
            )
        logger.info(f"Initializing local embedding model: {self.MODEL}")
        self._model = TextEmbedding(model_name=self.MODEL)

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    @property
    def name(self) -> str:
        return "local"

    def embed(self, text: str) -> list[float]:
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = list(self._model.embed(texts))
        return [e.tolist() for e in embeddings]


def detect_provider(api_key: str | None = None) -> EmbeddingProvider:
    """Auto-detect and create the best available embedding provider.

    Priority:
    1. JITAPI_EMBEDDING_PROVIDER env var (explicit override)
    2. VOYAGE_API_KEY set → Voyage AI
    3. OPENAI_API_KEY set or api_key param → OpenAI
    4. COHERE_API_KEY set → Cohere
    5. None set → local fastembed

    Args:
        api_key: Optional API key (used for OpenAI backward compatibility).

    Returns:
        An EmbeddingProvider instance.
    """
    explicit = os.environ.get("JITAPI_EMBEDDING_PROVIDER", "").lower()

    if explicit == "voyage":
        logger.info("Using Voyage AI embeddings (explicit config)")
        return VoyageProvider()
    elif explicit == "openai":
        logger.info("Using OpenAI embeddings (explicit config)")
        return OpenAIProvider(api_key=api_key)
    elif explicit == "cohere":
        logger.info("Using Cohere embeddings (explicit config)")
        return CohereProvider()
    elif explicit == "local":
        logger.info("Using local fastembed embeddings (explicit config)")
        return LocalProvider()

    # Auto-detect
    if os.environ.get("VOYAGE_API_KEY"):
        try:
            provider = VoyageProvider()
            logger.info("Using Voyage AI embeddings (auto-detected VOYAGE_API_KEY)")
            return provider
        except ImportError:
            logger.warning("VOYAGE_API_KEY set but voyageai not installed, trying next provider")

    if api_key or os.environ.get("OPENAI_API_KEY"):
        try:
            provider = OpenAIProvider(api_key=api_key)
            logger.info("Using OpenAI embeddings (auto-detected OPENAI_API_KEY)")
            return provider
        except ImportError:
            logger.warning("OPENAI_API_KEY set but openai not installed, trying next provider")

    if os.environ.get("COHERE_API_KEY"):
        try:
            provider = CohereProvider()
            logger.info("Using Cohere embeddings (auto-detected COHERE_API_KEY)")
            return provider
        except ImportError:
            logger.warning("COHERE_API_KEY set but cohere not installed, trying next provider")

    # Fallback to local
    logger.info("Using local fastembed embeddings (no API keys configured)")
    return LocalProvider()


class EndpointEmbedder:
    """Generates embeddings for API endpoints.

    Uses a pluggable embedding provider. Provider is auto-detected from
    available API keys, or can be explicitly configured via
    JITAPI_EMBEDDING_PROVIDER env var.
    """

    def __init__(
        self,
        api_key: str | None = None,
        provider: EmbeddingProvider | None = None,
    ):
        """Initialize the embedder.

        Args:
            api_key: Optional API key (for backward compatibility with OpenAI).
            provider: Optional explicit embedding provider. If None, auto-detected.
        """
        self.provider = provider or detect_provider(api_key=api_key)
        self._cache: dict[str, list[float]] = {}

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for the current provider."""
        return self.provider.dimension

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
        texts = [self._build_embedding_text(ep) for ep in endpoints]

        # Check cache, batch embed uncached
        results_map: dict[int, list[float]] = {}
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                results_map[i] = self._cache[cache_key]
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        if texts_to_embed:
            embeddings = self.provider.embed_batch(texts_to_embed)
            for j, emb in enumerate(embeddings):
                idx = indices_to_embed[j]
                results_map[idx] = emb
                self._cache[self._cache_key(texts_to_embed[j])] = emb

        results = []
        for i, (endpoint, text) in enumerate(zip(endpoints, texts)):
            results.append(
                EndpointEmbedding(
                    endpoint_id=endpoint.endpoint_id,
                    api_id=api_id,
                    embedding=results_map[i],
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
        """Build the text representation of an endpoint for embedding."""
        parts = []

        parts.append(f"{endpoint.method} {endpoint.path}")

        if endpoint.summary:
            parts.append(endpoint.summary)
        if endpoint.description and endpoint.description != endpoint.summary:
            desc = endpoint.description[:500]
            parts.append(desc)

        if endpoint.operation_id:
            op_words = self._split_identifier(endpoint.operation_id)
            parts.append(op_words)

        if endpoint.tags:
            parts.append(f"Category: {', '.join(endpoint.tags)}")

        param_names = [p.name for p in endpoint.parameters]
        if param_names:
            parts.append(f"Parameters: {', '.join(param_names)}")

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

        embedding = self.provider.embed(text)
        self._cache[cache_key] = embedding
        return embedding

    def _cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _split_identifier(self, identifier: str) -> str:
        """Split a camelCase or snake_case identifier into words."""
        import re

        if "_" in identifier:
            return identifier.replace("_", " ")

        words = re.sub(r"([a-z])([A-Z])", r"\1 \2", identifier)
        return words.lower()

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
