"""Vector store for endpoint embeddings using numpy.

Replaces ChromaDB with a lightweight numpy-based implementation.
Uses cosine similarity over flat arrays — identical math, fewer deps.
The dataset is small (50-500 vectors), so O(n) search is fine.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    endpoint_id: str
    api_id: str
    score: float  # Similarity score (higher = more similar)
    metadata: dict[str, Any]


class VectorStore:
    """Vector store for endpoint embeddings using numpy.

    Provides semantic search capabilities over API endpoints.
    Persists data as JSON with numpy arrays stored as lists.
    """

    def __init__(self, storage_dir: str | Path):
        """Initialize the vector store.

        Args:
            storage_dir: Directory for data persistence.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.data_file = self.storage_dir / "vector_store.json"

        # In-memory storage
        self._ids: list[str] = []
        self._embeddings: np.ndarray | None = None  # shape: (n, dim)
        self._metadatas: list[dict[str, Any]] = []
        self._documents: list[str] = []
        self._dimension: int | None = None

        self._load()

    def _load(self) -> None:
        """Load persisted data from disk."""
        if not self.data_file.exists():
            return

        try:
            with open(self.data_file) as f:
                data = json.load(f)

            self._ids = data.get("ids", [])
            self._metadatas = data.get("metadatas", [])
            self._documents = data.get("documents", [])
            self._dimension = data.get("dimension")

            embeddings = data.get("embeddings", [])
            if embeddings:
                self._embeddings = np.array(embeddings, dtype=np.float32)
                self._dimension = self._embeddings.shape[1]
            else:
                self._embeddings = None
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to load vector store: {e}")
            self._ids = []
            self._embeddings = None
            self._metadatas = []
            self._documents = []

    def _save(self) -> None:
        """Persist data to disk."""
        data = {
            "ids": self._ids,
            "metadatas": self._metadatas,
            "documents": self._documents,
            "dimension": self._dimension,
            "embeddings": self._embeddings.tolist() if self._embeddings is not None else [],
        }
        with open(self.data_file, "w") as f:
            json.dump(data, f)

    def add_endpoint(
        self,
        endpoint_id: str,
        api_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        document: str = "",
    ) -> None:
        """Add a single endpoint embedding to the store.

        Args:
            endpoint_id: Unique identifier for the endpoint.
            api_id: The API this endpoint belongs to.
            embedding: The embedding vector.
            metadata: Optional metadata to store with the embedding.
            document: The text that was embedded (for reference).
        """
        doc_id = self._make_id(api_id, endpoint_id)

        meta = metadata.copy() if metadata else {}
        meta["api_id"] = api_id
        meta["endpoint_id"] = endpoint_id
        meta = self._filter_metadata(meta)

        emb = np.array(embedding, dtype=np.float32)

        # Upsert: if exists, replace
        if doc_id in self._ids:
            idx = self._ids.index(doc_id)
            self._embeddings[idx] = emb
            self._metadatas[idx] = meta
            self._documents[idx] = document
        else:
            self._ids.append(doc_id)
            self._metadatas.append(meta)
            self._documents.append(document)

            if self._embeddings is None:
                self._embeddings = emb.reshape(1, -1)
                self._dimension = len(embedding)
            else:
                self._embeddings = np.vstack([self._embeddings, emb.reshape(1, -1)])

        self._save()

    def add_endpoints_batch(
        self,
        endpoints: list[dict[str, Any]],
    ) -> None:
        """Add multiple endpoint embeddings in batch.

        Args:
            endpoints: List of dicts with keys:
                - endpoint_id: str
                - api_id: str
                - embedding: list[float]
                - metadata: dict (optional)
                - document: str (optional)
        """
        if not endpoints:
            return

        for ep in endpoints:
            doc_id = self._make_id(ep["api_id"], ep["endpoint_id"])

            meta = ep.get("metadata", {}).copy()
            meta["api_id"] = ep["api_id"]
            meta["endpoint_id"] = ep["endpoint_id"]
            meta = self._filter_metadata(meta)

            emb = np.array(ep["embedding"], dtype=np.float32)

            if doc_id in self._ids:
                idx = self._ids.index(doc_id)
                self._embeddings[idx] = emb
                self._metadatas[idx] = meta
                self._documents[idx] = ep.get("document", "")
            else:
                self._ids.append(doc_id)
                self._metadatas.append(meta)
                self._documents.append(ep.get("document", ""))

                if self._embeddings is None:
                    self._embeddings = emb.reshape(1, -1)
                    self._dimension = len(ep["embedding"])
                else:
                    self._embeddings = np.vstack([self._embeddings, emb.reshape(1, -1)])

        self._save()

    def search(
        self,
        query_embedding: list[float],
        api_id: str | None = None,
        top_k: int = 10,
        filter_deprecated: bool = True,
    ) -> list[SearchResult]:
        """Search for similar endpoints.

        Args:
            query_embedding: The query embedding vector.
            api_id: Optional filter to search within a specific API.
            top_k: Number of results to return.
            filter_deprecated: Whether to exclude deprecated endpoints.

        Returns:
            List of SearchResult objects, sorted by similarity.
        """
        if self._embeddings is None or len(self._ids) == 0:
            return []

        query = np.array(query_embedding, dtype=np.float32)

        # Compute cosine similarity
        # similarity = dot(a, b) / (norm(a) * norm(b))
        norms = np.linalg.norm(self._embeddings, axis=1)
        query_norm = np.linalg.norm(query)

        # Avoid division by zero
        valid = (norms > 0) & (query_norm > 0)
        similarities = np.zeros(len(self._ids), dtype=np.float32)
        if query_norm > 0:
            similarities[valid] = (
                self._embeddings[valid] @ query / (norms[valid] * query_norm)
            )

        # Build candidate mask
        mask = np.ones(len(self._ids), dtype=bool)
        for i, meta in enumerate(self._metadatas):
            if api_id and meta.get("api_id") != api_id:
                mask[i] = False
            if filter_deprecated and meta.get("deprecated") is True:
                mask[i] = False

        # Apply mask and get top-k
        similarities[~mask] = -1.0
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if not mask[idx]:
                continue
            score = float(similarities[idx])
            if score <= -1.0:
                continue
            meta = self._metadatas[idx]
            results.append(
                SearchResult(
                    endpoint_id=meta.get("endpoint_id", ""),
                    api_id=meta.get("api_id", ""),
                    score=score,
                    metadata=meta,
                )
            )

        return results

    def search_by_text(
        self,
        query_text: str,
        embedder,  # EndpointEmbedder
        api_id: str | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search using a text query (will be embedded first).

        Args:
            query_text: The text query.
            embedder: An EndpointEmbedder instance to generate the embedding.
            api_id: Optional filter to search within a specific API.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects.
        """
        query_embedding = embedder.embed_query(query_text)
        return self.search(query_embedding, api_id=api_id, top_k=top_k)

    def delete_api(self, api_id: str) -> int:
        """Delete all endpoints for an API.

        Args:
            api_id: The API identifier.

        Returns:
            Number of endpoints deleted.
        """
        indices_to_delete = [
            i for i, meta in enumerate(self._metadatas)
            if meta.get("api_id") == api_id
        ]

        if not indices_to_delete:
            return 0

        count = len(indices_to_delete)
        self._remove_indices(indices_to_delete)
        self._save()
        return count

    def delete_endpoint(self, api_id: str, endpoint_id: str) -> bool:
        """Delete a specific endpoint.

        Args:
            api_id: The API identifier.
            endpoint_id: The endpoint identifier.

        Returns:
            True if deleted, False if not found.
        """
        doc_id = self._make_id(api_id, endpoint_id)

        if doc_id not in self._ids:
            return False

        idx = self._ids.index(doc_id)
        self._remove_indices([idx])
        self._save()
        return True

    def get_endpoint(
        self, api_id: str, endpoint_id: str
    ) -> dict[str, Any] | None:
        """Get a specific endpoint's data.

        Args:
            api_id: The API identifier.
            endpoint_id: The endpoint identifier.

        Returns:
            Dict with endpoint data, or None if not found.
        """
        doc_id = self._make_id(api_id, endpoint_id)

        if doc_id not in self._ids:
            return None

        idx = self._ids.index(doc_id)
        return {
            "endpoint_id": endpoint_id,
            "api_id": api_id,
            "metadata": self._metadatas[idx],
            "embedding": self._embeddings[idx].tolist() if self._embeddings is not None else None,
            "document": self._documents[idx],
        }

    def count_endpoints(self, api_id: str | None = None) -> int:
        """Count endpoints in the store.

        Args:
            api_id: Optional filter to count for a specific API.

        Returns:
            Number of endpoints.
        """
        if api_id:
            return sum(
                1 for meta in self._metadatas
                if meta.get("api_id") == api_id
            )
        return len(self._ids)

    def list_apis(self) -> list[str]:
        """List all APIs in the store.

        Returns:
            List of unique API IDs.
        """
        api_ids = set()
        for meta in self._metadatas:
            if "api_id" in meta:
                api_ids.add(meta["api_id"])
        return list(api_ids)

    def _remove_indices(self, indices: list[int]) -> None:
        """Remove entries at given indices."""
        indices_set = set(indices)
        self._ids = [v for i, v in enumerate(self._ids) if i not in indices_set]
        self._metadatas = [v for i, v in enumerate(self._metadatas) if i not in indices_set]
        self._documents = [v for i, v in enumerate(self._documents) if i not in indices_set]

        if self._embeddings is not None:
            keep = [i for i in range(len(self._embeddings)) if i not in indices_set]
            if keep:
                self._embeddings = self._embeddings[keep]
            else:
                self._embeddings = None
                self._dimension = None

    def _make_id(self, api_id: str, endpoint_id: str) -> str:
        """Create a unique document ID."""
        return f"{api_id}::{endpoint_id}"

    def _filter_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Filter metadata to store-compatible values."""
        filtered = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                filtered[key] = value
            elif isinstance(value, list):
                if all(isinstance(v, str) for v in value):
                    filtered[key] = ",".join(value)
        return filtered
