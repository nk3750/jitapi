"""Vector store for endpoint embeddings using ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    endpoint_id: str
    api_id: str
    score: float  # Similarity score (higher = more similar)
    metadata: dict[str, Any]


class VectorStore:
    """Vector store for endpoint embeddings using ChromaDB.

    Provides semantic search capabilities over API endpoints.
    """

    COLLECTION_NAME = "endpoints"

    def __init__(self, storage_dir: str | Path):
        """Initialize the vector store.

        Args:
            storage_dir: Directory for ChromaDB persistence.
        """
        self.storage_dir = Path(storage_dir)
        self.chroma_dir = self.storage_dir / "chroma"
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create the endpoints collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

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
        # Create unique ID combining api_id and endpoint_id
        doc_id = self._make_id(api_id, endpoint_id)

        # Prepare metadata
        meta = metadata or {}
        meta["api_id"] = api_id
        meta["endpoint_id"] = endpoint_id

        # Filter out None values and non-string/numeric values for ChromaDB
        meta = self._filter_metadata(meta)

        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[meta],
            documents=[document] if document else None,
        )

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

        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for ep in endpoints:
            doc_id = self._make_id(ep["api_id"], ep["endpoint_id"])
            ids.append(doc_id)
            embeddings.append(ep["embedding"])

            meta = ep.get("metadata", {})
            meta["api_id"] = ep["api_id"]
            meta["endpoint_id"] = ep["endpoint_id"]
            metadatas.append(self._filter_metadata(meta))

            documents.append(ep.get("document", ""))

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

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
        # Build where clause
        where = None
        where_conditions = []

        if api_id:
            where_conditions.append({"api_id": {"$eq": api_id}})

        if filter_deprecated:
            where_conditions.append({"deprecated": {"$ne": True}})

        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}

        # Execute search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances", "documents"],
        )

        # Convert to SearchResult objects
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                # ChromaDB returns distances (lower = better)
                # Convert to similarity scores (higher = better)
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance  # For cosine distance

                search_results.append(
                    SearchResult(
                        endpoint_id=metadata.get("endpoint_id", ""),
                        api_id=metadata.get("api_id", ""),
                        score=similarity,
                        metadata=metadata,
                    )
                )

        return search_results

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
        # Get all endpoints for this API
        results = self.collection.get(
            where={"api_id": {"$eq": api_id}},
            include=[],
        )

        if not results["ids"]:
            return 0

        count = len(results["ids"])
        self.collection.delete(ids=results["ids"])
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

        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False

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

        results = self.collection.get(
            ids=[doc_id],
            include=["metadatas", "embeddings", "documents"],
        )

        if not results["ids"]:
            return None

        return {
            "endpoint_id": endpoint_id,
            "api_id": api_id,
            "metadata": results["metadatas"][0] if results["metadatas"] else {},
            "embedding": results["embeddings"][0] if results["embeddings"] else None,
            "document": results["documents"][0] if results["documents"] else "",
        }

    def count_endpoints(self, api_id: str | None = None) -> int:
        """Count endpoints in the store.

        Args:
            api_id: Optional filter to count for a specific API.

        Returns:
            Number of endpoints.
        """
        if api_id:
            results = self.collection.get(
                where={"api_id": {"$eq": api_id}},
                include=[],
            )
            return len(results["ids"])
        else:
            return self.collection.count()

    def list_apis(self) -> list[str]:
        """List all APIs in the store.

        Returns:
            List of unique API IDs.
        """
        # Get all unique api_ids
        results = self.collection.get(include=["metadatas"])

        api_ids = set()
        if results["metadatas"]:
            for meta in results["metadatas"]:
                if "api_id" in meta:
                    api_ids.add(meta["api_id"])

        return list(api_ids)

    def _make_id(self, api_id: str, endpoint_id: str) -> str:
        """Create a unique document ID."""
        return f"{api_id}::{endpoint_id}"

    def _filter_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Filter metadata to only include ChromaDB-compatible values."""
        filtered = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                filtered[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                if all(isinstance(v, str) for v in value):
                    filtered[key] = ",".join(value)
        return filtered
