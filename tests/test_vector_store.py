"""Tests for the numpy-based vector store."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from jitapi.stores.vector_store import SearchResult, VectorStore


@pytest.fixture
def temp_storage():
    """Create a temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def vector_store(temp_storage):
    """Create a vector store instance."""
    return VectorStore(temp_storage)


def _make_embedding(dim=384, seed=0):
    """Create a deterministic random embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


class TestVectorStore:
    """Tests for the numpy VectorStore."""

    def test_add_and_count(self, vector_store):
        """Test adding a single endpoint and counting."""
        vector_store.add_endpoint(
            endpoint_id="GET /pets",
            api_id="petstore",
            embedding=_make_embedding(seed=1),
            metadata={"path": "/pets", "method": "GET"},
            document="List all pets",
        )
        assert vector_store.count_endpoints() == 1
        assert vector_store.count_endpoints(api_id="petstore") == 1
        assert vector_store.count_endpoints(api_id="other") == 0

    def test_add_batch(self, vector_store):
        """Test batch adding endpoints."""
        endpoints = [
            {
                "endpoint_id": "GET /pets",
                "api_id": "petstore",
                "embedding": _make_embedding(seed=1),
                "metadata": {"path": "/pets", "method": "GET"},
                "document": "List all pets",
            },
            {
                "endpoint_id": "POST /pets",
                "api_id": "petstore",
                "embedding": _make_embedding(seed=2),
                "metadata": {"path": "/pets", "method": "POST"},
                "document": "Create a pet",
            },
            {
                "endpoint_id": "GET /users",
                "api_id": "userapi",
                "embedding": _make_embedding(seed=3),
                "metadata": {"path": "/users", "method": "GET"},
                "document": "List users",
            },
        ]
        vector_store.add_endpoints_batch(endpoints)

        assert vector_store.count_endpoints() == 3
        assert vector_store.count_endpoints(api_id="petstore") == 2
        assert vector_store.count_endpoints(api_id="userapi") == 1

    def test_upsert(self, vector_store):
        """Test that adding with same ID updates instead of duplicating."""
        vector_store.add_endpoint(
            endpoint_id="GET /pets",
            api_id="petstore",
            embedding=_make_embedding(seed=1),
            document="Version 1",
        )
        assert vector_store.count_endpoints() == 1

        vector_store.add_endpoint(
            endpoint_id="GET /pets",
            api_id="petstore",
            embedding=_make_embedding(seed=2),
            document="Version 2",
        )
        assert vector_store.count_endpoints() == 1

        ep = vector_store.get_endpoint("petstore", "GET /pets")
        assert ep["document"] == "Version 2"

    def test_search_cosine_similarity(self, vector_store):
        """Test that search returns results sorted by cosine similarity."""
        # Create a query and a very similar endpoint
        query_emb = _make_embedding(seed=42)
        similar_emb = _make_embedding(seed=42)  # Same seed = identical = max similarity
        different_emb = _make_embedding(seed=99)

        vector_store.add_endpoint(
            endpoint_id="GET /similar",
            api_id="test",
            embedding=similar_emb,
        )
        vector_store.add_endpoint(
            endpoint_id="GET /different",
            api_id="test",
            embedding=different_emb,
        )

        results = vector_store.search(query_emb, top_k=2)
        assert len(results) == 2
        assert results[0].endpoint_id == "GET /similar"
        assert results[0].score > results[1].score
        # Identical vector should have similarity ~1.0
        assert results[0].score > 0.99

    def test_search_filter_by_api(self, vector_store):
        """Test that search filters by api_id."""
        emb = _make_embedding(seed=1)
        vector_store.add_endpoint("GET /a", "api1", emb)
        vector_store.add_endpoint("GET /b", "api2", emb)

        results = vector_store.search(emb, api_id="api1")
        assert len(results) == 1
        assert results[0].api_id == "api1"

    def test_search_filter_deprecated(self, vector_store):
        """Test that search can filter deprecated endpoints."""
        emb = _make_embedding(seed=1)
        vector_store.add_endpoint(
            "GET /old", "test", emb,
            metadata={"deprecated": True},
        )
        vector_store.add_endpoint(
            "GET /new", "test", emb,
            metadata={"deprecated": False},
        )

        # With filter (default)
        results = vector_store.search(emb, filter_deprecated=True)
        assert len(results) == 1
        assert results[0].endpoint_id == "GET /new"

        # Without filter
        results = vector_store.search(emb, filter_deprecated=False)
        assert len(results) == 2

    def test_search_top_k(self, vector_store):
        """Test top_k limit on search."""
        emb = _make_embedding(seed=1)
        for i in range(10):
            vector_store.add_endpoint(f"GET /ep{i}", "test", _make_embedding(seed=i))

        results = vector_store.search(emb, top_k=3)
        assert len(results) <= 3

    def test_search_empty_store(self, vector_store):
        """Test searching an empty store."""
        results = vector_store.search(_make_embedding(seed=1))
        assert results == []

    def test_delete_api(self, vector_store):
        """Test deleting all endpoints for an API."""
        vector_store.add_endpoint("GET /a", "api1", _make_embedding(seed=1))
        vector_store.add_endpoint("GET /b", "api1", _make_embedding(seed=2))
        vector_store.add_endpoint("GET /c", "api2", _make_embedding(seed=3))

        count = vector_store.delete_api("api1")
        assert count == 2
        assert vector_store.count_endpoints() == 1
        assert vector_store.count_endpoints(api_id="api1") == 0
        assert vector_store.count_endpoints(api_id="api2") == 1

    def test_delete_api_not_found(self, vector_store):
        """Test deleting a non-existent API."""
        count = vector_store.delete_api("nonexistent")
        assert count == 0

    def test_delete_endpoint(self, vector_store):
        """Test deleting a single endpoint."""
        vector_store.add_endpoint("GET /pets", "test", _make_embedding(seed=1))
        vector_store.add_endpoint("POST /pets", "test", _make_embedding(seed=2))

        assert vector_store.delete_endpoint("test", "GET /pets") is True
        assert vector_store.count_endpoints() == 1
        assert vector_store.delete_endpoint("test", "GET /pets") is False

    def test_get_endpoint(self, vector_store):
        """Test getting a specific endpoint."""
        emb = _make_embedding(seed=1)
        vector_store.add_endpoint(
            "GET /pets", "petstore", emb,
            metadata={"path": "/pets", "method": "GET"},
            document="List all pets",
        )

        ep = vector_store.get_endpoint("petstore", "GET /pets")
        assert ep is not None
        assert ep["endpoint_id"] == "GET /pets"
        assert ep["api_id"] == "petstore"
        assert ep["document"] == "List all pets"
        assert ep["embedding"] is not None
        assert len(ep["embedding"]) == 384

    def test_get_endpoint_not_found(self, vector_store):
        """Test getting a non-existent endpoint."""
        assert vector_store.get_endpoint("x", "y") is None

    def test_list_apis(self, vector_store):
        """Test listing all APIs."""
        vector_store.add_endpoint("GET /a", "api1", _make_embedding(seed=1))
        vector_store.add_endpoint("GET /b", "api2", _make_embedding(seed=2))
        vector_store.add_endpoint("GET /c", "api1", _make_embedding(seed=3))

        apis = vector_store.list_apis()
        assert set(apis) == {"api1", "api2"}

    def test_persistence(self, temp_storage):
        """Test that data persists across VectorStore instances."""
        store1 = VectorStore(temp_storage)
        store1.add_endpoint("GET /pets", "petstore", _make_embedding(seed=1),
                            document="List pets")

        # Create a new instance from same directory
        store2 = VectorStore(temp_storage)
        assert store2.count_endpoints() == 1
        ep = store2.get_endpoint("petstore", "GET /pets")
        assert ep is not None
        assert ep["document"] == "List pets"

        # Search should also work
        results = store2.search(_make_embedding(seed=1))
        assert len(results) == 1
        assert results[0].endpoint_id == "GET /pets"

    def test_filter_metadata(self, vector_store):
        """Test metadata filtering for storage compatibility."""
        meta = vector_store._filter_metadata({
            "string": "value",
            "integer": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list_str": ["a", "b"],
            "list_int": [1, 2],
            "nested": {"key": "value"},
        })

        assert meta["string"] == "value"
        assert meta["integer"] == 42
        assert meta["float"] == 3.14
        assert meta["bool"] is True
        assert "none" not in meta
        assert meta["list_str"] == "a,b"
        assert "list_int" not in meta
        assert "nested" not in meta

    def test_batch_add_empty(self, vector_store):
        """Test batch adding with empty list."""
        vector_store.add_endpoints_batch([])
        assert vector_store.count_endpoints() == 0

    def test_search_returns_search_result_type(self, vector_store):
        """Test that search results are SearchResult instances."""
        vector_store.add_endpoint("GET /pets", "test", _make_embedding(seed=1))
        results = vector_store.search(_make_embedding(seed=1))

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert isinstance(results[0].score, float)
        assert isinstance(results[0].metadata, dict)
