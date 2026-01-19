"""Tests for the retrieval components."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jitapi.retrieval.graph_expander import ExpandedEndpoint, GraphExpander
from jitapi.retrieval.vector_search import VectorSearcher, VectorSearchResult
from jitapi.stores.graph_store import GraphStore
from jitapi.stores.spec_store import SpecStore
from jitapi.stores.vector_store import SearchResult, VectorStore


@pytest.fixture
def temp_storage():
    """Create a temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    embedder = MagicMock()
    # Return a fixed embedding for any input
    embedder.embed_query.return_value = [0.1] * 1536
    return embedder


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = MagicMock(spec=VectorStore)
    return store


@pytest.fixture
def mock_graph_store():
    """Create a mock graph store."""
    store = MagicMock(spec=GraphStore)
    return store


@pytest.fixture
def mock_spec_store():
    """Create a mock spec store."""
    store = MagicMock(spec=SpecStore)
    return store


class TestVectorSearcher:
    """Tests for VectorSearcher."""

    def test_search_returns_results(self, mock_vector_store, mock_embedder):
        """Test that search returns formatted results."""
        # Setup mock return value
        mock_vector_store.search.return_value = [
            SearchResult(
                endpoint_id="GET /pets",
                api_id="petstore",
                score=0.95,
                metadata={
                    "path": "/pets",
                    "method": "GET",
                    "summary": "List all pets",
                    "tags": "pets",
                },
            ),
            SearchResult(
                endpoint_id="POST /pets",
                api_id="petstore",
                score=0.85,
                metadata={
                    "path": "/pets",
                    "method": "POST",
                    "summary": "Create a pet",
                    "tags": "pets",
                },
            ),
        ]

        searcher = VectorSearcher(mock_vector_store, mock_embedder)
        results = searcher.search("find all pets", api_id="petstore", top_k=5)

        assert len(results) == 2
        assert results[0].endpoint_id == "GET /pets"
        assert results[0].score == 0.95
        assert results[0].path == "/pets"
        assert results[0].method == "GET"

    def test_search_filters_by_score(self, mock_vector_store, mock_embedder):
        """Test that search can filter by minimum score."""
        mock_vector_store.search.return_value = [
            SearchResult(
                endpoint_id="GET /pets",
                api_id="petstore",
                score=0.95,
                metadata={"path": "/pets", "method": "GET", "summary": "", "tags": ""},
            ),
            SearchResult(
                endpoint_id="POST /pets",
                api_id="petstore",
                score=0.3,  # Low score
                metadata={"path": "/pets", "method": "POST", "summary": "", "tags": ""},
            ),
        ]

        searcher = VectorSearcher(mock_vector_store, mock_embedder)
        results = searcher.search("find pets", min_score=0.5)

        # Only high-score result should be returned
        assert len(results) == 1
        assert results[0].endpoint_id == "GET /pets"

    def test_search_by_method(self, mock_vector_store, mock_embedder):
        """Test searching with method filter."""
        mock_vector_store.search.return_value = [
            SearchResult(
                endpoint_id="GET /pets",
                api_id="petstore",
                score=0.9,
                metadata={"path": "/pets", "method": "GET", "summary": "", "tags": ""},
            ),
            SearchResult(
                endpoint_id="POST /pets",
                api_id="petstore",
                score=0.85,
                metadata={"path": "/pets", "method": "POST", "summary": "", "tags": ""},
            ),
        ]

        searcher = VectorSearcher(mock_vector_store, mock_embedder)
        results = searcher.search_by_method("pets", "POST")

        assert len(results) == 1
        assert results[0].method == "POST"

    def test_query_expansion(self, mock_vector_store, mock_embedder):
        """Test that query expansion generates variants."""
        searcher = VectorSearcher(mock_vector_store, mock_embedder)
        variants = searcher._expand_query("create user")

        # Should have some variants
        assert len(variants) > 0
        # Variants should be different from original
        assert all(v != "create user" for v in variants)


class TestGraphExpander:
    """Tests for GraphExpander."""

    def test_expand_adds_dependencies(self, mock_graph_store, mock_spec_store):
        """Test that expansion adds dependent endpoints."""
        # Setup mocks
        mock_graph_store.get_dependencies.return_value = [
            {
                "endpoint_id": "GET /pets",
                "parameter": "petId",
                "type": "body_param",
                "confidence": 0.8,
            }
        ]

        # Mock spec store to return endpoint details
        mock_endpoint = MagicMock()
        mock_endpoint.path = "/pets"
        mock_endpoint.method = "GET"
        mock_endpoint.summary = "List pets"
        mock_endpoint.tags = ["pets"]
        mock_spec_store.get_endpoint.return_value = mock_endpoint

        expander = GraphExpander(mock_graph_store, mock_spec_store)

        # Create initial search results
        initial_results = [
            VectorSearchResult(
                endpoint_id="POST /orders",
                api_id="petstore",
                score=0.9,
                path="/orders",
                method="POST",
                summary="Create order",
                tags=["orders"],
                metadata={},
            )
        ]

        result = expander.expand(initial_results, "petstore", max_depth=1)

        assert result.original_count == 1
        # Should have expanded to include the dependency
        assert len(result.endpoints) >= 1

    def test_expand_with_no_dependencies(self, mock_graph_store, mock_spec_store):
        """Test expansion when there are no dependencies."""
        mock_graph_store.get_dependencies.return_value = []

        expander = GraphExpander(mock_graph_store, mock_spec_store)

        initial_results = [
            VectorSearchResult(
                endpoint_id="GET /pets",
                api_id="petstore",
                score=0.9,
                path="/pets",
                method="GET",
                summary="List pets",
                tags=["pets"],
                metadata={},
            )
        ]

        result = expander.expand(initial_results, "petstore")

        # Should just return the original endpoint
        assert result.original_count == 1
        assert result.expanded_count == 0
        assert len(result.endpoints) == 1

    def test_expand_respects_max_total(self, mock_graph_store, mock_spec_store):
        """Test that expansion respects the max_total limit."""
        # Setup to return many dependencies
        mock_graph_store.get_dependencies.return_value = [
            {"endpoint_id": f"GET /endpoint{i}", "parameter": "id", "confidence": 0.8}
            for i in range(10)
        ]

        mock_endpoint = MagicMock()
        mock_endpoint.path = "/test"
        mock_endpoint.method = "GET"
        mock_endpoint.summary = "Test"
        mock_endpoint.tags = []
        mock_spec_store.get_endpoint.return_value = mock_endpoint

        expander = GraphExpander(mock_graph_store, mock_spec_store)

        initial_results = [
            VectorSearchResult(
                endpoint_id="POST /orders",
                api_id="petstore",
                score=0.9,
                path="/orders",
                method="POST",
                summary="Create order",
                tags=[],
                metadata={},
            )
        ]

        result = expander.expand(initial_results, "petstore", max_total=5)

        # Should not exceed max_total
        assert len(result.endpoints) <= 5

    def test_expanded_endpoint_metadata(self, mock_graph_store, mock_spec_store):
        """Test that expanded endpoints have proper metadata."""
        mock_graph_store.get_dependencies.return_value = [
            {
                "endpoint_id": "GET /pets",
                "parameter": "petId",
                "type": "body_param",
                "confidence": 0.8,
            }
        ]

        mock_endpoint = MagicMock()
        mock_endpoint.path = "/pets"
        mock_endpoint.method = "GET"
        mock_endpoint.summary = "List pets"
        mock_endpoint.tags = ["pets"]
        mock_spec_store.get_endpoint.return_value = mock_endpoint

        expander = GraphExpander(mock_graph_store, mock_spec_store)

        initial_results = [
            VectorSearchResult(
                endpoint_id="POST /orders",
                api_id="petstore",
                score=0.9,
                path="/orders",
                method="POST",
                summary="Create order",
                tags=["orders"],
                metadata={},
            )
        ]

        result = expander.expand(initial_results, "petstore", max_depth=1)

        # Find the original endpoint
        original = next(ep for ep in result.endpoints if ep.endpoint_id == "POST /orders")
        assert original.is_dependency is False
        assert len(original.depends_on) > 0

        # If dependency was added, check its metadata
        if len(result.endpoints) > 1:
            dependency = next(
                (ep for ep in result.endpoints if ep.endpoint_id == "GET /pets"),
                None,
            )
            if dependency:
                assert dependency.is_dependency is True
                assert "POST /orders" in dependency.provides_for
