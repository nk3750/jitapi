"""Tests for the embedding provider abstraction."""

from unittest.mock import MagicMock, patch

import pytest

from jitapi.ingestion.embedder import (
    EmbeddingProvider,
    EndpointEmbedder,
    LocalProvider,
    detect_provider,
)
from jitapi.ingestion.parser import Endpoint, Parameter


class FakeProvider(EmbeddingProvider):
    """A fake embedding provider for testing."""

    DIMENSION = 128

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    @property
    def name(self) -> str:
        return "fake"

    def embed(self, text: str) -> list[float]:
        # Deterministic: hash the text to a vector
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        vec = [float(b) / 255.0 for b in h[:self.DIMENSION]]
        # Pad to dimension
        while len(vec) < self.DIMENSION:
            vec.extend(vec[:self.DIMENSION - len(vec)])
        return vec[:self.DIMENSION]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def fake_provider():
    return FakeProvider()


@pytest.fixture
def embedder(fake_provider):
    return EndpointEmbedder(provider=fake_provider)


@pytest.fixture
def sample_endpoint():
    """Create a sample endpoint for testing."""
    return Endpoint(
        endpoint_id="GET /pets",
        path="/pets",
        method="GET",
        summary="List all pets",
        description="Returns a list of pets",
        operation_id="listPets",
        tags=["pets"],
        parameters=[
            Parameter(name="limit", location="query", required=False, schema_type="integer"),
            Parameter(name="offset", location="query", required=False, schema_type="integer"),
        ],
        required_params=[],
    )


class TestEmbeddingProviderInterface:
    """Tests for the EmbeddingProvider abstract interface."""

    def test_provider_has_dimension(self, fake_provider):
        assert fake_provider.dimension == 128

    def test_provider_has_name(self, fake_provider):
        assert fake_provider.name == "fake"

    def test_provider_embed(self, fake_provider):
        result = fake_provider.embed("hello world")
        assert isinstance(result, list)
        assert len(result) == 128
        assert all(isinstance(v, float) for v in result)

    def test_provider_embed_batch(self, fake_provider):
        results = fake_provider.embed_batch(["hello", "world"])
        assert len(results) == 2
        assert len(results[0]) == 128
        assert len(results[1]) == 128

    def test_provider_embed_deterministic(self, fake_provider):
        """Same input should produce same output."""
        emb1 = fake_provider.embed("test")
        emb2 = fake_provider.embed("test")
        assert emb1 == emb2

    def test_provider_embed_different_inputs(self, fake_provider):
        """Different inputs should produce different outputs."""
        emb1 = fake_provider.embed("hello")
        emb2 = fake_provider.embed("world")
        assert emb1 != emb2


class TestEndpointEmbedder:
    """Tests for EndpointEmbedder."""

    def test_embedder_uses_provider(self, embedder, fake_provider):
        assert embedder.provider is fake_provider

    def test_embedder_dimension(self, embedder):
        assert embedder.dimension == 128

    def test_embed_query(self, embedder):
        result = embedder.embed_query("find all pets")
        assert isinstance(result, list)
        assert len(result) == 128

    def test_embed_endpoint(self, embedder, sample_endpoint):
        result = embedder.embed_endpoint(sample_endpoint, "petstore")
        assert result.endpoint_id == "GET /pets"
        assert result.api_id == "petstore"
        assert len(result.embedding) == 128
        assert "GET /pets" in result.text
        assert result.metadata["api_id"] == "petstore"

    def test_embed_endpoints_batch(self, embedder, sample_endpoint):
        endpoints = [sample_endpoint, sample_endpoint]
        results = embedder.embed_endpoints(endpoints, "petstore")
        assert len(results) == 2
        for r in results:
            assert len(r.embedding) == 128

    def test_embedding_text_includes_key_info(self, embedder, sample_endpoint):
        text = embedder._build_embedding_text(sample_endpoint)
        assert "GET /pets" in text
        assert "List all pets" in text
        assert "pets" in text
        assert "limit" in text

    def test_metadata_includes_key_fields(self, embedder, sample_endpoint):
        meta = embedder._build_metadata(sample_endpoint, "petstore")
        assert meta["api_id"] == "petstore"
        assert meta["endpoint_id"] == "GET /pets"
        assert meta["path"] == "/pets"
        assert meta["method"] == "GET"
        assert meta["summary"] == "List all pets"

    def test_caching(self, embedder):
        """Test that embeddings are cached."""
        emb1 = embedder.embed_query("test query")
        emb2 = embedder.embed_query("test query")
        assert emb1 == emb2

        embedder.clear_cache()
        emb3 = embedder.embed_query("test query")
        assert emb1 == emb3  # Same result even after cache clear

    def test_split_identifier_camelcase(self, embedder):
        assert embedder._split_identifier("listPets") == "list Pets".lower()

    def test_split_identifier_snakecase(self, embedder):
        assert embedder._split_identifier("list_pets") == "list pets"


class TestDetectProvider:
    """Tests for the auto-detection logic."""

    @patch.dict("os.environ", {}, clear=True)
    def test_detect_local_fallback(self):
        """With no env vars, should fall back to local provider."""
        provider = detect_provider()
        assert isinstance(provider, LocalProvider)

    @patch.dict("os.environ", {"JITAPI_EMBEDDING_PROVIDER": "local"}, clear=True)
    def test_detect_explicit_local(self):
        """Explicit local config."""
        provider = detect_provider()
        assert isinstance(provider, LocalProvider)

    @patch.dict("os.environ", {"JITAPI_EMBEDDING_PROVIDER": "openai", "OPENAI_API_KEY": "test"})
    def test_detect_explicit_openai(self):
        """Explicit OpenAI config (skipped if openai not installed)."""
        try:
            provider = detect_provider()
            assert provider.name == "openai"
        except ImportError:
            pytest.skip("openai not installed")

    @patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"})
    def test_detect_voyage_auto(self):
        """Auto-detect Voyage from env (skipped if voyageai not installed)."""
        try:
            import voyageai
        except ImportError:
            pytest.skip("voyageai not installed")
        provider = detect_provider()
        assert provider.name == "voyage"

    @patch.dict("os.environ", {"COHERE_API_KEY": "test-key"})
    def test_detect_cohere_auto(self):
        """Auto-detect Cohere from env (skipped if cohere not installed)."""
        try:
            import cohere
        except ImportError:
            pytest.skip("cohere not installed")
        provider = detect_provider()
        assert provider.name == "cohere"

    @patch.dict("os.environ", {}, clear=True)
    def test_detect_with_api_key_param(self):
        """Passing api_key should try OpenAI (skipped if not installed)."""
        try:
            provider = detect_provider(api_key="sk-test")
            assert provider.name == "openai"
        except ImportError:
            # OpenAI not installed, should fall through to local
            provider = detect_provider(api_key="sk-test")
            assert isinstance(provider, LocalProvider)


class TestLocalProvider:
    """Tests for the local fastembed provider."""

    @pytest.fixture(scope="class")
    def local_provider(self):
        return LocalProvider()

    def test_dimension(self, local_provider):
        assert local_provider.dimension == 384

    def test_name(self, local_provider):
        assert local_provider.name == "local"

    def test_embed_single(self, local_provider):
        result = local_provider.embed("test query")
        assert isinstance(result, list)
        assert len(result) == 384

    def test_embed_batch(self, local_provider):
        results = local_provider.embed_batch(["hello", "world"])
        assert len(results) == 2
        assert all(len(r) == 384 for r in results)

    def test_embed_deterministic(self, local_provider):
        emb1 = local_provider.embed("deterministic test")
        emb2 = local_provider.embed("deterministic test")
        assert emb1 == emb2
