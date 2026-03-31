"""Tests for the LLM reranker with MCP sampling and OpenAI fallback."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jitapi.retrieval.graph_expander import ExpandedEndpoint
from jitapi.retrieval.reranker import LLMReranker, RerankedWorkflow, WorkflowStep


@pytest.fixture
def sample_endpoints():
    """Create sample expanded endpoints for testing."""
    return [
        ExpandedEndpoint(
            endpoint_id="GET /geo/1.0/direct",
            api_id="weather",
            path="/geo/1.0/direct",
            method="GET",
            summary="Geocode a city name to coordinates",
            tags=["geocoding"],
            score=0.9,
            is_dependency=True,
            depends_on=[],
            provides_for=["GET /data/2.5/weather"],
            dependency_params=[],
        ),
        ExpandedEndpoint(
            endpoint_id="GET /data/2.5/weather",
            api_id="weather",
            path="/data/2.5/weather",
            method="GET",
            summary="Get current weather for coordinates",
            tags=["weather"],
            score=0.95,
            is_dependency=False,
            depends_on=["GET /geo/1.0/direct"],
            provides_for=[],
            dependency_params=["lat", "lon"],
        ),
    ]


@pytest.fixture
def mock_mcp_session():
    """Create a mock MCP session with create_message."""
    session = AsyncMock()
    return session


@pytest.fixture
def sample_llm_response():
    """Sample JSON response from the LLM."""
    return json.dumps({
        "reasoning": "Need to geocode Tokyo first, then get weather.",
        "steps": [
            {
                "endpoint_id": "GET /geo/1.0/direct",
                "purpose": "Get coordinates for Tokyo",
                "requires": [],
                "provides": ["lat", "lon"],
                "relevance_score": 0.9,
                "parameters": {
                    "q": {"value": "Tokyo", "source": "user_query"}
                },
                "output_mapping": {
                    "lat": "$[0].lat",
                    "lon": "$[0].lon"
                }
            },
            {
                "endpoint_id": "GET /data/2.5/weather",
                "purpose": "Get weather at coordinates",
                "requires": ["lat", "lon"],
                "provides": ["weather_data"],
                "relevance_score": 1.0,
                "parameters": {
                    "lat": {"value": None, "source": "step_1.lat"},
                    "lon": {"value": None, "source": "step_1.lon"}
                },
                "output_mapping": {}
            }
        ],
        "excluded": []
    })


class TestLLMReranker:
    """Tests for LLMReranker."""

    @pytest.mark.asyncio
    async def test_rerank_empty_endpoints(self):
        """Test reranking with no endpoints."""
        reranker = LLMReranker()
        result = await reranker.rerank("weather in Tokyo", [])

        assert isinstance(result, RerankedWorkflow)
        assert result.steps == []
        assert result.total_steps == 0

    @pytest.mark.asyncio
    async def test_rerank_with_mcp_sampling(
        self, sample_endpoints, mock_mcp_session, sample_llm_response
    ):
        """Test reranking using MCP sampling."""
        # Mock the MCP session response
        mock_result = MagicMock()
        mock_result.content = MagicMock()
        mock_result.content.text = sample_llm_response
        mock_mcp_session.create_message.return_value = mock_result

        reranker = LLMReranker()
        result = await reranker.rerank(
            "weather in Tokyo",
            sample_endpoints,
            mcp_session=mock_mcp_session,
        )

        assert isinstance(result, RerankedWorkflow)
        assert result.total_steps == 2
        assert result.steps[0].endpoint_id == "GET /geo/1.0/direct"
        assert result.steps[1].endpoint_id == "GET /data/2.5/weather"

        # Check parameter extraction
        assert result.steps[0].parameters["q"].value == "Tokyo"
        assert result.steps[0].parameters["q"].source == "user_query"

        # Check data flow mapping
        assert result.steps[1].parameters["lat"].source == "step_1.lat"

        # Verify MCP session was called
        mock_mcp_session.create_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_rerank_mcp_failure_falls_back_to_openai(
        self, sample_endpoints, mock_mcp_session, sample_llm_response
    ):
        """Test that MCP failure falls back to OpenAI."""
        # MCP sampling fails
        mock_mcp_session.create_message.side_effect = Exception("Sampling not supported")

        # Mock OpenAI client
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = sample_llm_response
        mock_openai.chat.completions.create.return_value = mock_response

        reranker = LLMReranker()
        reranker._openai_client = mock_openai

        result = await reranker.rerank(
            "weather in Tokyo",
            sample_endpoints,
            mcp_session=mock_mcp_session,
        )

        assert result.total_steps == 2
        mock_openai.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_rerank_no_backend_raises_error(self, sample_endpoints):
        """Test that missing both backends raises RuntimeError."""
        reranker = LLMReranker()
        reranker._openai_client = None

        with pytest.raises(RuntimeError, match="No LLM backend available"):
            await reranker.rerank("weather in Tokyo", sample_endpoints)

    @pytest.mark.asyncio
    async def test_rerank_max_steps(
        self, sample_endpoints, mock_mcp_session
    ):
        """Test that max_steps limits the output."""
        # Return 2 steps but limit to 1
        response = json.dumps({
            "reasoning": "test",
            "steps": [
                {
                    "endpoint_id": "GET /geo/1.0/direct",
                    "purpose": "Geocode",
                    "relevance_score": 0.9,
                    "parameters": {},
                    "output_mapping": {},
                },
                {
                    "endpoint_id": "GET /data/2.5/weather",
                    "purpose": "Weather",
                    "relevance_score": 1.0,
                    "parameters": {},
                    "output_mapping": {},
                },
            ],
            "excluded": [],
        })

        mock_result = MagicMock()
        mock_result.content.text = response
        mock_mcp_session.create_message.return_value = mock_result

        reranker = LLMReranker()
        result = await reranker.rerank(
            "weather", sample_endpoints, max_steps=1,
            mcp_session=mock_mcp_session,
        )

        assert result.total_steps == 1

    @pytest.mark.asyncio
    async def test_score_relevance_with_mcp(self, mock_mcp_session, sample_endpoints):
        """Test scoring a single endpoint."""
        mock_result = MagicMock()
        mock_result.content.text = "0.85"
        mock_mcp_session.create_message.return_value = mock_result

        reranker = LLMReranker()
        score = await reranker.score_relevance(
            "weather in Tokyo",
            sample_endpoints[0],
            mcp_session=mock_mcp_session,
        )

        assert score == 0.85

    @pytest.mark.asyncio
    async def test_score_relevance_clamps(self, mock_mcp_session, sample_endpoints):
        """Test that scores are clamped to [0, 1]."""
        mock_result = MagicMock()
        mock_result.content.text = "1.5"
        mock_mcp_session.create_message.return_value = mock_result

        reranker = LLMReranker()
        score = await reranker.score_relevance(
            "test", sample_endpoints[0], mcp_session=mock_mcp_session,
        )
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_score_relevance_invalid_returns_default(
        self, mock_mcp_session, sample_endpoints
    ):
        """Test that invalid score response returns 0.5."""
        mock_result = MagicMock()
        mock_result.content.text = "not a number"
        mock_mcp_session.create_message.return_value = mock_result

        reranker = LLMReranker()
        score = await reranker.score_relevance(
            "test", sample_endpoints[0], mcp_session=mock_mcp_session,
        )
        assert score == 0.5

    def test_format_endpoints(self, sample_endpoints):
        """Test endpoint formatting for prompts."""
        reranker = LLMReranker()
        text = reranker._format_endpoints(sample_endpoints)

        assert "GET /geo/1.0/direct" in text
        assert "GET /data/2.5/weather" in text
        assert "Geocode a city name" in text

    def test_format_dependencies(self, sample_endpoints):
        """Test dependency formatting for prompts."""
        reranker = LLMReranker()
        text = reranker._format_dependencies(sample_endpoints)

        assert "depends on" in text
        assert "provides for" in text

    def test_extract_json_from_code_block(self):
        """Test JSON extraction from markdown code blocks."""
        reranker = LLMReranker()
        text = '```json\n{"reasoning": "test", "steps": [], "excluded": []}\n```'
        result = reranker._extract_json(text)
        assert result["reasoning"] == "test"

    def test_extract_json_raw(self):
        """Test JSON extraction from raw text."""
        reranker = LLMReranker()
        text = 'Here is the result: {"reasoning": "test", "steps": [], "excluded": []}'
        result = reranker._extract_json(text)
        assert result["reasoning"] == "test"

    def test_extract_json_fallback(self):
        """Test JSON extraction fallback for non-JSON text."""
        reranker = LLMReranker()
        text = "This is not JSON at all"
        result = reranker._extract_json(text)
        assert result["steps"] == []
        assert result["reasoning"] == text


class TestWorkflowStep:
    """Tests for WorkflowStep serialization."""

    def test_to_dict(self):
        from jitapi.retrieval.reranker import ParameterSource
        step = WorkflowStep(
            step_number=1,
            endpoint_id="GET /pets",
            path="/pets",
            method="GET",
            purpose="List pets",
            requires=[],
            provides=["pet_id"],
            relevance_score=0.9,
            parameters={"limit": ParameterSource(value=10, source="literal")},
            output_mapping={"pet_id": "$.data[0].id"},
        )

        d = step.to_dict()
        assert d["step_number"] == 1
        assert d["parameters"]["limit"]["value"] == 10
        assert d["output_mapping"]["pet_id"] == "$.data[0].id"


class TestRerankedWorkflow:
    """Tests for RerankedWorkflow serialization."""

    def test_to_dict(self):
        workflow = RerankedWorkflow(
            query="test",
            steps=[],
            reasoning="No steps needed",
            total_steps=0,
            excluded_endpoints=["GET /unused"],
        )
        d = workflow.to_dict()
        assert d["query"] == "test"
        assert d["excluded_endpoints"] == ["GET /unused"]
