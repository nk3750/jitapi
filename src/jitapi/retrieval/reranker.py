"""LLM-based reranker for endpoint relevance scoring.

Uses MCP sampling (delegates to host LLM) by default, with optional
OpenAI fallback if sampling is unavailable and OPENAI_API_KEY is set.

Responsibilities:
1. Score endpoint relevance to user query
2. Determine logical execution order
3. Extract parameters from user query
4. Map output data flow between steps
5. Generate a structured, executable workflow
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .graph_expander import ExpandedEndpoint

logger = logging.getLogger(__name__)


@dataclass
class ParameterSource:
    """Describes where a parameter value comes from."""

    value: Any  # The actual value (if from user_query) or None (if from previous step)
    source: str  # "user_query", "step_N.field", or "literal"

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "source": self.source}


@dataclass
class WorkflowStep:
    """A single step in an API workflow with full parameter information."""

    step_number: int
    endpoint_id: str
    path: str
    method: str
    purpose: str  # Why this step is needed
    requires: list[str]  # What this step needs from previous steps (legacy, for backward compat)
    provides: list[str]  # What this step provides for later steps (legacy, for backward compat)
    relevance_score: float
    # New fields for generic execution
    parameters: dict[str, ParameterSource] = field(default_factory=dict)  # param_name -> source
    output_mapping: dict[str, str] = field(default_factory=dict)  # output_name -> JSONPath expression

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_number": self.step_number,
            "endpoint_id": self.endpoint_id,
            "path": self.path,
            "method": self.method,
            "purpose": self.purpose,
            "requires": self.requires,
            "provides": self.provides,
            "relevance_score": self.relevance_score,
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
            "output_mapping": self.output_mapping,
        }


@dataclass
class RerankedWorkflow:
    """The complete reranked and ordered workflow."""

    query: str
    steps: list[WorkflowStep]
    reasoning: str  # LLM's explanation of the workflow
    total_steps: int
    excluded_endpoints: list[str]  # Endpoints deemed irrelevant

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "reasoning": self.reasoning,
            "total_steps": self.total_steps,
            "excluded_endpoints": self.excluded_endpoints,
        }


class LLMReranker:
    """Reranks and orders endpoints using LLM-powered workflow planning.

    Uses MCP sampling (host LLM) by default — zero extra API keys.
    Falls back to OpenAI if MCP sampling is unavailable and OPENAI_API_KEY is set.
    """

    OPENAI_MODEL = "gpt-4o-mini"

    SYSTEM_PROMPT = """You are an API workflow planner. Given a user query and a list of API endpoints, your task is to:

1. Evaluate which endpoints are actually needed to fulfill the user's request
2. Order them in the correct execution sequence (dependencies first)
3. EXTRACT parameter values from the user's query
4. MAP data flow between steps using JSONPath expressions
5. Explain what each step does and why it's needed

CRITICAL: Extract actual values from the user query for parameters. For example:
- "weather in Tokyo" → extract "Tokyo" as the city/location parameter
- "create a pet named Max" → extract "Max" as the name parameter
- "get user with id 123" → extract 123 as the id parameter

Instructions:
- Only include endpoints that are directly needed for the user's request
- Order steps so that dependencies are fulfilled (e.g., if GET /weather needs lat/lon, GET /geocode must come first)
- For EACH parameter in EACH step, specify:
  - If the value comes from the user query: {"value": "actual_value", "source": "user_query"}
  - If the value comes from a previous step: {"value": null, "source": "step_N.field_name"}
  - For literal/default values: {"value": "the_value", "source": "literal"}
- For steps that provide data to later steps, include output_mapping with JSONPath expressions
  - Use JSONPath syntax like "$.field", "$[0].field", "$.data.items[0].id"
- Assign a relevance score (0.0-1.0) to each included endpoint
- Exclude endpoints that aren't needed

Respond in JSON format only, no other text:
{
  "reasoning": "Brief explanation of your workflow design",
  "steps": [
    {
      "endpoint_id": "GET /geo/1.0/direct",
      "purpose": "Get coordinates for the city",
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
      "purpose": "Get current weather at coordinates",
      "requires": ["lat", "lon"],
      "provides": ["weather_data"],
      "relevance_score": 1.0,
      "parameters": {
        "lat": {"value": null, "source": "step_1.lat"},
        "lon": {"value": null, "source": "step_1.lon"}
      },
      "output_mapping": {}
    }
  ],
  "excluded": ["endpoint_id1"]
}"""

    USER_PROMPT_TEMPLATE = """User Query: {query}

Available Endpoints:
{endpoints}

Dependency Information:
{dependencies}"""

    def __init__(
        self,
        api_key: str | None = None,
    ):
        """Initialize the reranker.

        Args:
            api_key: Optional OpenAI API key for fallback. If None, uses env var.
        """
        self._openai_client = None
        self._openai_api_key = api_key

        # Lazily initialize OpenAI client only if needed
        openai_key = api_key or os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=openai_key)
                logger.debug("OpenAI fallback initialized for reranker")
            except ImportError:
                logger.debug("openai package not installed, MCP sampling only")

    async def rerank(
        self,
        query: str,
        endpoints: list[ExpandedEndpoint],
        max_steps: int = 5,
        mcp_session: Any = None,
    ) -> RerankedWorkflow:
        """Rerank and order endpoints into a workflow.

        Args:
            query: The original user query.
            endpoints: Expanded endpoints from graph expansion.
            max_steps: Maximum steps in the workflow.
            mcp_session: MCP server session for sampling (preferred).

        Returns:
            RerankedWorkflow with ordered steps.
        """
        if not endpoints:
            return RerankedWorkflow(
                query=query,
                steps=[],
                reasoning="No endpoints provided.",
                total_steps=0,
                excluded_endpoints=[],
            )

        # Format endpoints for the prompt
        endpoints_text = self._format_endpoints(endpoints)
        dependencies_text = self._format_dependencies(endpoints)

        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            query=query,
            endpoints=endpoints_text,
            dependencies=dependencies_text,
        )

        # Try MCP sampling first, then OpenAI fallback
        response_text = await self._get_llm_response(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1500,
            mcp_session=mcp_session,
        )

        # Parse response
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            result = self._extract_json(response_text)

        # Build endpoint lookup
        endpoint_lookup = {ep.endpoint_id: ep for ep in endpoints}

        # Convert to WorkflowSteps
        steps = []
        for i, step_data in enumerate(result.get("steps", [])[:max_steps]):
            endpoint_id = step_data.get("endpoint_id", "")
            endpoint = endpoint_lookup.get(endpoint_id)

            if endpoint:
                # Parse parameters
                parameters = {}
                for param_name, param_info in step_data.get("parameters", {}).items():
                    if isinstance(param_info, dict):
                        parameters[param_name] = ParameterSource(
                            value=param_info.get("value"),
                            source=param_info.get("source", "literal"),
                        )
                    else:
                        parameters[param_name] = ParameterSource(
                            value=param_info,
                            source="literal",
                        )

                output_mapping = step_data.get("output_mapping", {})

                steps.append(
                    WorkflowStep(
                        step_number=i + 1,
                        endpoint_id=endpoint_id,
                        path=endpoint.path,
                        method=endpoint.method,
                        purpose=step_data.get("purpose", ""),
                        requires=step_data.get("requires", []),
                        provides=step_data.get("provides", []),
                        relevance_score=step_data.get("relevance_score", 0.5),
                        parameters=parameters,
                        output_mapping=output_mapping,
                    )
                )

        return RerankedWorkflow(
            query=query,
            steps=steps,
            reasoning=result.get("reasoning", ""),
            total_steps=len(steps),
            excluded_endpoints=result.get("excluded", []),
        )

    async def score_relevance(
        self,
        query: str,
        endpoint: ExpandedEndpoint,
        mcp_session: Any = None,
    ) -> float:
        """Score a single endpoint's relevance to a query.

        Args:
            query: The user query.
            endpoint: The endpoint to score.
            mcp_session: MCP server session for sampling.

        Returns:
            Relevance score from 0.0 to 1.0.
        """
        prompt = f"""Rate the relevance of this API endpoint to the user's query.

User Query: {query}

Endpoint:
- ID: {endpoint.endpoint_id}
- Path: {endpoint.path}
- Method: {endpoint.method}
- Summary: {endpoint.summary}
- Tags: {', '.join(endpoint.tags)}

Respond with only a number from 0.0 to 1.0, where:
- 0.0 = completely irrelevant
- 0.5 = somewhat related
- 1.0 = exactly what the user needs

Score:"""

        response_text = await self._get_llm_response(
            system_prompt=None,
            user_prompt=prompt,
            max_tokens=10,
            mcp_session=mcp_session,
        )

        try:
            score = float(response_text.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5

    async def _get_llm_response(
        self,
        system_prompt: str | None,
        user_prompt: str,
        max_tokens: int,
        mcp_session: Any = None,
    ) -> str:
        """Get an LLM response using MCP sampling or OpenAI fallback.

        Args:
            system_prompt: Optional system prompt.
            user_prompt: The user prompt.
            max_tokens: Max tokens in response.
            mcp_session: MCP server session for sampling.

        Returns:
            The LLM response text.

        Raises:
            RuntimeError: If neither MCP sampling nor OpenAI is available.
        """
        # Try MCP sampling first
        if mcp_session is not None:
            try:
                return await self._call_mcp_sampling(
                    mcp_session, system_prompt, user_prompt, max_tokens
                )
            except Exception as e:
                logger.warning(f"MCP sampling failed, trying OpenAI fallback: {e}")

        # Fall back to OpenAI
        if self._openai_client is not None:
            return self._call_openai(system_prompt, user_prompt, max_tokens)

        raise RuntimeError(
            "No LLM backend available for workflow planning. Either:\n"
            "1. Use a client that supports MCP sampling (e.g., Claude Desktop, Claude Code)\n"
            "2. Set OPENAI_API_KEY for OpenAI fallback\n"
            "3. Install openai: pip install jitapi[openai]"
        )

    async def _call_mcp_sampling(
        self,
        session: Any,
        system_prompt: str | None,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        """Call MCP sampling to get an LLM response from the host.

        Args:
            session: The MCP ServerSession.
            system_prompt: Optional system prompt.
            user_prompt: The user prompt.
            max_tokens: Max tokens in response.

        Returns:
            The response text.
        """
        from mcp.types import SamplingMessage, TextContent

        messages = [
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=user_prompt),
            )
        ]

        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            kwargs["system_prompt"] = system_prompt

        result = await session.create_message(**kwargs)

        # Extract text from response
        if hasattr(result.content, "text"):
            return result.content.text
        return str(result.content)

    def _call_openai(
        self,
        system_prompt: str | None,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        """Call OpenAI API as fallback.

        Args:
            system_prompt: Optional system prompt.
            user_prompt: The user prompt.
            max_tokens: Max tokens in response.

        Returns:
            The response text.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self._openai_client.chat.completions.create(
            model=self.OPENAI_MODEL,
            max_tokens=max_tokens,
            messages=messages,
        )

        return response.choices[0].message.content

    def _format_endpoints(self, endpoints: list[ExpandedEndpoint]) -> str:
        """Format endpoints for the prompt."""
        lines = []
        for ep in endpoints:
            lines.append(
                f"- {ep.endpoint_id}\n"
                f"  Path: {ep.path}\n"
                f"  Method: {ep.method}\n"
                f"  Summary: {ep.summary}\n"
                f"  Tags: {', '.join(ep.tags) if ep.tags else 'none'}\n"
                f"  Search Score: {ep.score:.2f}\n"
                f"  Is Dependency: {ep.is_dependency}"
            )
        return "\n".join(lines)

    def _format_dependencies(self, endpoints: list[ExpandedEndpoint]) -> str:
        """Format dependency information for the prompt."""
        lines = []
        for ep in endpoints:
            if ep.depends_on:
                lines.append(
                    f"- {ep.endpoint_id} depends on: {', '.join(ep.depends_on)}"
                )
                if ep.dependency_params:
                    lines.append(
                        f"  (needs: {', '.join(ep.dependency_params)})"
                    )
            if ep.provides_for:
                lines.append(
                    f"- {ep.endpoint_id} provides for: {', '.join(ep.provides_for)}"
                )

        if not lines:
            return "No explicit dependencies detected."

        return "\n".join(lines)

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Try to extract JSON from a text response."""
        import re

        # Try to find JSON in code blocks
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Return empty structure
        return {"reasoning": text, "steps": [], "excluded": []}
