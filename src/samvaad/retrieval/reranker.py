"""LLM-based reranker for endpoint relevance scoring.

Uses OpenAI gpt-4o-mini to:
1. Score endpoint relevance to user query
2. Determine logical execution order
3. Extract parameters from user query
4. Map output data flow between steps
5. Generate a structured, executable workflow
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .graph_expander import ExpandedEndpoint


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
    """Reranks and orders endpoints using OpenAI gpt-4o-mini.

    Takes expanded search results and uses an LLM to:
    - Score relevance to the original query
    - Determine the logical order of operations
    - Extract parameters from the user query
    - Map data flow between steps using JSONPath
    - Filter out irrelevant endpoints
    """

    DEFAULT_MODEL = "gpt-4o-mini"

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
        model: str = DEFAULT_MODEL,
    ):
        """Initialize the reranker.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: The model to use for reranking.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def rerank(
        self,
        query: str,
        endpoints: list[ExpandedEndpoint],
        max_steps: int = 5,
    ) -> RerankedWorkflow:
        """Rerank and order endpoints into a workflow.

        Args:
            query: The original user query.
            endpoints: Expanded endpoints from graph expansion.
            max_steps: Maximum steps in the workflow.

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

        # Build the user prompt
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            query=query,
            endpoints=endpoints_text,
            dependencies=dependencies_text,
        )

        # Call OpenAI
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        # Parse response
        response_text = response.choices[0].message.content

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
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
                        # Handle simple value (backward compatibility)
                        parameters[param_name] = ParameterSource(
                            value=param_info,
                            source="literal",
                        )

                # Parse output mapping
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

    def score_relevance(
        self,
        query: str,
        endpoint: ExpandedEndpoint,
    ) -> float:
        """Score a single endpoint's relevance to a query.

        Args:
            query: The user query.
            endpoint: The endpoint to score.

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

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5

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
