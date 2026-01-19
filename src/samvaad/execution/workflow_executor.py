"""Generic workflow executor for API orchestration.

Executes multi-step API workflows by:
1. Reading workflow JSON from reranker
2. Substituting parameters from user query or previous step results
3. Using JSONPath expressions to extract data from responses
4. Making HTTP calls in sequence
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..retrieval.reranker import RerankedWorkflow, WorkflowStep
from .http_executor import HTTPExecutor, APICallResult

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of executing a single workflow step."""

    step_number: int
    endpoint_id: str
    success: bool
    response: Any  # Parsed response data
    extracted_data: dict[str, Any]  # Data extracted via output_mapping
    error: str | None = None
    status_code: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "endpoint_id": self.endpoint_id,
            "success": self.success,
            "response": self.response,
            "extracted_data": self.extracted_data,
            "error": self.error,
            "status_code": self.status_code,
        }


@dataclass
class WorkflowResult:
    """Result of executing an entire workflow."""

    success: bool
    steps: list[StepResult]
    final_result: Any  # The result of the last step
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "steps": [s.to_dict() for s in self.steps],
            "final_result": self.final_result,
            "error": self.error,
        }


class WorkflowExecutor:
    """Executes API workflows with parameter substitution and data flow.

    Takes a RerankedWorkflow and executes each step, substituting
    parameters from the user query or from previous step results
    using JSONPath expressions.
    """

    def __init__(
        self,
        http_executor: HTTPExecutor,
        spec_store: Any,  # SpecStore - using Any to avoid circular import
    ):
        """Initialize the workflow executor.

        Args:
            http_executor: HTTP executor for making API calls.
            spec_store: Spec store for endpoint information.
        """
        self.http_executor = http_executor
        self.spec_store = spec_store

    async def execute(
        self,
        workflow: RerankedWorkflow,
        api_id: str,
        additional_params: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Execute a workflow.

        Args:
            workflow: The workflow to execute (from reranker).
            api_id: The API identifier.
            additional_params: Additional parameters to merge (e.g., API keys).

        Returns:
            WorkflowResult with all step results.
        """
        if not workflow.steps:
            return WorkflowResult(
                success=False,
                steps=[],
                final_result=None,
                error="No steps in workflow",
            )

        step_results: list[StepResult] = []
        step_data: dict[int, dict[str, Any]] = {}  # step_number -> extracted data

        for step in workflow.steps:
            logger.info(f"Executing step {step.step_number}: {step.endpoint_id}")

            # Resolve parameters for this step
            resolved_params = self._resolve_parameters(step, step_data)

            # Get endpoint info
            endpoint = self.spec_store.get_endpoint(api_id, step.endpoint_id)
            if not endpoint:
                step_result = StepResult(
                    step_number=step.step_number,
                    endpoint_id=step.endpoint_id,
                    success=False,
                    response=None,
                    extracted_data={},
                    error=f"Endpoint not found: {step.endpoint_id}",
                )
                step_results.append(step_result)
                return WorkflowResult(
                    success=False,
                    steps=step_results,
                    final_result=None,
                    error=f"Endpoint not found: {step.endpoint_id}",
                )

            # Merge additional params
            if additional_params:
                resolved_params = {**additional_params, **resolved_params}

            # Separate params by type (path, query, body)
            path_params, query_params, body = self._categorize_params(
                resolved_params, endpoint
            )

            # Execute the API call
            try:
                call_result = await self.http_executor.call_endpoint(
                    endpoint=endpoint,
                    api_id=api_id,
                    path_params=path_params,
                    query_params=query_params,
                    body=body if body else None,
                )

                # Extract data using output_mapping
                extracted = self._extract_output_data(
                    call_result.body, step.output_mapping
                )
                step_data[step.step_number] = extracted

                step_result = StepResult(
                    step_number=step.step_number,
                    endpoint_id=step.endpoint_id,
                    success=call_result.success,
                    response=call_result.body,
                    extracted_data=extracted,
                    error=call_result.error_message,
                    status_code=call_result.status_code,
                )
                step_results.append(step_result)

                if not call_result.success:
                    logger.error(
                        f"Step {step.step_number} failed: {call_result.error_message}"
                    )
                    return WorkflowResult(
                        success=False,
                        steps=step_results,
                        final_result=call_result.body,
                        error=f"Step {step.step_number} failed: {call_result.error_message}",
                    )

                logger.info(
                    f"Step {step.step_number} completed, extracted: {list(extracted.keys())}"
                )

            except Exception as e:
                logger.exception(f"Error executing step {step.step_number}")
                step_result = StepResult(
                    step_number=step.step_number,
                    endpoint_id=step.endpoint_id,
                    success=False,
                    response=None,
                    extracted_data={},
                    error=str(e),
                )
                step_results.append(step_result)
                return WorkflowResult(
                    success=False,
                    steps=step_results,
                    final_result=None,
                    error=str(e),
                )

        # All steps completed successfully
        final_result = step_results[-1].response if step_results else None
        return WorkflowResult(
            success=True,
            steps=step_results,
            final_result=final_result,
        )

    def _resolve_parameters(
        self,
        step: WorkflowStep,
        previous_step_data: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        """Resolve parameter values for a step.

        Args:
            step: The workflow step.
            previous_step_data: Data extracted from previous steps.

        Returns:
            Dict of parameter name -> resolved value.
        """
        resolved = {}

        for param_name, param_source in step.parameters.items():
            value = None

            if param_source.source == "user_query":
                # Value is directly from the user query (already extracted by LLM)
                value = param_source.value

            elif param_source.source == "literal":
                # Literal value
                value = param_source.value

            elif param_source.source.startswith("step_"):
                # Value comes from a previous step
                # Format: "step_N.field_name"
                match = re.match(r"step_(\d+)\.(.+)", param_source.source)
                if match:
                    step_num = int(match.group(1))
                    field_name = match.group(2)

                    if step_num in previous_step_data:
                        value = previous_step_data[step_num].get(field_name)
                        if value is None:
                            logger.warning(
                                f"Field '{field_name}' not found in step {step_num} data"
                            )
                    else:
                        logger.warning(f"Step {step_num} data not found")

            if value is not None:
                resolved[param_name] = value
            else:
                logger.warning(
                    f"Could not resolve parameter '{param_name}' from source '{param_source.source}'"
                )

        return resolved

    def _extract_output_data(
        self,
        response: Any,
        output_mapping: dict[str, str],
    ) -> dict[str, Any]:
        """Extract data from response using JSONPath expressions.

        Args:
            response: The API response.
            output_mapping: Dict of output_name -> JSONPath expression.

        Returns:
            Dict of extracted values.
        """
        if not output_mapping or response is None:
            return {}

        extracted = {}

        for output_name, jsonpath_expr in output_mapping.items():
            try:
                value = self._evaluate_jsonpath(response, jsonpath_expr)
                if value is not None:
                    extracted[output_name] = value
                    logger.debug(f"Extracted {output_name}={value} using {jsonpath_expr}")
            except Exception as e:
                logger.warning(
                    f"Failed to extract '{output_name}' with path '{jsonpath_expr}': {e}"
                )

        return extracted

    def _evaluate_jsonpath(self, data: Any, path: str) -> Any:
        """Evaluate a JSONPath expression against data.

        Implements a simple JSONPath subset:
        - $ - root
        - .field - object field access
        - [N] - array index access
        - [0].field - array index then field

        Args:
            data: The data to query.
            path: JSONPath expression (e.g., "$[0].lat", "$.data.id").

        Returns:
            The extracted value, or None if not found.
        """
        if not path or path == "$":
            return data

        # Remove leading $
        if path.startswith("$"):
            path = path[1:]

        current = data

        # Parse path segments
        # Handles: .field, [N], [N].field
        segments = re.findall(r"\.([^.\[\]]+)|\[(\d+)\]", path)

        for segment in segments:
            field, index = segment

            if field:
                # Object field access
                if isinstance(current, dict):
                    current = current.get(field)
                else:
                    return None

            elif index:
                # Array index access
                idx = int(index)
                if isinstance(current, list) and idx < len(current):
                    current = current[idx]
                else:
                    return None

            if current is None:
                return None

        return current

    def _categorize_params(
        self,
        params: dict[str, Any],
        endpoint: Any,  # Endpoint type
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Categorize parameters into path, query, and body.

        Args:
            params: All resolved parameters.
            endpoint: The endpoint definition.

        Returns:
            Tuple of (path_params, query_params, body_params).
        """
        path_params = {}
        query_params = {}
        body_params = {}

        # Get parameter names from endpoint definition
        path_param_names = set()
        query_param_names = set()

        if hasattr(endpoint, "parameters"):
            for param in endpoint.parameters:
                if param.location == "path":
                    path_param_names.add(param.name)
                elif param.location == "query":
                    query_param_names.add(param.name)

        # Also check for path parameters in the path itself
        path_placeholders = re.findall(r"\{(\w+)\}", endpoint.path)
        path_param_names.update(path_placeholders)

        # Categorize parameters
        for name, value in params.items():
            if name in path_param_names:
                path_params[name] = value
            elif name in query_param_names:
                query_params[name] = value
            else:
                # Assume it's a body parameter
                body_params[name] = value

        return path_params, query_params, body_params


async def execute_workflow_from_dict(
    workflow_dict: dict[str, Any],
    api_id: str,
    http_executor: HTTPExecutor,
    spec_store: Any,
    additional_params: dict[str, Any] | None = None,
) -> WorkflowResult:
    """Execute a workflow from a dictionary representation.

    Convenience function for executing workflows directly from JSON.

    Args:
        workflow_dict: Workflow as a dictionary (from reranker.to_dict()).
        api_id: The API identifier.
        http_executor: HTTP executor.
        spec_store: Spec store.
        additional_params: Additional parameters.

    Returns:
        WorkflowResult.
    """
    from ..retrieval.reranker import (
        RerankedWorkflow,
        WorkflowStep,
        ParameterSource,
    )

    # Parse workflow from dict
    steps = []
    for step_dict in workflow_dict.get("steps", []):
        parameters = {}
        for param_name, param_info in step_dict.get("parameters", {}).items():
            parameters[param_name] = ParameterSource(
                value=param_info.get("value"),
                source=param_info.get("source", "literal"),
            )

        steps.append(
            WorkflowStep(
                step_number=step_dict.get("step_number", 0),
                endpoint_id=step_dict.get("endpoint_id", ""),
                path=step_dict.get("path", ""),
                method=step_dict.get("method", "GET"),
                purpose=step_dict.get("purpose", ""),
                requires=step_dict.get("requires", []),
                provides=step_dict.get("provides", []),
                relevance_score=step_dict.get("relevance_score", 0.5),
                parameters=parameters,
                output_mapping=step_dict.get("output_mapping", {}),
            )
        )

    workflow = RerankedWorkflow(
        query=workflow_dict.get("query", ""),
        steps=steps,
        reasoning=workflow_dict.get("reasoning", ""),
        total_steps=len(steps),
        excluded_endpoints=workflow_dict.get("excluded_endpoints", []),
    )

    executor = WorkflowExecutor(http_executor, spec_store)
    return await executor.execute(workflow, api_id, additional_params)
