"""Graph expander for dependency resolution.

Takes vector search results and expands them with dependent endpoints
from the dependency graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..stores.graph_store import GraphStore
from ..stores.spec_store import SpecStore
from .vector_search import VectorSearchResult


@dataclass
class ExpandedEndpoint:
    """An endpoint with dependency information."""

    endpoint_id: str
    api_id: str
    path: str
    method: str
    summary: str
    tags: list[str]
    score: float  # Original search score or derived score
    is_dependency: bool  # True if added via graph expansion
    depends_on: list[str]  # Endpoint IDs this depends on
    provides_for: list[str]  # Endpoint IDs that need this
    dependency_params: list[str]  # Parameters that created dependencies
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpansionResult:
    """Result of graph expansion."""

    endpoints: list[ExpandedEndpoint]
    original_count: int
    expanded_count: int
    dependency_edges: list[dict[str, Any]]


class GraphExpander:
    """Expands search results with dependencies from the graph.

    Takes the top-k results from vector search and adds any endpoints
    that are required as dependencies (e.g., if POST /orders needs
    product_id, adds GET /products).
    """

    def __init__(
        self,
        graph_store: GraphStore,
        spec_store: SpecStore,
    ):
        """Initialize the graph expander.

        Args:
            graph_store: The graph store containing dependency graphs.
            spec_store: The spec store for endpoint details.
        """
        self.graph_store = graph_store
        self.spec_store = spec_store

    def expand(
        self,
        search_results: list[VectorSearchResult],
        api_id: str,
        max_depth: int = 2,
        max_total: int = 10,
    ) -> ExpansionResult:
        """Expand search results with dependencies.

        Args:
            search_results: Initial vector search results.
            api_id: The API to expand within.
            max_depth: Maximum depth of dependency traversal.
            max_total: Maximum total endpoints to return.

        Returns:
            ExpansionResult with expanded endpoints.
        """
        if not search_results:
            return ExpansionResult(
                endpoints=[],
                original_count=0,
                expanded_count=0,
                dependency_edges=[],
            )

        # Track all endpoints and their sources
        endpoint_map: dict[str, ExpandedEndpoint] = {}
        dependency_edges: list[dict[str, Any]] = []
        original_ids = set()

        # Add original search results
        for result in search_results:
            original_ids.add(result.endpoint_id)
            endpoint_map[result.endpoint_id] = ExpandedEndpoint(
                endpoint_id=result.endpoint_id,
                api_id=result.api_id,
                path=result.path,
                method=result.method,
                summary=result.summary,
                tags=result.tags,
                score=result.score,
                is_dependency=False,
                depends_on=[],
                provides_for=[],
                dependency_params=[],
                metadata=result.metadata,
            )

        # Expand with dependencies
        to_expand = list(original_ids)
        visited = set()

        for depth in range(max_depth):
            if not to_expand or len(endpoint_map) >= max_total:
                break

            next_expand = []

            for endpoint_id in to_expand:
                if endpoint_id in visited:
                    continue
                visited.add(endpoint_id)

                # Get dependencies for this endpoint
                deps = self.graph_store.get_dependencies(api_id, endpoint_id)

                for dep in deps:
                    dep_id = dep["endpoint_id"]
                    dep_param = dep.get("parameter", "")
                    confidence = dep.get("confidence", 0.0)

                    # Record the edge
                    dependency_edges.append(
                        {
                            "from": endpoint_id,
                            "to": dep_id,
                            "parameter": dep_param,
                            "confidence": confidence,
                        }
                    )

                    # Update the consuming endpoint
                    if endpoint_id in endpoint_map:
                        endpoint_map[endpoint_id].depends_on.append(dep_id)
                        if dep_param:
                            endpoint_map[endpoint_id].dependency_params.append(dep_param)

                    # Add dependency if not already present
                    if dep_id not in endpoint_map:
                        # Get endpoint details from spec store
                        endpoint_details = self.spec_store.get_endpoint(api_id, dep_id)

                        if endpoint_details:
                            # Calculate a derived score
                            # Dependencies get a slightly lower score than what they support
                            base_score = endpoint_map.get(endpoint_id, ExpandedEndpoint(
                                endpoint_id="", api_id="", path="", method="",
                                summary="", tags=[], score=0.5, is_dependency=False,
                                depends_on=[], provides_for=[], dependency_params=[]
                            )).score
                            derived_score = base_score * 0.8 * confidence

                            endpoint_map[dep_id] = ExpandedEndpoint(
                                endpoint_id=dep_id,
                                api_id=api_id,
                                path=endpoint_details.path,
                                method=endpoint_details.method,
                                summary=endpoint_details.summary,
                                tags=endpoint_details.tags,
                                score=derived_score,
                                is_dependency=True,
                                depends_on=[],
                                provides_for=[endpoint_id],
                                dependency_params=[dep_param] if dep_param else [],
                                metadata={},
                            )

                            next_expand.append(dep_id)
                    else:
                        # Update provides_for for existing endpoint
                        if endpoint_id not in endpoint_map[dep_id].provides_for:
                            endpoint_map[dep_id].provides_for.append(endpoint_id)

            to_expand = next_expand

        # Convert to list and sort
        endpoints = list(endpoint_map.values())

        # Sort: original results first (by score), then dependencies
        endpoints.sort(key=lambda x: (x.is_dependency, -x.score))

        # Limit total
        endpoints = endpoints[:max_total]

        return ExpansionResult(
            endpoints=endpoints,
            original_count=len(original_ids),
            expanded_count=len(endpoints) - len(original_ids),
            dependency_edges=dependency_edges,
        )

    def expand_single(
        self,
        endpoint_id: str,
        api_id: str,
        max_depth: int = 3,
    ) -> list[str]:
        """Get the full dependency chain for a single endpoint.

        Args:
            endpoint_id: The endpoint to expand.
            api_id: The API identifier.
            max_depth: Maximum traversal depth.

        Returns:
            List of endpoint IDs in dependency order (dependencies first).
        """
        return self.graph_store.get_full_dependency_chain(
            api_id, endpoint_id, max_depth
        )

    def find_providing_endpoints(
        self,
        api_id: str,
        param_name: str,
    ) -> list[dict[str, Any]]:
        """Find endpoints that can provide a specific parameter.

        Args:
            api_id: The API identifier.
            param_name: The parameter name to find providers for.

        Returns:
            List of provider information dicts.
        """
        providers = self.graph_store.find_providers_for_param(api_id, param_name)

        # Enhance with endpoint details
        enhanced = []
        for provider in providers:
            endpoint = self.spec_store.get_endpoint(api_id, provider["endpoint_id"])
            if endpoint:
                enhanced.append(
                    {
                        "endpoint_id": provider["endpoint_id"],
                        "path": endpoint.path,
                        "method": endpoint.method,
                        "summary": endpoint.summary,
                        "confidence": provider["confidence"],
                    }
                )

        return enhanced

    def get_reverse_dependencies(
        self,
        endpoint_id: str,
        api_id: str,
    ) -> list[dict[str, Any]]:
        """Find endpoints that depend on a given endpoint.

        Args:
            endpoint_id: The provider endpoint.
            api_id: The API identifier.

        Returns:
            List of dependent endpoint information.
        """
        dependents = self.graph_store.get_dependents(api_id, endpoint_id)

        enhanced = []
        for dep in dependents:
            endpoint = self.spec_store.get_endpoint(api_id, dep["endpoint_id"])
            if endpoint:
                enhanced.append(
                    {
                        "endpoint_id": dep["endpoint_id"],
                        "path": endpoint.path,
                        "method": endpoint.method,
                        "summary": endpoint.summary,
                        "parameter": dep.get("parameter", ""),
                        "confidence": dep.get("confidence", 0.0),
                    }
                )

        return enhanced
