"""Dependency graph builder for API endpoints.

Analyzes OpenAPI specs to build a graph showing which endpoints depend on
which other endpoints (e.g., POST /orders needs product_id from GET /products).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import networkx as nx

from .parser import Endpoint, ParsedSpec


@dataclass
class DependencyEdge:
    """Represents a dependency between two endpoints."""

    source_endpoint: str  # endpoint that needs something
    target_endpoint: str  # endpoint that provides something
    parameter_name: str  # what is needed (e.g., "product_id")
    dependency_type: str  # "path_param", "body_param", "query_param", "schema_ref"
    confidence: float  # 0.0 to 1.0


@dataclass
class ParameterProvider:
    """Information about an endpoint that provides a parameter value."""

    endpoint_id: str
    field_name: str
    confidence: float


class DependencyGraphBuilder:
    """Builds a dependency graph from parsed OpenAPI specifications.

    The graph captures which endpoints depend on which other endpoints:
    - POST /orders needs product_id → GET /products provides id
    - GET /users/{user_id}/orders needs user_id → GET /users provides id
    """

    # Common ID field patterns (case-insensitive matching)
    ID_PATTERNS = [
        r"^id$",
        r"^_id$",
        r".*_id$",
        r".*id$",  # Matches camelCase like petId, userId
        r"^uuid$",
        r".*_uuid$",
        r".*uuid$",
        r".*_key$",
        r".*key$",
        r"^key$",
    ]

    # Common parameter name to entity mappings
    ENTITY_MAPPINGS = {
        "user_id": ["user", "users", "account", "accounts"],
        "product_id": ["product", "products", "item", "items"],
        "order_id": ["order", "orders"],
        "customer_id": ["customer", "customers"],
        "cart_id": ["cart", "carts", "shopping_cart"],
        "location_id": ["location", "locations", "place", "places"],
        "location_key": ["location", "locations", "city", "cities"],
        "locationKey": ["location", "locations", "city", "cities"],
    }

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self._endpoint_map: dict[str, Endpoint] = {}
        self._tag_endpoints: dict[str, list[str]] = {}
        self._returned_fields_index: dict[str, list[str]] = {}  # field -> [endpoint_ids]

    def build(self, parsed_spec: ParsedSpec) -> nx.DiGraph:
        """Build the dependency graph for a parsed API spec.

        Args:
            parsed_spec: The parsed OpenAPI specification.

        Returns:
            A NetworkX directed graph where nodes are endpoint IDs and
            edges represent dependencies.
        """
        self.graph = nx.DiGraph()
        self._endpoint_map = {ep.endpoint_id: ep for ep in parsed_spec.endpoints}
        self._build_indices(parsed_spec)

        # Add all endpoints as nodes
        for endpoint in parsed_spec.endpoints:
            self.graph.add_node(
                endpoint.endpoint_id,
                path=endpoint.path,
                method=endpoint.method,
                summary=endpoint.summary,
                tags=endpoint.tags,
            )

        # Analyze dependencies for each endpoint
        for endpoint in parsed_spec.endpoints:
            self._analyze_endpoint_dependencies(endpoint, parsed_spec)

        return self.graph

    def _build_indices(self, parsed_spec: ParsedSpec) -> None:
        """Build indices for fast lookups."""
        # Index endpoints by tag
        self._tag_endpoints = {}
        for endpoint in parsed_spec.endpoints:
            for tag in endpoint.tags:
                tag_lower = tag.lower()
                if tag_lower not in self._tag_endpoints:
                    self._tag_endpoints[tag_lower] = []
                self._tag_endpoints[tag_lower].append(endpoint.endpoint_id)

        # Index endpoints by returned fields
        self._returned_fields_index = {}
        for endpoint in parsed_spec.endpoints:
            for field in endpoint.returned_fields:
                field_lower = field.lower()
                if field_lower not in self._returned_fields_index:
                    self._returned_fields_index[field_lower] = []
                self._returned_fields_index[field_lower].append(endpoint.endpoint_id)

    def _analyze_endpoint_dependencies(
        self, endpoint: Endpoint, parsed_spec: ParsedSpec
    ) -> None:
        """Analyze an endpoint's parameters to find dependencies."""
        # Check path parameters
        for param in endpoint.parameters:
            if param.location == "path":
                providers = self._find_parameter_providers(
                    param.name, endpoint, parsed_spec, "path_param"
                )
                for provider in providers:
                    self._add_dependency(
                        endpoint.endpoint_id,
                        provider.endpoint_id,
                        param.name,
                        "path_param",
                        provider.confidence,
                    )

            elif param.location == "query" and param.required:
                # Only analyze required query params that look like IDs
                if self._is_id_like(param.name):
                    providers = self._find_parameter_providers(
                        param.name, endpoint, parsed_spec, "query_param"
                    )
                    for provider in providers:
                        self._add_dependency(
                            endpoint.endpoint_id,
                            provider.endpoint_id,
                            param.name,
                            "query_param",
                            provider.confidence,
                        )

        # Check request body for ID-like fields
        if endpoint.request_body and endpoint.request_body.schema:
            body_schema = endpoint.request_body.schema
            self._analyze_schema_dependencies(
                endpoint, body_schema, parsed_spec, "body_param"
            )

    def _find_parameter_providers(
        self,
        param_name: str,
        consuming_endpoint: Endpoint,
        parsed_spec: ParsedSpec,
        dep_type: str,
    ) -> list[ParameterProvider]:
        """Find endpoints that can provide a given parameter value."""
        providers = []
        param_lower = param_name.lower()

        # Strategy 1: Direct field name match in returned_fields_index
        # Look for 'id' if param is something_id
        search_fields = [param_lower]
        if param_lower.endswith("_id"):
            search_fields.append("id")
            search_fields.append(param_lower[:-3])  # entity name
        elif param_lower.endswith("id"):
            search_fields.append("id")
        if param_lower.endswith("_key"):
            search_fields.append("key")
            search_fields.append(param_lower[:-4])
        elif param_lower.endswith("key"):
            search_fields.append("key")

        for search_field in search_fields:
            if search_field in self._returned_fields_index:
                for endpoint_id in self._returned_fields_index[search_field]:
                    # Don't self-reference
                    if endpoint_id == consuming_endpoint.endpoint_id:
                        continue
                    # Prefer GET/POST endpoints as providers
                    ep = self._endpoint_map.get(endpoint_id)
                    if ep and ep.method in ["GET", "POST"]:
                        confidence = 0.8 if search_field == param_lower else 0.6
                        providers.append(
                            ParameterProvider(endpoint_id, search_field, confidence)
                        )

        # Strategy 2: Entity mapping (e.g., user_id -> users endpoints)
        entity_names = self._get_entity_names_for_param(param_name)
        for entity in entity_names:
            # Look for endpoints with this entity in the path
            for ep_id, ep in self._endpoint_map.items():
                if ep_id == consuming_endpoint.endpoint_id:
                    continue

                # Check if endpoint path contains the entity
                path_lower = ep.path.lower()
                if f"/{entity}" in path_lower or path_lower.endswith(f"/{entity}"):
                    # Prefer list/create endpoints as providers
                    if ep.method in ["GET", "POST"]:
                        # Check if it returns id-like fields
                        if any(self._is_id_like(f) for f in ep.returned_fields):
                            if not self._already_has_provider(providers, ep_id):
                                providers.append(
                                    ParameterProvider(ep_id, "id", 0.7)
                                )

        # Strategy 3: Tag-based matching
        entity_names = self._get_entity_names_for_param(param_name)
        for entity in entity_names:
            if entity in self._tag_endpoints:
                for ep_id in self._tag_endpoints[entity]:
                    if ep_id == consuming_endpoint.endpoint_id:
                        continue
                    ep = self._endpoint_map.get(ep_id)
                    if ep and ep.method in ["GET", "POST"]:
                        if not self._already_has_provider(providers, ep_id):
                            providers.append(ParameterProvider(ep_id, "id", 0.5))

        # Deduplicate and sort by confidence
        seen = set()
        unique_providers = []
        for p in sorted(providers, key=lambda x: -x.confidence):
            if p.endpoint_id not in seen:
                seen.add(p.endpoint_id)
                unique_providers.append(p)

        return unique_providers[:3]  # Return top 3 providers

    def _analyze_schema_dependencies(
        self,
        endpoint: Endpoint,
        schema: dict[str, Any],
        parsed_spec: ParsedSpec,
        dep_type: str,
    ) -> None:
        """Analyze a schema for ID-like fields that might create dependencies."""
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])

        for field_name, field_schema in properties.items():
            # Only look at required fields or fields that look like IDs
            if field_name not in required_fields and not self._is_id_like(field_name):
                continue

            if self._is_id_like(field_name):
                providers = self._find_parameter_providers(
                    field_name, endpoint, parsed_spec, dep_type
                )
                for provider in providers:
                    self._add_dependency(
                        endpoint.endpoint_id,
                        provider.endpoint_id,
                        field_name,
                        dep_type,
                        provider.confidence,
                    )

    def _add_dependency(
        self,
        source: str,
        target: str,
        param_name: str,
        dep_type: str,
        confidence: float,
    ) -> None:
        """Add a dependency edge to the graph."""
        # Avoid self-loops
        if source == target:
            return

        # Check if edge already exists
        if self.graph.has_edge(source, target):
            # Update if higher confidence
            existing = self.graph.edges[source, target]
            if confidence > existing.get("confidence", 0):
                existing["confidence"] = confidence
                existing["parameter"] = param_name
                existing["type"] = dep_type
        else:
            self.graph.add_edge(
                source,
                target,
                parameter=param_name,
                type=dep_type,
                confidence=confidence,
            )

    def _is_id_like(self, field_name: str) -> bool:
        """Check if a field name looks like an ID field."""
        field_lower = field_name.lower()
        for pattern in self.ID_PATTERNS:
            if re.match(pattern, field_lower):
                return True
        return False

    def _get_entity_names_for_param(self, param_name: str) -> list[str]:
        """Get entity names that might provide this parameter."""
        param_lower = param_name.lower()

        # Check explicit mappings
        if param_lower in self.ENTITY_MAPPINGS:
            return self.ENTITY_MAPPINGS[param_lower]

        # Extract entity from param name
        # e.g., "user_id" -> ["user", "users"]
        # e.g., "productId" -> ["product", "products"]
        entities = []

        if "_" in param_lower:
            # snake_case: user_id -> user
            parts = param_lower.split("_")
            if parts[-1] in ["id", "key", "uuid"]:
                entity = "_".join(parts[:-1])
                entities.extend([entity, entity + "s"])
        else:
            # camelCase: productId -> product
            # Find where lowercase ends and Id/Key starts
            match = re.match(r"(.+?)(id|key|uuid)$", param_lower, re.IGNORECASE)
            if match:
                entity = match.group(1).lower()
                entities.extend([entity, entity + "s"])

        return entities

    def _already_has_provider(
        self, providers: list[ParameterProvider], endpoint_id: str
    ) -> bool:
        """Check if we already have this endpoint as a provider."""
        return any(p.endpoint_id == endpoint_id for p in providers)

    def get_dependencies(self, endpoint_id: str) -> list[DependencyEdge]:
        """Get all dependencies for an endpoint.

        Args:
            endpoint_id: The endpoint to get dependencies for.

        Returns:
            List of dependency edges (what this endpoint needs from others).
        """
        if endpoint_id not in self.graph:
            return []

        edges = []
        for _, target, data in self.graph.out_edges(endpoint_id, data=True):
            edges.append(
                DependencyEdge(
                    source_endpoint=endpoint_id,
                    target_endpoint=target,
                    parameter_name=data.get("parameter", ""),
                    dependency_type=data.get("type", ""),
                    confidence=data.get("confidence", 0.0),
                )
            )

        return edges

    def get_providers(self, endpoint_id: str) -> list[DependencyEdge]:
        """Get all endpoints that depend on this endpoint.

        Args:
            endpoint_id: The endpoint to check.

        Returns:
            List of dependency edges (what other endpoints need from this one).
        """
        if endpoint_id not in self.graph:
            return []

        edges = []
        for source, _, data in self.graph.in_edges(endpoint_id, data=True):
            edges.append(
                DependencyEdge(
                    source_endpoint=source,
                    target_endpoint=endpoint_id,
                    parameter_name=data.get("parameter", ""),
                    dependency_type=data.get("type", ""),
                    confidence=data.get("confidence", 0.0),
                )
            )

        return edges

    def get_full_dependency_chain(
        self, endpoint_id: str, max_depth: int = 3
    ) -> list[str]:
        """Get the full dependency chain for an endpoint.

        Traverses the graph to find all endpoints needed to call
        the given endpoint.

        Args:
            endpoint_id: The endpoint to analyze.
            max_depth: Maximum depth to traverse (prevents cycles).

        Returns:
            List of endpoint IDs in dependency order (dependencies first).
        """
        if endpoint_id not in self.graph:
            return [endpoint_id]

        visited = set()
        chain = []

        def traverse(ep_id: str, depth: int):
            if depth > max_depth or ep_id in visited:
                return
            visited.add(ep_id)

            # First, get this endpoint's dependencies
            for _, target, _ in self.graph.out_edges(ep_id, data=True):
                traverse(target, depth + 1)

            # Then add this endpoint
            chain.append(ep_id)

        traverse(endpoint_id, 0)
        return chain
