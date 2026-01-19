"""Graph store for API dependency graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph


class GraphStore:
    """Store for API endpoint dependency graphs.

    Uses NetworkX graphs serialized to JSON for persistence.
    """

    def __init__(self, storage_dir: str | Path):
        """Initialize the graph store.

        Args:
            storage_dir: Directory to store graph data.
        """
        self.storage_dir = Path(storage_dir)
        self.graphs_dir = self.storage_dir / "graphs"
        self.graphs_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache of loaded graphs
        self._cache: dict[str, nx.DiGraph] = {}

    def store_graph(self, api_id: str, graph: nx.DiGraph) -> None:
        """Store a dependency graph for an API.

        Args:
            api_id: The API identifier.
            graph: The NetworkX directed graph to store.
        """
        graph_file = self.graphs_dir / f"{api_id}.json"

        # Convert to JSON-serializable format
        data = json_graph.node_link_data(graph)

        with open(graph_file, "w") as f:
            json.dump(data, f, indent=2)

        # Update cache
        self._cache[api_id] = graph

    def get_graph(self, api_id: str) -> nx.DiGraph | None:
        """Get the dependency graph for an API.

        Args:
            api_id: The API identifier.

        Returns:
            The NetworkX graph, or None if not found.
        """
        # Check cache first
        if api_id in self._cache:
            return self._cache[api_id]

        graph_file = self.graphs_dir / f"{api_id}.json"
        if not graph_file.exists():
            return None

        with open(graph_file) as f:
            data = json.load(f)

        graph = json_graph.node_link_graph(data, directed=True)

        # Cache for future use
        self._cache[api_id] = graph
        return graph

    def delete_graph(self, api_id: str) -> bool:
        """Delete the graph for an API.

        Args:
            api_id: The API identifier.

        Returns:
            True if deleted, False if not found.
        """
        graph_file = self.graphs_dir / f"{api_id}.json"

        if api_id in self._cache:
            del self._cache[api_id]

        if graph_file.exists():
            graph_file.unlink()
            return True

        return False

    def graph_exists(self, api_id: str) -> bool:
        """Check if a graph exists for an API.

        Args:
            api_id: The API identifier.

        Returns:
            True if the graph exists.
        """
        if api_id in self._cache:
            return True
        graph_file = self.graphs_dir / f"{api_id}.json"
        return graph_file.exists()

    def get_dependencies(self, api_id: str, endpoint_id: str) -> list[dict[str, Any]]:
        """Get dependencies for an endpoint.

        Args:
            api_id: The API identifier.
            endpoint_id: The endpoint identifier.

        Returns:
            List of dependency info dicts with target endpoint and edge data.
        """
        graph = self.get_graph(api_id)
        if graph is None or endpoint_id not in graph:
            return []

        dependencies = []
        for _, target, data in graph.out_edges(endpoint_id, data=True):
            dependencies.append(
                {
                    "endpoint_id": target,
                    "parameter": data.get("parameter", ""),
                    "type": data.get("type", ""),
                    "confidence": data.get("confidence", 0.0),
                }
            )

        return dependencies

    def get_dependents(self, api_id: str, endpoint_id: str) -> list[dict[str, Any]]:
        """Get endpoints that depend on this endpoint.

        Args:
            api_id: The API identifier.
            endpoint_id: The endpoint identifier.

        Returns:
            List of dependent endpoint info dicts.
        """
        graph = self.get_graph(api_id)
        if graph is None or endpoint_id not in graph:
            return []

        dependents = []
        for source, _, data in graph.in_edges(endpoint_id, data=True):
            dependents.append(
                {
                    "endpoint_id": source,
                    "parameter": data.get("parameter", ""),
                    "type": data.get("type", ""),
                    "confidence": data.get("confidence", 0.0),
                }
            )

        return dependents

    def get_endpoint_node(self, api_id: str, endpoint_id: str) -> dict[str, Any] | None:
        """Get node data for an endpoint.

        Args:
            api_id: The API identifier.
            endpoint_id: The endpoint identifier.

        Returns:
            Node data dict, or None if not found.
        """
        graph = self.get_graph(api_id)
        if graph is None or endpoint_id not in graph:
            return None

        return dict(graph.nodes[endpoint_id])

    def find_providers_for_param(
        self, api_id: str, param_name: str
    ) -> list[dict[str, Any]]:
        """Find endpoints that can provide a given parameter.

        Args:
            api_id: The API identifier.
            param_name: The parameter name to find providers for.

        Returns:
            List of provider endpoint info.
        """
        graph = self.get_graph(api_id)
        if graph is None:
            return []

        providers = []
        param_lower = param_name.lower()

        for _, _, data in graph.edges(data=True):
            edge_param = data.get("parameter", "").lower()
            if edge_param == param_lower:
                # The target of this edge provides the parameter
                target = data.get("target")
                if target:
                    providers.append(
                        {
                            "endpoint_id": target,
                            "confidence": data.get("confidence", 0.0),
                        }
                    )

        # Deduplicate
        seen = set()
        unique = []
        for p in sorted(providers, key=lambda x: -x["confidence"]):
            if p["endpoint_id"] not in seen:
                seen.add(p["endpoint_id"])
                unique.append(p)

        return unique

    def get_full_dependency_chain(
        self, api_id: str, endpoint_id: str, max_depth: int = 3
    ) -> list[str]:
        """Get the full dependency chain for an endpoint.

        Args:
            api_id: The API identifier.
            endpoint_id: The endpoint to analyze.
            max_depth: Maximum traversal depth.

        Returns:
            List of endpoint IDs in dependency order (dependencies first).
        """
        graph = self.get_graph(api_id)
        if graph is None or endpoint_id not in graph:
            return [endpoint_id]

        visited = set()
        chain = []

        def traverse(ep_id: str, depth: int):
            if depth > max_depth or ep_id in visited:
                return
            visited.add(ep_id)

            # Get dependencies first
            for _, target, _ in graph.out_edges(ep_id, data=True):
                traverse(target, depth + 1)

            chain.append(ep_id)

        traverse(endpoint_id, 0)
        return chain

    def get_graph_stats(self, api_id: str) -> dict[str, Any] | None:
        """Get statistics about a graph.

        Args:
            api_id: The API identifier.

        Returns:
            Dict with node/edge counts and other stats.
        """
        graph = self.get_graph(api_id)
        if graph is None:
            return None

        return {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(graph),
            "density": nx.density(graph),
        }

    def clear_cache(self) -> None:
        """Clear the in-memory graph cache."""
        self._cache.clear()
