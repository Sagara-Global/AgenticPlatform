"""GraphDefinition — the blueprint for a graph execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from suluv.core.types import NodeID, new_id, SuluvID
from suluv.core.engine.node import GraphNode
from suluv.core.engine.edge import GraphEdge


@dataclass
class GraphDefinition:
    """A complete graph definition containing nodes and edges.

    This is the serializable blueprint. GraphRuntime executes it.
    """

    graph_id: SuluvID = field(default_factory=new_id)
    name: str = ""
    nodes: dict[NodeID, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    entry_nodes: list[NodeID] = field(default_factory=list)
    exit_nodes: list[NodeID] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node

    def add_edge(
        self,
        source: GraphNode | NodeID,
        target: GraphNode | NodeID,
        **kwargs: Any,
    ) -> GraphEdge:
        """Add an edge between two nodes."""
        source_id = source.node_id if isinstance(source, GraphNode) else source
        target_id = target.node_id if isinstance(target, GraphNode) else target
        edge = GraphEdge(source_id=source_id, target_id=target_id, **kwargs)
        self.edges.append(edge)
        return edge

    def set_entry(self, *nodes: GraphNode | NodeID) -> None:
        """Set entry point nodes."""
        self.entry_nodes = [
            n.node_id if isinstance(n, GraphNode) else n for n in nodes
        ]

    def set_exit(self, *nodes: GraphNode | NodeID) -> None:
        """Set exit point nodes."""
        self.exit_nodes = [
            n.node_id if isinstance(n, GraphNode) else n for n in nodes
        ]

    def get_outgoing_edges(self, node_id: NodeID) -> list[GraphEdge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.source_id == node_id]

    def get_incoming_edges(self, node_id: NodeID) -> list[GraphEdge]:
        """Get all edges targeting a node."""
        return [e for e in self.edges if e.target_id == node_id]

    def get_source_nodes(self, node_id: NodeID) -> list[NodeID]:
        """Get IDs of nodes that feed into this node."""
        return [e.source_id for e in self.get_incoming_edges(node_id)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "entry_nodes": list(self.entry_nodes),
            "exit_nodes": list(self.exit_nodes),
            "metadata": self.metadata,
        }

    def validate(self) -> list[str]:
        """Validate the graph definition. Returns a list of errors."""
        errors = []
        if not self.nodes:
            errors.append("Graph has no nodes")
        if not self.entry_nodes:
            errors.append("Graph has no entry nodes")
        for nid in self.entry_nodes:
            if nid not in self.nodes:
                errors.append(f"Entry node {nid} not found in graph")
        for edge in self.edges:
            if edge.source_id not in self.nodes:
                errors.append(f"Edge source {edge.source_id} not found in graph")
            if edge.target_id not in self.nodes:
                errors.append(f"Edge target {edge.target_id} not found in graph")
        return errors
