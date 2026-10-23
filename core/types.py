"""Common data structures used throughout the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """A single node in an accessibility tree."""

    id: int                                     # internal sequential ID
    backend_node_id: int                        # CDP backendDOMNodeId
    role: str                                   # accessibility role (link, button, heading, etc.)
    name: str                                   # accessible name (button text, image alt, etc.)
    value: str = ""                             # current input value (textbox content, etc.)
    bid: str = ""                               # BrowserGym element id (benchmark time only; empty during training)
    children: List[TreeNode] = field(default_factory=list)
    parent: Optional[TreeNode] = field(default=None, repr=False)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tree:
    """A full accessibility tree."""

    root: TreeNode
    nodes: Dict[int, TreeNode] = field(default_factory=dict)       # internal id -> node
    backend_id_map: Dict[int, int] = field(default_factory=dict)   # backend_node_id -> internal id

    def post_order(self) -> Iterator[TreeNode]:
        """Yield nodes in post-order (children before parent)."""
        def _walk(node: TreeNode) -> Iterator[TreeNode]:
            for child in node.children:
                yield from _walk(child)
            yield node
        yield from _walk(self.root)

    def pre_order(self) -> Iterator[TreeNode]:
        """Yield nodes in pre-order (parent before children)."""
        def _walk(node: TreeNode) -> Iterator[TreeNode]:
            yield node
            for child in node.children:
                yield from _walk(child)
        yield from _walk(self.root)

    def get_subtree_nodes(self, node_id: int) -> List[TreeNode]:
        """Return all nodes in the subtree rooted at *node_id*."""
        root = self.nodes[node_id]
        result: List[TreeNode] = []
        def _collect(node: TreeNode) -> None:
            result.append(node)
            for child in node.children:
                _collect(child)
        _collect(root)
        return result

    def get_node_by_backend_id(self, backend_id: int) -> Optional[TreeNode]:
        """Look up a node by its backend DOM/AX node id."""
        internal_id = self.backend_id_map.get(backend_id)
        if internal_id is None:
            return None
        return self.nodes.get(internal_id)


# ---------------------------------------------------------------------------
# Region (Decomposition output)
# ---------------------------------------------------------------------------

@dataclass
class Region:
    """A functional region: a non-overlapping subtree slice of an AXTree."""

    region_id: int
    root_node_id: int                   # internal ID of the region root
    member_node_ids: Set[int]           # all internal node IDs in this region
    root_backend_id: int                # backendDOMNodeId of the root


@dataclass
class RegionPartition:
    """Complete partitioning of a tree into non-overlapping regions."""

    regions: List[Region]
    tree: Tree
    node_to_region: Dict[int, int] = field(default_factory=dict)  # node_id -> region_id


# ---------------------------------------------------------------------------
# Semantic Abstraction output
# ---------------------------------------------------------------------------

@dataclass
class RegionRepr:
    """Semantic representation of a region: purpose + state summary."""

    region_id: int
    purpose: str                        # what the region is for
    state_summary: str                  # current condition of the region
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Transition (Page-aware observation management)
# ---------------------------------------------------------------------------

@dataclass
class TransitionEntry:
    """A single node-level change detected between two AXTree snapshots."""

    change_type: str                    # "added" | "removed" | "modified"
    backend_node_id: int
    role: str
    name: str
    detail: str = ""                    # e.g. "name: 'old' -> 'new'"
    depth: int = 0                      # depth for serialization indentation.
                                        #  - added / modified : depth in the *after* tree
                                        #  - removed          : (nearest surviving ancestor's
                                        #                        after-tree depth) + 1 + its
                                        #                        relative depth inside the
                                        #                        removed subtree


@dataclass
class TransitionRecord:
    """All changes detected after one agent action within the same page."""

    step: int
    entries: List[TransitionEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Observation (delivered to actor agent)
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """Observation payload delivered to the actor agent."""

    url: str
    region_reprs: List[RegionRepr]                          # all regions' purpose + state summary
    selected_region_ids: List[int]                          # task-relevant region IDs
    selected_content: Dict[int, str]                        # region_id -> serialized AXTree subtree (for analysis/dump)
    selected_text: str = ""                                 # single connected skeleton: selected rendered, others "(omitted)"
    transition_history: List[TransitionRecord] = field(default_factory=list)
    action_history: List[str] = field(default_factory=list)
