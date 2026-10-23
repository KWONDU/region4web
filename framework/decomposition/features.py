"""Per-node feature extraction for Hierarchical Decomposition.

Each node N's local feature vector: x_N = Concat(E_role(N), num_N)  →  16d

Categorical:
  - E_role: role embedding (11d, learned)

Numerical (5d):
  - depth: tree depth of the node
  - subtree_size: number of nodes in the subtree rooted at this node
  - num_children: number of direct children
  - has_name: 1 if accessible name present, else 0
  - child_role_diversity: unique_child_roles / num_children (0 if leaf)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from core.types import Tree


@dataclass
class NodeFeatures:
    """Feature container for a single node."""

    node_id: int
    role: str
    depth: int
    subtree_size: int
    num_children: int
    has_name: int
    child_role_diversity: float

    def numerical_vector(self) -> list[float]:
        """Return numerical features as a list."""
        return [
            float(self.depth),
            float(self.subtree_size),
            float(self.num_children),
            float(self.has_name),
            float(self.child_role_diversity),
        ]


def extract_node_features(tree: Tree) -> Dict[int, NodeFeatures]:
    """Extract features for every node in the tree.

    Returns mapping from internal node id to NodeFeatures.
    """
    features: Dict[int, NodeFeatures] = {}

    subtree_sizes: Dict[int, int] = {}
    for node in tree.post_order():
        subtree_sizes[node.id] = 1 + sum(subtree_sizes[c.id] for c in node.children)

    depths: Dict[int, int] = {}
    for node in tree.pre_order():
        if node.parent is None:
            depths[node.id] = 0
        else:
            depths[node.id] = depths[node.parent.id] + 1

    for node in tree.nodes.values():
        has_name = 1 if node.name.strip() else 0
        if node.children:
            unique_roles = len({c.role for c in node.children})
            diversity = unique_roles / len(node.children)
        else:
            diversity = 0.0

        features[node.id] = NodeFeatures(
            node_id=node.id,
            role=node.role,
            depth=depths[node.id],
            subtree_size=subtree_sizes[node.id],
            num_children=len(node.children),
            has_name=has_name,
            child_role_diversity=diversity,
        )

    return features
