"""RegionPartitioner: single bottom-up traversal for hierarchical decomposition.

Architecture (3 learned components):
  1. Embedding table: role -> vector
  2. RegionEncoder: MLP (x_N, children_aggregate) -> region representation
  3. EdgeClassifier: MLP (x_N, r_child, sibling_mean) -> cut probability

Inference follows the pseudo code exactly:
  - Bottom-up traversal (post-order)
  - At each node N, classify edges to N's children as merge/cut
  - Cut children become completed regions
  - Merged children's representations are aggregated into N's representation
  - Root's region set becomes the final region
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn

from core.types import Region, RegionPartition, Tree, TreeNode
from framework.decomposition.features import NodeFeatures


class RegionPartitioner(nn.Module):
    """Neural region partitioner with decision-dependent encoding."""

    def __init__(
        self,
        num_roles: int | None = None,
        role_embed_dim: int = 64,
        num_features: int = 5,       # depth, subtree_size, num_children, has_name, child_role_diversity
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_roles is None:
            from core.roles import NUM_ROLES
            num_roles = NUM_ROLES
        from core.roles import ROLE_TO_IDX
        self.role_embedding = nn.Embedding(num_roles, role_embed_dim, padding_idx=0)
        self._role_to_idx: Dict[str, int] = dict(ROLE_TO_IDX)

        local_dim = role_embed_dim + num_features

        def _dropout_layer() -> List[nn.Module]:
            return [nn.Dropout(dropout)] if dropout > 0 else []

        # RegionEncoder: (x_N, children_aggregate) -> region representation
        encoder_layers: List[nn.Module] = [nn.Linear(local_dim + hidden_dim, hidden_dim), nn.ReLU(), *_dropout_layer()]
        for _ in range(num_layers - 1):
            encoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), *_dropout_layer()])
        encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.region_encoder = nn.Sequential(*encoder_layers)

        # EdgeClassifier: (x_N, r_child, sibling_mean) -> cut probability
        classifier_layers: List[nn.Module] = [nn.Linear(local_dim + hidden_dim + hidden_dim, hidden_dim), nn.ReLU(), *_dropout_layer()]
        for _ in range(num_layers - 1):
            classifier_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), *_dropout_layer()])
        classifier_layers.append(nn.Linear(hidden_dim, 1))
        self.edge_classifier = nn.Sequential(*classifier_layers)

        self.hidden_dim = hidden_dim
        self.local_dim = local_dim

    def _get_role_idx(self, role: str) -> int:
        return self._role_to_idx.get(role, 0)  # 0 = unknown

    def _local_feature(
        self, node: TreeNode, features: Dict[int, NodeFeatures],
    ) -> torch.Tensor:
        """Compute x_N = Concat(E_role(N), num_N)."""
        role_idx = self._get_role_idx(node.role)
        role_idx_t = torch.tensor([role_idx], device=self._device())
        role_emb = self.role_embedding(role_idx_t).squeeze(0)

        nf = features[node.id]
        num_vec = torch.tensor(nf.numerical_vector(), dtype=torch.float32, device=self._device())

        return torch.cat([role_emb, num_vec])

    def _device(self) -> torch.device:
        return self.role_embedding.weight.device

    def forward(
        self,
        tree: Tree,
        features: Dict[int, NodeFeatures],
        threshold: float = 0.5,
        edge_labels: Dict[Tuple[int, int], int] | None = None,
    ) -> Tuple[RegionPartition, Dict[Tuple[int, int], torch.Tensor]]:
        """Run single bottom-up traversal.

        Args:
            tree: Input AXTree.
            features: Per-node features from extract_node_features.
            threshold: Cut probability threshold for inference.
            edge_labels: If provided (training), use ground-truth labels
                instead of model predictions for region assignment.

        Returns:
            (partition, edge_logits) where edge_logits maps (parent_id, child_id)
            to the raw logit tensor (for loss computation).
        """
        device = self._device()
        zero = torch.zeros(self.hidden_dim, device=device)

        representations: Dict[int, torch.Tensor] = {}
        region_sets: Dict[int, Set[int]] = {}
        completed_regions: List[Set[int]] = []
        edge_logits: Dict[Tuple[int, int], torch.Tensor] = {}

        for node in tree.post_order():
            region_sets[node.id] = {node.id}
            x_N = self._local_feature(node, features)

            if not node.children:
                # Leaf: r_N = RegionEncoder(x_N, 0)
                encoder_input = torch.cat([x_N, zero])
                representations[node.id] = self.region_encoder(encoder_input)
            else:
                children = node.children

                # Sibling mean (across N's children)
                sibling_mean = torch.stack([representations[c.id] for c in children]).mean(dim=0)

                merged_children: List[TreeNode] = []

                for child in children:
                    # EdgeClassifier(x_N, r_child, sibling_mean) -> cut logit
                    classifier_input = torch.cat([x_N, representations[child.id], sibling_mean])
                    logit = self.edge_classifier(classifier_input).squeeze(-1)
                    edge_logits[(node.id, child.id)] = logit

                    # Decide cut/merge
                    if edge_labels is not None:
                        is_cut = edge_labels.get((node.id, child.id), 0) == 1
                    else:
                        is_cut = torch.sigmoid(logit).item() >= threshold

                    if is_cut:
                        completed_regions.append(region_sets[child.id])
                    else:
                        merged_children.append(child)
                        region_sets[node.id] |= region_sets[child.id]

                # r_N = RegionEncoder(x_N, mean(merged children reprs))
                if merged_children:
                    merged_agg = torch.stack([representations[c.id] for c in merged_children]).mean(dim=0)
                else:
                    merged_agg = zero
                encoder_input = torch.cat([x_N, merged_agg])
                representations[node.id] = self.region_encoder(encoder_input)

        # Root's region set is the final region
        completed_regions.append(region_sets[tree.root.id])

        # Build RegionPartition
        partition = _build_partition(tree, completed_regions)

        return partition, edge_logits

    def partition(
        self, tree: Tree, features: Dict[int, NodeFeatures], threshold: float = 0.5,
    ) -> RegionPartition:
        """Inference-only: return partition without edge logits."""
        with torch.no_grad():
            partition, _ = self.forward(tree, features, threshold)
        return partition


def _build_partition(tree: Tree, region_sets: List[Set[int]]) -> RegionPartition:
    """Convert list of node-id sets into a RegionPartition."""
    regions: List[Region] = []
    node_to_region: Dict[int, int] = {}

    for rid, member_ids in enumerate(region_sets):
        # Find root of this region (shallowest node)
        root_id = min(member_ids, key=lambda nid: _node_depth(tree.nodes[nid]))
        root_node = tree.nodes[root_id]

        region = Region(
            region_id=rid,
            root_node_id=root_id,
            member_node_ids=member_ids,
            root_backend_id=root_node.backend_node_id,
        )
        regions.append(region)

        for nid in member_ids:
            node_to_region[nid] = rid

    return RegionPartition(regions=regions, tree=tree, node_to_region=node_to_region)


def _node_depth(node: TreeNode) -> int:
    depth = 0
    current = node
    while current.parent is not None:
        depth += 1
        current = current.parent
    return depth
