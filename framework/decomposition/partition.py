"""Region extraction from cut edge sets."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from core.types import Region, RegionPartition, Tree, TreeNode


def extract_regions(
    tree: Tree,
    cut_edges: Set[Tuple[int, int]],
) -> RegionPartition:
    """Build a RegionPartition from a set of cut edges.

    Each parent-child edge in the tree is either merge (not in cut_edges)
    or cut (in cut_edges). Connected components of merge edges define regions.

    Args:
        tree: The AXTree.
        cut_edges: Set of (parent_id, child_id) tuples where edges are cut.

    Returns:
        A RegionPartition with non-overlapping regions.
    """
    # Assign each node to a region via DFS
    region_sets: List[Set[int]] = []
    node_to_region: Dict[int, int] = {}

    def _dfs(node: TreeNode, current_set: Set[int]) -> None:
        current_set.add(node.id)
        for child in node.children:
            if (node.id, child.id) in cut_edges:
                # Cut: child starts a new region
                new_set: Set[int] = set()
                region_sets.append(new_set)
                _dfs(child, new_set)
            else:
                # Merge: child belongs to same region
                _dfs(child, current_set)

    root_set: Set[int] = set()
    region_sets.append(root_set)
    _dfs(tree.root, root_set)

    # Build Region objects
    regions: List[Region] = []
    for rid, member_ids in enumerate(region_sets):
        # Root of region = shallowest node
        root_id = min(member_ids, key=lambda nid: _depth(tree.nodes[nid]))
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


def _depth(node: TreeNode) -> int:
    d = 0
    current = node
    while current.parent is not None:
        d += 1
        current = current.parent
    return d
