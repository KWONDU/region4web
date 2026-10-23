"""Construct the actor-facing observation for the digest_page (pagedigest) mode.

Per page entry, the entry AXTree is split into per-region blocks
``<Rx purpose="..."> ... </Rx>``. Selected regions render their full subtree
inside the block; non-selected regions appear as the opening-tag line only.

Within-page steps re-render against the entry tree with
``serialize_digest_observation_minus_removed`` so removed nodes vanish (and
fully-gone regions drop entirely). The newly-appeared nodes are listed in a
separate `<added_elements>` block by the caller.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from core.types import Region, RegionPartition, RegionRepr, Tree, TreeNode
from extraction.serialization import _format_node


def _format_region_header(region_id: int, purpose: str) -> str:
    safe_p = purpose.replace('"', '&quot;')
    return f'<R{region_id} purpose="{safe_p}">'


def _format_region_footer(region_id: int) -> str:
    return f"</R{region_id}>"


def _serialize_region_subtree_minus_removed(
    tree: Tree, region: Region, removed_bids: Set[int],
) -> Tuple[str, int, int]:
    """Render a region's subtree silently skipping removed-bid nodes (and
    their descendants). Returns ``(text, n_active, n_removed)``:

    * ``n_active``  — region members whose bid is still in the current tree
                      (the caller drops the region entirely when this is 0).
    * ``n_removed`` — region members removed since entry; surfaced by the
                      caller as a single ``(N nodes removed)`` line under
                      the region header.
    """
    root = tree.nodes[region.root_node_id]
    member_ids = region.member_node_ids

    n_active = 0
    n_removed_count = 0
    for nid in member_ids:
        n = tree.nodes.get(nid)
        if n is None or not n.backend_node_id:
            continue
        if n.backend_node_id in removed_bids:
            n_removed_count += 1
        else:
            n_active += 1
    if n_active == 0:
        return "", 0, n_removed_count

    lines: List[str] = []

    def _dfs(node: TreeNode, depth: int, parent_name: str) -> None:
        if node.id not in member_ids:
            return
        if node.backend_node_id and node.backend_node_id in removed_bids:
            return
        skip, line = _format_node(node, depth, parent_name, member_ids)
        if not skip:
            lines.append(line)
        child_depth = depth if skip else depth + 1
        child_parent_name = node.name if not skip else parent_name
        for child in node.children:
            _dfs(child, child_depth, child_parent_name)

    _dfs(root, 0, "")
    return "\n".join(lines), n_active, n_removed_count


def serialize_digest_observation_minus_removed(
    tree: Tree,
    partition: RegionPartition,
    region_reprs: List[RegionRepr],
    selected_ids: List[int],
    removed_bids: Set[int],
) -> str:
    """Render the digest_page observation against the entry tree, splicing
    out nodes whose backend_id is in ``removed_bids``:

    * Regions whose member set is fully gone are dropped entirely.
    * Selected regions render their (post-splice) subtree inside the block.
    * Non-selected regions emit the opening tag with no inner content.
    * If a region has any removed members, a ``(N nodes removed)`` line is
      surfaced under the header.
    """
    selected_set = set(selected_ids)
    purpose_lookup: Dict[int, str] = {}
    for rr in region_reprs:
        purpose_lookup[rr.region_id] = rr.purpose or "(no purpose)"

    blocks: List[str] = []
    for region in sorted(partition.regions, key=lambda r: r.region_id):
        rid = region.region_id
        if rid in selected_set:
            subtree_text, n_active, n_removed = _serialize_region_subtree_minus_removed(
                tree, region, removed_bids,
            )
        else:
            subtree_text = ""
            n_active = 0
            n_removed = 0
            for nid in region.member_node_ids:
                n = tree.nodes.get(nid)
                if not n or not n.backend_node_id:
                    continue
                if n.backend_node_id in removed_bids:
                    n_removed += 1
                else:
                    n_active += 1

        if n_active == 0:
            continue

        lines = [_format_region_header(rid, purpose_lookup.get(rid, "(unknown)"))]
        if n_removed > 0:
            lines.append(f"({n_removed} nodes removed)")
        if subtree_text:
            lines.append(subtree_text)
        lines.append(_format_region_footer(rid))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
