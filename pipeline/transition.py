"""Detect and serialize transitions between two AXTree snapshots.

Used by the ``digest_page`` mode to render a cumulative ``<added_elements>``
block (page-entry vs current). Modifications and removals are NOT surfaced
through this block — the digest renderer applies modifications in-place on
the entry tree and splices removed nodes out with a ``(N nodes removed)``
placeholder, so only ``Added`` ever appears in the actor-facing block.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterator, List, Set

from core.types import TransitionEntry, TransitionRecord, Tree, TreeNode


def detect_transitions(before: Tree, after: Tree) -> TransitionRecord:
    """Compare two AXTree snapshots and return a TransitionRecord.

    Walks the *after* tree in pre-order. For each visited node, emits an
    ``added`` or ``modified`` entry if applicable, then any removed-subtree
    anchored at this node. Orphan removed subtrees (no surviving ancestor in
    *after*) are appended at the end with depth 0.
    """
    before_ids: Set[int] = set(before.backend_id_map.keys())
    after_ids: Set[int] = set(after.backend_id_map.keys())

    added_ids = after_ids - before_ids
    removed_ids = before_ids - after_ids
    common_ids = before_ids & after_ids

    anchored: Dict[int, List[TreeNode]] = defaultdict(list)
    orphan_removed: List[TreeNode] = []

    for node in before.pre_order():
        bid = node.backend_node_id
        if not bid or bid not in removed_ids:
            continue
        parent = node.parent
        parent_removed = parent is not None and parent.backend_node_id in removed_ids
        if parent_removed:
            continue   # not a subtree root — covered by its ancestor's walk

        cur = parent
        while cur is not None and (not cur.backend_node_id or cur.backend_node_id not in after_ids):
            cur = cur.parent
        if cur is None:
            orphan_removed.append(node)
        else:
            anchored[cur.backend_node_id].append(node)

    entries: List[TransitionEntry] = []
    for node in after.pre_order():
        bid = node.backend_node_id
        if not bid:
            continue
        d = _node_depth(node)

        if bid in added_ids:
            entries.append(TransitionEntry(
                change_type="added",
                backend_node_id=bid, role=node.role, name=node.name, depth=d,
            ))
        elif bid in common_ids:
            before_node = before.get_node_by_backend_id(bid)
            if before_node is not None and _node_changed(before_node, node):
                entries.append(TransitionEntry(
                    change_type="modified",
                    backend_node_id=bid, role=node.role, name=node.name,
                    detail=_describe_change(before_node, node), depth=d,
                ))

        if bid in anchored:
            for rem_root in anchored[bid]:
                _emit_removed_subtree(rem_root, base_depth=d + 1, out=entries)

    for rem_root in orphan_removed:
        _emit_removed_subtree(rem_root, base_depth=0, out=entries)

    return TransitionRecord(step=0, entries=entries)


def serialize_observation_transitions(
    record: TransitionRecord,
    include_removed: bool = True,
    include_modified: bool = True,
) -> str:
    """Render a cumulative ``TransitionRecord`` as a textual block.

    For ``digest_page`` (``include_modified=False, include_removed=False``)
    the output collapses to ``<added_elements>...</added_elements>`` listing
    just the new nodes (tree-depth preserved). With both flags True the
    output is ``<observation_transitions>`` with ``Added``/``Modified``/
    ``Removed`` sections.
    """
    if not record.entries:
        return ""

    added = [e for e in record.entries if e.change_type == "added"]
    modified = [e for e in record.entries if e.change_type == "modified"]
    removed = [e for e in record.entries if e.change_type == "removed"]

    sections: List[str] = []
    if added:
        lines = ["Added:"]
        min_d = min(e.depth for e in added)
        for e in added:
            rel = "\t" * (e.depth - min_d + 1)
            name_str = f" {repr(e.name)}" if e.name else ""
            lines.append(f"{rel}[{e.backend_node_id}] {e.role}{name_str}")
        sections.append("\n".join(lines))

    if modified and include_modified:
        lines = ["Modified:"]
        for e in modified:
            name_str = f" {repr(e.name)}" if e.name else ""
            detail_str = f" ({e.detail})" if e.detail else ""
            lines.append(f"\t[{e.backend_node_id}] {e.role}{name_str}{detail_str}")
        sections.append("\n".join(lines))

    if removed and include_removed:
        lines = ["Removed:"]
        for e in removed:
            name_str = f" {repr(e.name)}" if e.name else ""
            lines.append(f"\t[{e.backend_node_id}] {e.role}{name_str}")
        sections.append("\n".join(lines))

    if not sections:
        return ""
    body = "\n".join(sections)
    if not include_modified and not include_removed:
        if not added:
            return ""
        added_body = body.replace("Added:\n", "", 1)
        return f"<added_elements>\n{added_body}\n</added_elements>"
    return f"<observation_transitions>\n{body}\n</observation_transitions>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_depth(node: TreeNode) -> int:
    d, cur = 0, node
    while cur.parent is not None:
        d += 1
        cur = cur.parent
    return d


def _subtree_pre_order(node: TreeNode) -> Iterator[TreeNode]:
    yield node
    for child in node.children:
        yield from _subtree_pre_order(child)


def _emit_removed_subtree(
    root: TreeNode, base_depth: int, out: List[TransitionEntry],
) -> None:
    """Append ``removed`` entries for all nodes in ``root``'s subtree.

    Internal structure of the removed subtree is preserved: each descendant's
    depth is ``base_depth + (its before-tree depth − root's before-tree depth)``.
    Nodes without a backend_id are skipped.
    """
    root_depth_before = _node_depth(root)
    for n in _subtree_pre_order(root):
        bid = n.backend_node_id
        if not bid:
            continue
        rel = _node_depth(n) - root_depth_before
        out.append(TransitionEntry(
            change_type="removed",
            backend_node_id=bid, role=n.role, name=n.name,
            depth=base_depth + rel,
        ))


def _node_changed(before: TreeNode, after: TreeNode) -> bool:
    return (
        before.name != after.name
        or before.role != after.role
        or before.value != after.value
        or before.properties != after.properties
    )


def _describe_change(before: TreeNode, after: TreeNode) -> str:
    parts: List[str] = []
    if before.name != after.name:
        parts.append(f"name: {repr(before.name)} -> {repr(after.name)}")
    if before.role != after.role:
        parts.append(f"role: {repr(before.role)} -> {repr(after.role)}")
    if before.value != after.value:
        parts.append(f"value: {repr(before.value)} -> {repr(after.value)}")
    if before.properties != after.properties:
        parts.append("properties changed")
    return ", ".join(parts)
