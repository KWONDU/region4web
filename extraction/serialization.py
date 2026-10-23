"""AXTree serialization.

Based on BrowserGym ``flatten_axtree_to_str`` with structural grouping preservation.

Filtering rules (applied at serialization time):
  - Skip LineBreak roles.
  - Skip InlineTextBox roles.
  - Skip generic/none nodes that have no properties, UNLESS they are
    structural grouping nodes. A generic/none node is a structural
    grouping node when it has 2+ direct children that each contain
    at least one visible descendant (diverges from BrowserGym which
    skips all generic/none without properties).
  - Skip StaticText whose text is already contained in parent's name.

Attribute rules (BrowserGym denylist):
  - Ignored: editable, readonly, level, settable, multiline, invalid, focusable.
  - Boolean-only (bare name when true): required, focused, atomic.
  - All other properties: key=repr(value).
"""

from __future__ import annotations

from typing import Set

from core.types import Region, Tree, TreeNode

IGNORED_AXTREE_ROLES: Set[str] = {"LineBreak"}

IGNORED_AXTREE_PROPERTIES: Set[str] = {
    "editable", "readonly", "level", "settable",
    "multiline", "invalid", "focusable",
}

_BOOL_ONLY_PROPERTIES: Set[str] = {"required", "focused", "atomic"}

_SKIPPABLE_ROLES: Set[str] = {"generic", "none", "InlineTextBox", "LineBreak"}


def _has_visible_descendant(node: TreeNode, member_ids: Set[int] | None = None) -> bool:
    """Check if a node's subtree contains at least one visible (non-skippable) node."""
    if member_ids is not None and node.id not in member_ids:
        return False
    if node.role not in _SKIPPABLE_ROLES:
        return True
    if node.role in ("generic", "none") and node.name and node.name.strip():
        return True
    for child in node.children:
        if _has_visible_descendant(child, member_ids):
            return True
    return False


def _is_structural_grouping(node: TreeNode, member_ids: Set[int] | None = None) -> bool:
    """Check if a none/generic node groups 2+ branches with visible content."""
    visible_branches = 0
    for child in node.children:
        if member_ids is not None and child.id not in member_ids:
            continue
        if _has_visible_descendant(child, member_ids):
            visible_branches += 1
            if visible_branches >= 2:
                return True
    return False


def serialize_full_tree(tree: Tree) -> str:
    """Serialize the entire AXTree to indented text."""
    lines: list[str] = []

    def _dfs(node: TreeNode, depth: int, parent_name: str) -> None:
        skip, node_str = _format_node(node, depth, parent_name, None)
        if not skip:
            lines.append(node_str)
        child_depth = depth if skip else depth + 1
        child_parent_name = node.name if not skip else parent_name
        for child in node.children:
            _dfs(child, child_depth, child_parent_name)

    _dfs(tree.root, 0, "")
    return "\n".join(lines)


def serialize_region_subtree(tree: Tree, region: Region) -> str:
    """Serialize only nodes belonging to a region."""
    root = tree.nodes[region.root_node_id]
    member_ids = region.member_node_ids
    lines: list[str] = []

    def _dfs(node: TreeNode, depth: int, parent_name: str) -> None:
        if node.id not in member_ids:
            return
        skip, node_str = _format_node(node, depth, parent_name, member_ids)
        if not skip:
            lines.append(node_str)
        child_depth = depth if skip else depth + 1
        child_parent_name = node.name if not skip else parent_name
        for child in node.children:
            _dfs(child, child_depth, child_parent_name)

    _dfs(root, 0, "")
    return "\n".join(lines)


def _format_node(
    node: TreeNode, depth: int, parent_name: str,
    member_ids: Set[int] | None = None,
) -> tuple[bool, str]:
    """Return (skip, formatted_line). Skip=True means node not printed but children traversed."""
    node_role = node.role
    node_name = node.name

    # Skip ignored roles
    if node_role in IGNORED_AXTREE_ROLES:
        return True, ""

    # Skip nodes without name (InlineTextBox etc.)
    if node_role == "InlineTextBox":
        return True, ""

    # Build attributes
    attributes: list[str] = []
    for prop_name, prop_value in node.properties.items():
        if prop_name in IGNORED_AXTREE_PROPERTIES:
            continue
        if prop_name in ("src", "href"):
            continue
        if prop_name in _BOOL_ONLY_PROPERTIES:
            if str(prop_value).lower() == "true":
                attributes.append(prop_name)
        else:
            attributes.append(f"{prop_name}={repr(prop_value)}")

    # Skip generic/none without attributes, unless structural grouping node
    # (2+ child branches each containing visible descendants)
    if node_role in ("generic", "none") and not attributes:
        if not _is_structural_grouping(node, member_ids):
            return True, ""

    # Skip redundant StaticText
    if node_role == "StaticText":
        if parent_name and node_name in parent_name:
            return True, ""

    # --- Build output line ---
    indent = "\t" * depth

    # Prefer BrowserGym bid (benchmark time); fall back to backend_node_id (training time).
    # The id shown here is what the actor will reference in its action output.
    node_id_str = node.bid or (str(node.backend_node_id) if node.backend_node_id else "")
    if node_id_str:
        line = f"{indent}[{node_id_str}] {node_role}"
    else:
        line = f"{indent}{node_role}"

    # Name (or src/href fallback for nameless image/link)
    if node_role == "generic" and not node_name:
        pass
    elif node_name is not None:
        if node_name.strip():
            line += f" {repr(node_name.strip())}"
        elif node_role == "image":
            src = node.properties.get("src", "")
            if src:
                line += f" {repr(src)}"
        elif node_role == "link":
            href = node.properties.get("href", "")
            if href:
                line += f" {repr(href)}"

    # Value
    if node.value:
        line += f" value={repr(node.value)}"

    # Attributes
    if attributes:
        line += ", ".join([""] + attributes)

    return False, line
