"""Build AXTree from raw CDP Accessibility.getFullAXTree response."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.types import Tree, TreeNode


def build_axtree(raw_nodes: List[Dict[str, Any]]) -> Tree:
    """Convert the flat node list from ``Accessibility.getFullAXTree`` into a :class:`Tree`.

    Raw nodes should already be enriched via :func:`enrich_raw_nodes_from_dom`
    before calling this function.
    """
    cdp_id_to_node: Dict[str, TreeNode] = {}
    internal_id = 0

    for raw in raw_nodes:
        cdp_node_id: str = raw["nodeId"]
        role_value = _extract_value(raw.get("role", {}))
        name_value = _extract_value(raw.get("name", {}))
        node_value = _extract_value(raw.get("value", {}))
        backend_id = raw.get("backendDOMNodeId", 0)

        props: Dict[str, Any] = {}
        for prop in raw.get("properties", []):
            props[prop["name"]] = _extract_value(prop.get("value", {}))

        node = TreeNode(
            id=internal_id,
            backend_node_id=backend_id,
            role=role_value,
            name=name_value,
            value=node_value,
            bid=str(raw.get("browsergym_id", "")),   # populated when AXTree came through BrowserGym
            properties=props,
        )
        cdp_id_to_node[cdp_node_id] = node
        internal_id += 1

    for raw in raw_nodes:
        cdp_node_id = raw["nodeId"]
        parent_node = cdp_id_to_node[cdp_node_id]
        for child_id in raw.get("childIds", []):
            child = cdp_id_to_node.get(child_id)
            if child is not None:
                child.parent = parent_node
                parent_node.children.append(child)

    root = next(
        (n for n in cdp_id_to_node.values() if n.parent is None),
        list(cdp_id_to_node.values())[0],
    )

    nodes_map = {n.id: n for n in cdp_id_to_node.values()}
    backend_map = {n.backend_node_id: n.id for n in cdp_id_to_node.values() if n.backend_node_id}

    return Tree(root=root, nodes=nodes_map, backend_id_map=backend_map)


def enrich_raw_nodes_from_dom(
    raw_nodes: List[Dict[str, Any]],
    cdp_session: Any,
) -> None:
    """Enrich raw CDP nodes in-place with DOM attributes (src/href).

    For alt-less images, fetches ``src``. For text-less links, fetches ``href``.
    Call this while the CDP session is alive, before saving or building a Tree.
    """
    for raw in raw_nodes:
        role = _extract_value(raw.get("role", {}))
        name = _extract_value(raw.get("name", {}))
        backend_id = raw.get("backendDOMNodeId", 0)
        attr_name = _dom_attr_to_fetch(role, name, backend_id)
        if attr_name is None:
            continue
        value = _fetch_dom_attr(cdp_session, backend_id, attr_name)
        if value is not None:
            props = raw.setdefault("properties", [])
            props.append({
                "name": attr_name,
                "value": {"type": "string", "value": value},
            })


def _dom_attr_to_fetch(role: str, name: str, backend_id: int) -> Optional[str]:
    """Return the DOM attribute name to fetch, or None if not needed."""
    if role not in ("image", "link"):
        return None
    if name and name.strip():
        return None
    if not backend_id:
        return None
    return "src" if role == "image" else "href"


def _fetch_dom_attr(cdp_session: Any, backend_id: int, attr_name: str) -> Optional[str]:
    """Fetch a single DOM attribute via CDP."""
    try:
        result = cdp_session.send(
            "DOM.describeNode",
            {"backendNodeId": backend_id},
        )
        attrs = result.get("node", {}).get("attributes", [])
        for i in range(0, len(attrs) - 1, 2):
            if attrs[i] == attr_name:
                return attrs[i + 1]
    except Exception:
        pass
    return None


def _extract_value(value_obj: Dict[str, Any] | str) -> str:
    if isinstance(value_obj, str):
        return value_obj
    return str(value_obj.get("value", ""))
