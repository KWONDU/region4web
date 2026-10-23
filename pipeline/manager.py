"""Pipeline manager: orchestrates decomposition -> abstraction -> selection ->
observation rendering for the ``pagedigest`` (digest_page) mode.

Per page entry: full pipeline runs; the entry AXTree snapshot is cached as
the cumulative-diff baseline. Within-page steps reuse the cached entry tree
plus the action's after-tree to render an `<added_elements>` block of new
nodes. The actor's ``view_all()`` action re-runs the framework against the
current tree, exposes every region's full subtree for the rest of the page,
and resets the diff baseline.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.types import (
    Observation,
    RegionPartition,
    RegionRepr,
    TransitionRecord,
    Tree,
)
from framework.decomposition.features import extract_node_features
from framework.decomposition.model import RegionPartitioner
from framework.abstraction.representor import RegionRepresentor
from pipeline.observation_builder import serialize_digest_observation_minus_removed
from pipeline.transition import detect_transitions, serialize_observation_transitions
from pipeline.selection.selector import RegionSelector


class ObservationManager:
    """Manage per-URL observation lifecycle for the digest_page mode."""

    def __init__(
        self,
        partitioner: Optional[RegionPartitioner] = None,
        representor: Optional[RegionRepresentor] = None,
        selector: Optional[RegionSelector] = None,
        threshold: float = 0.5,
        observation_mode: str = "digest_page",
    ) -> None:
        if observation_mode != "digest_page":
            raise ValueError(
                f"Only observation_mode='digest_page' is supported "
                f"(got {observation_mode!r})."
            )
        self.partitioner = partitioner
        self.representor = representor
        self.selector = selector
        self.threshold = threshold
        self.observation_mode = observation_mode

        # State
        self.current_url: Optional[str] = None
        self.current_observation: Optional[Observation] = None
        self.current_partition: Optional[RegionPartition] = None
        self.before_snapshot: Optional[Tree] = None
        self.transition_history: List[TransitionRecord] = []
        self.action_history: List[str] = []
        self.step_counter: int = 0

        # Page-entry snapshot held constant until the next URL change. The
        # cumulative `<added_elements>` block for every within-page step
        # is computed against this tree.
        self.entry_axtree: Optional[Tree] = None

        # When the actor calls the custom ``view_all()`` action, the manager
        # re-runs decomposition + abstraction against the current AXTree,
        # pins ``selected_ids`` to all regions, and resets the diff baseline
        # to the view_all moment. Cleared on the next page entry.
        self.view_all_active: bool = False

        # Per-call diagnostics (populated by on_url_change / on_action_performed;
        # the runner reads these to dump full trajectory JSONL files).
        self.last_diagnostics: Dict[str, Any] = {}

    def reset_for_new_task(self) -> None:
        """Clear all per-task state so the manager can be reused across tasks."""
        self.current_url = None
        self.current_observation = None
        self.current_partition = None
        self.before_snapshot = None
        self.transition_history = []
        self.action_history = []
        self.step_counter = 0
        self.entry_axtree = None
        self.last_diagnostics = {}
        self.view_all_active = False

    def on_url_change(self, new_url: str, axtree: Tree, task: str) -> Observation:
        """Run the full pipeline (decomp + abstract + select) on a new page."""
        self.current_url = new_url
        self.step_counter = 0
        self.transition_history = []
        self.last_diagnostics = {}
        self.view_all_active = False

        partition = self._run_decomposition(axtree)
        region_reprs = self._run_abstraction(partition, axtree)
        selected_ids = self._run_selection(region_reprs, task)

        self.entry_axtree = axtree
        from extraction.serialization import serialize_region_subtree
        obs_text = self._build_page_text(
            axtree, partition=partition,
            region_reprs=region_reprs, selected_ids=selected_ids,
        )
        selected_set = set(selected_ids)
        content: Dict[int, str] = {
            r.region_id: serialize_region_subtree(axtree, r)
            for r in partition.regions if r.region_id in selected_set
        }

        self.current_observation = Observation(
            url=new_url,
            region_reprs=region_reprs,
            selected_region_ids=selected_ids,
            selected_content=content,
            selected_text=obs_text,
            transition_history=[],
            action_history=list(self.action_history),
        )
        self.before_snapshot = axtree
        return self.current_observation

    def on_view_all(self, axtree: Tree, task: str) -> Observation:
        """Handle the actor's ``view_all()`` action.

        Re-runs decomposition + abstraction on the **current** AXTree (no
        selection LLM call), pins ``selected_ids`` to every region so every
        subtree is rendered in full, and resets the cumulative-diff baseline
        to this moment.
        """
        self.step_counter += 1
        self.action_history.append("view_all()")
        self.last_diagnostics = {}

        partition = self._run_decomposition(axtree)
        region_reprs = self._run_abstraction(partition, axtree)
        self.last_diagnostics["selection_call"] = None
        self.last_diagnostics["selection_ms"] = 0

        self.entry_axtree = axtree
        self.view_all_active = True
        selected_ids = [r.region_id for r in partition.regions]

        from extraction.serialization import serialize_region_subtree
        obs_text = self._build_page_text(
            axtree, partition=partition,
            region_reprs=region_reprs, selected_ids=selected_ids,
        )
        content: Dict[int, str] = {
            r.region_id: serialize_region_subtree(axtree, r)
            for r in partition.regions
        }

        self.current_observation = Observation(
            url=self.current_url or "",
            region_reprs=region_reprs,
            selected_region_ids=selected_ids,
            selected_content=content,
            selected_text=obs_text,
            transition_history=[],
            action_history=list(self.action_history),
        )
        self.before_snapshot = axtree
        return self.current_observation

    def on_action_performed(self, action_str: str, after_axtree: Tree) -> TransitionRecord:
        """Recompute the actor-facing text after an env action.

        Cumulative diff entry → after, rendered as `<added_elements>`. The
        cached page-entry digest base text is reused unchanged.
        """
        self.step_counter += 1
        self.action_history.append(action_str)

        if self.before_snapshot is None:
            self.before_snapshot = after_axtree
            return TransitionRecord(step=self.step_counter)

        if (self.entry_axtree is not None and self.current_observation is not None):
            self.current_observation.selected_text = self._build_page_text(after_axtree)
            self.current_observation.action_history = list(self.action_history)
        self.before_snapshot = after_axtree
        return TransitionRecord(step=self.step_counter)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _run_decomposition(self, axtree: Tree) -> RegionPartition:
        t0 = time.perf_counter()
        features = extract_node_features(axtree)
        if self.partitioner is not None:
            partition = self.partitioner.partition(axtree, features, self.threshold)
        else:
            from core.types import Region
            region = Region(
                region_id=0,
                root_node_id=axtree.root.id,
                member_node_ids=set(axtree.nodes.keys()),
                root_backend_id=axtree.root.backend_node_id,
            )
            partition = RegionPartition(
                regions=[region], tree=axtree,
                node_to_region={nid: 0 for nid in axtree.nodes},
            )
        self.current_partition = partition
        self.last_diagnostics["decomposition_ms"] = int((time.perf_counter() - t0) * 1000)
        return partition

    def _run_abstraction(self, partition: RegionPartition, axtree: Tree) -> List[RegionRepr]:
        t0 = time.perf_counter()
        if self.representor is not None:
            region_reprs = self.representor.represent_all(partition, axtree)
        else:
            region_reprs = [
                RegionRepr(region_id=r.region_id, purpose="", state_summary="")
                for r in partition.regions
            ]
        self.last_diagnostics["abstraction_ms"] = int((time.perf_counter() - t0) * 1000)
        return region_reprs

    def _run_selection(self, region_reprs: List[RegionRepr], task: str) -> List[int]:
        t0 = time.perf_counter()
        if self.selector is not None:
            selected_ids = self.selector.select(
                region_reprs, task,
                action_history=list(self.action_history),
            )
            self.last_diagnostics["selection_call"] = self.selector.last_call
        else:
            selected_ids = [r.region_id for r in region_reprs]
            self.last_diagnostics["selection_call"] = None
        self.last_diagnostics["selection_ms"] = int((time.perf_counter() - t0) * 1000)
        return selected_ids

    def _build_page_text(
        self,
        after_axtree: Tree,
        partition: Optional[RegionPartition] = None,
        region_reprs: Optional[List[RegionRepr]] = None,
        selected_ids: Optional[List[int]] = None,
    ) -> str:
        """Render the actor-facing text for digest_page.

        The base AXTree is rendered against the entry snapshot but with
        removed-bid nodes spliced out (regions whose member set is fully
        gone are dropped). An `<added_elements>` block listing every node
        that has appeared since page entry is appended.

        ``partition`` / ``region_reprs`` / ``selected_ids`` are passed in
        on the page-entry call (when ``self.current_observation`` does not
        exist yet); on subsequent within-page calls they are read off
        ``self.current_partition`` and ``self.current_observation``.
        """
        if self.entry_axtree is None:
            return ""

        entry_bids = set(self.entry_axtree.backend_id_map.keys())
        current_bids = set(after_axtree.backend_id_map.keys())
        removed_bids = entry_bids - current_bids

        # Apply in-place attribute updates from after_axtree onto the entry
        # tree for any bid present in both. This makes modified nodes render
        # with their current attributes when the renderer walks the entry
        # tree, so a separate "Modified" footer is unnecessary.
        for bid in entry_bids & current_bids:
            entry_node = self.entry_axtree.get_node_by_backend_id(bid)
            after_node = after_axtree.get_node_by_backend_id(bid)
            if entry_node is None or after_node is None:
                continue
            entry_node.role = after_node.role
            entry_node.name = after_node.name
            entry_node.value = after_node.value
            entry_node.properties = dict(after_node.properties)

        part = partition or self.current_partition
        reprs = region_reprs or (
            self.current_observation.region_reprs
            if self.current_observation else []
        )
        sel = selected_ids if selected_ids is not None else (
            self.current_observation.selected_region_ids
            if self.current_observation else []
        )
        base = serialize_digest_observation_minus_removed(
            tree=self.entry_axtree, partition=part,
            region_reprs=reprs, selected_ids=sel,
            removed_bids=removed_bids,
        )

        record = detect_transitions(self.entry_axtree, after_axtree)
        block = serialize_observation_transitions(
            record, include_removed=False, include_modified=False,
        )
        return f"{base}\n\n{block}" if base and block else (base or block)
