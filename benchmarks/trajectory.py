"""Multi-file per-run trajectory logger.

File layout (inside run_dir):
  config.json            run config snapshot (all knobs that produced this run)
  meta.json              task metadata (goal, start_url, sites, seed, ...)
  summary.json           task outcome on completion

  trajectory.jsonl       one per step — thin summary + cross-refs to other files
  axtrees.jsonl          one per step — serialized full AXTree
  decompositions.jsonl   one per page_entry — region partitioning output
  abstractions.jsonl     one per page_entry — per-region SLM purpose/state_summary + full SLM I/O
  selections.jsonl       one per page_entry — selection LLM prompt/output/parsed IDs
  transitions.jsonl      one per non-page_entry — within-page diff
  actor_calls.jsonl      one per step — full actor LLM prompt/output (all retry attempts)

All records start with ``{"step": N, ...}`` so files can be joined on step.
page_entry-only records also carry ``page_entry_idx`` (0-indexed within a run)
for grouping same-page events.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ----------------------------- Dataclasses ---------------------------------


@dataclass
class StepSummary:
    """Thin per-step row. Heavy payloads live in sibling JSONL files."""
    step: int
    url: str
    page_entry: bool
    page_entry_idx: Optional[int]            # null on within-page steps
    action_parsed: Optional[str]
    action_executed: bool
    action_error: Optional[str]
    reward: float
    terminated: bool
    truncated: bool
    observation_chars: int
    env_step_ms: int
    # Timing breakdown (from ObservationManager.last_diagnostics)
    decomposition_ms: Optional[int] = None
    abstraction_ms: Optional[int] = None
    selection_ms: Optional[int] = None
    actor_ms: Optional[int] = None


@dataclass
class AxtreeDump:
    step: int
    full_axtree_text: str
    num_tree_nodes: int


@dataclass
class RegionDump:
    region_id: int
    root_backend_id: int
    root_bid: str
    root_role: str
    root_name: str
    member_count: int
    subtree_text: str


@dataclass
class DecompositionDump:
    step: int
    page_entry_idx: int
    url: str
    regions: List[RegionDump]
    total_regions: int
    elapsed_ms: int


@dataclass
class RegionReprDump:
    region_id: int
    purpose: str
    state_summary: str
    slm_prompt: str
    slm_raw_output: str


@dataclass
class AbstractionDump:
    step: int
    page_entry_idx: int
    url: str
    region_reprs: List[RegionReprDump]
    total_slm_calls: int
    elapsed_ms: int


@dataclass
class SelectionDump:
    step: int
    page_entry_idx: int
    url: str
    prompt: str
    raw_output: str
    parsed_region_ids: List[int]
    fallback_used: bool
    llm_usage: Dict[str, Any]
    elapsed_ms: int
    llm_cost: Optional[Dict[str, float]] = None     # API runs only; None for vLLM (→ dropped from dump)


@dataclass
class TransitionEntryDump:
    change_type: str
    backend_node_id: int
    role: str
    name: str
    detail: str = ""


@dataclass
class TransitionDump:
    step: int
    url: str
    delta_text: str
    entries: List[TransitionEntryDump]
    cumulative_history_size: int


@dataclass
class ActorAttempt:
    attempt: int
    prompt: str
    raw_output: str
    parse_ok: bool
    usage: Dict[str, Any]
    cost: Optional[Dict[str, float]] = None         # API runs only; None for vLLM (→ dropped from dump)


@dataclass
class ActorDump:
    step: int
    attempts: List[ActorAttempt]         # every LLM call for this step (incl. retries)
    think: str                            # chain-of-thought from final attempt
    action_parsed: Optional[str]
    elapsed_ms: int


@dataclass
class TaskSummary:
    task_id: str
    seed: int
    success: bool
    final_reward: float
    num_steps: int
    num_page_entries: int
    terminated: bool
    truncated: bool
    wall_time_s: float
    exit_reason: str                      # "success" | "truncated" | "terminated" | "parse_fail" | "error"
    goal: str = ""                        # task intent (what the user asked)
    reference_answers: Optional[Dict[str, Any]] = None  # WebArena ground truth (exact_match / must_include / fuzzy_match)
    final_answer: Optional[str] = None    # text passed to send_msg_to_user on the last stop action

    # Aggregated cost / token counts. All Optional — stay None for runs where
    # the backbone has no PRICING entry (vLLM / unknown). None fields are
    # stripped by the logger so summary.json doesn't carry misleading zeros.
    total_cost_usd:      Optional[float]           = None
    cost_by_component:   Optional[Dict[str, float]] = None    # {"actor_input": .., "actor_cached": .., "actor_output": .., "selection_*": ..}
    total_input_tokens:  Optional[int]             = None
    total_cached_tokens: Optional[int]             = None
    total_output_tokens: Optional[int]             = None


# ------------------------------ Logger -------------------------------------


def _drop_none(obj: Any) -> Any:
    """Recursively strip ``None`` values from dicts / lists.

    Used to keep cost / token-count fields *absent* from the on-disk JSONL
    when we did not compute them (e.g. vLLM runs where PRICING returns None).
    An absent key is semantically different from ``null`` or ``0``: it means
    "not applicable", not "free" or "failed".
    """
    if isinstance(obj, dict):
        return {k: _drop_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_none(x) for x in obj]
    return obj


_JSONL_FILES = (
    "trajectory",
    "axtrees",
    "decompositions",
    "abstractions",
    "selections",
    "transitions",
    "actor_calls",
)


class TrajectoryLogger:
    """Writes one run's outputs to the standard directory layout."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._files = {
            name: (self.run_dir / f"{name}.jsonl").open("w")
            for name in _JSONL_FILES
        }

    # ---- Top-level metadata files ----

    def write_config(self, config: Dict[str, Any]) -> None:
        (self.run_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False, default=str),
        )

    def write_meta(self, meta: Dict[str, Any]) -> None:
        (self.run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str),
        )

    def write_summary(self, summary: TaskSummary) -> None:
        (self.run_dir / "summary.json").write_text(
            json.dumps(_drop_none(asdict(summary)), indent=2, ensure_ascii=False, default=str),
        )

    # ---- Per-step appends ----

    def _append(self, kind: str, obj: Any) -> None:
        data = asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
        data = _drop_none(data)
        f = self._files[kind]
        f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
        f.flush()

    def log_step(self, s: StepSummary) -> None:
        self._append("trajectory", s)

    def log_axtree(self, d: AxtreeDump) -> None:
        self._append("axtrees", d)

    def log_decomposition(self, d: DecompositionDump) -> None:
        self._append("decompositions", d)

    def log_abstraction(self, d: AbstractionDump) -> None:
        self._append("abstractions", d)

    def log_selection(self, d: SelectionDump) -> None:
        self._append("selections", d)

    def log_transition(self, d: TransitionDump) -> None:
        self._append("transitions", d)

    def log_actor(self, d: ActorDump) -> None:
        self._append("actor_calls", d)

    def close(self) -> None:
        for f in self._files.values():
            f.close()

    def __enter__(self) -> "TrajectoryLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ------------------------ Path composition ---------------------------------


def _backbone_slug(model_string: str) -> str:
    """Friendly directory name for a backbone LLM id.

    Drops the org prefix for readability (meta-llama/Llama-3-70B-Instruct →
    Llama-3-70B-Instruct) and escapes anything that's still path-unfriendly.
    """
    slug = model_string.rsplit("/", 1)[-1]
    return slug.replace(":", "_")


def build_run_dir(
    output_dir: Path,
    benchmark: str,
    backbone: str,
    method: str,
    pipeline_mode: str,
    task_id: str,
) -> Path:
    """Per-run directory.

    Layout: <output_dir>/<benchmark>/<backbone>/<method>/<pipeline>/<task_id>/
    e.g.    results/webarena/Llama-3-70B-Instruct/base/full/webarena.0/
    """
    return (
        Path(output_dir)
        / benchmark
        / _backbone_slug(backbone)
        / method
        / pipeline_mode
        / task_id
    )
