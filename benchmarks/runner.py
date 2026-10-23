"""Benchmark runner (supplementary release).

Two modes:
  * ``backbone``   — pure LLM baseline. The full AXTree is fed straight to the
                     actor, no decomposition / abstraction / selection.
  * ``pagedigest`` — our framework. Per page entry: decomposition → SLM
                     abstraction → relevance selection → render compact
                     <Rx purpose="..."> blocks (selected regions show their
                     full subtree; others show the opening tag only). The
                     actor may emit ``view_all()`` to re-expose every region
                     for the rest of the page when it gets stuck.

Usage:
    python benchmarks/runner.py --config config.yaml \\
        --task-list benchmarks/task_lists/full.txt \\
        --mode pagedigest
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", module="beartype")

import yaml

from core.pricing import calculate_cost

from benchmarks.actor import Actor
from benchmarks.trajectory import (
    AbstractionDump,
    ActorAttempt,
    ActorDump,
    AxtreeDump,
    DecompositionDump,
    RegionDump,
    RegionReprDump,
    SelectionDump,
    StepSummary,
    TaskSummary,
    TrajectoryLogger,
    TransitionDump,
    TransitionEntryDump,
    build_run_dir,
)
from benchmarks.webarena import configure_webarena_env


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Map user-facing mode names to the internal observation_mode string used by
# pipeline.manager / observation_builder. ``backbone`` is the no-pipeline path
# (no manager is built); only ``pagedigest`` ever lands in the manager.
_MODE_CHOICES = ("backbone", "pagedigest")
_INTERNAL_OBS_MODE = {"pagedigest": "digest_page"}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _load_task_list(path: str) -> List[str]:
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


# ---------------------------------------------------------------------------
# Actor / pipeline factories
# ---------------------------------------------------------------------------


def _build_actor(config: Dict[str, Any], action_space_description: str, mode: str) -> Actor:
    bc = config.get("benchmark", {})
    lc = bc.get("llm", {})
    ac = bc.get("actor", {})

    from agent.llm_client import create_llm_client
    llm = create_llm_client(
        model_string=lc.get("model"),
        params=lc.get("params", {}),
        base_url=lc.get("base_url"),
    )

    prompt_path = ac.get("prompt_path", "prompts/inference/agent.txt")
    prompt_template = Path(prompt_path).read_text(encoding="utf-8")
    od = ac.get("obs_description", {}) or {}
    obs_desc = od.get(mode, "")
    prompt_template = prompt_template.replace("{obs_description}", obs_desc)

    return Actor(
        llm_client=llm,
        prompt_template=prompt_template,
        action_space_description=action_space_description,
        max_retries=ac.get("max_llm_retries", 1),
        max_prompt_tokens=bc.get("max_prompt_tokens"),
    )


def _load_partitioner(config: Dict[str, Any]):
    import torch
    from framework.decomposition.model import RegionPartitioner

    fc = config.get("framework", {}).get("decomposition", {})
    mc = fc.get("model", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegionPartitioner(
        num_roles=mc.get("num_roles"),
        role_embed_dim=mc.get("role_embed_dim", 11),
        hidden_dim=mc.get("hidden_dim", 256),
        num_layers=mc.get("num_layers", 3),
    ).to(device)
    ckpt = torch.load(fc["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _load_representor():
    from framework.abstraction.representor import load_representor
    return load_representor()


def _load_selector(config: Dict[str, Any]):
    from pipeline.selection.selector import RegionSelector

    bc = config.get("benchmark", {})
    lc = bc.get("llm", {})
    sc = bc.get("selection", {})
    prompt_path = sc.get("prompt_path", "prompts/inference/selection.txt")

    from agent.llm_client import create_llm_client
    llm = create_llm_client(
        model_string=lc.get("model"),
        params=lc.get("params", {}),
        base_url=lc.get("base_url"),
    )
    prompt_template = Path(prompt_path).read_text(encoding="utf-8")
    return RegionSelector(llm, prompt_template)


def _build_observation_manager(config: Dict[str, Any], mode: str):
    """``backbone`` returns None (full AXTree fed to actor unchanged).
    ``pagedigest`` builds the full pipeline (decomposition + abstraction +
    selection) with observation_mode='digest_page'."""
    if mode == "backbone":
        return None

    from pipeline.manager import ObservationManager
    partitioner = _load_partitioner(config)
    representor = _load_representor()
    selector = _load_selector(config)
    threshold = config.get("framework", {}).get("decomposition", {}).get("threshold", 0.5)
    return ObservationManager(
        partitioner=partitioner,
        representor=representor,
        selector=selector,
        threshold=threshold,
        observation_mode=_INTERNAL_OBS_MODE[mode],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _short_url(url: str, maxlen: int = 70) -> str:
    if len(url) <= maxlen:
        return url
    return url[:maxlen - 3] + "..."


def _inject_view_all_action(description: str) -> str:
    """Append ``view_all()`` description into browsergym's action-set text.

    The action is intercepted by the runner before env.step; browsergym
    itself does not know about it. Source lives in
    ``prompts/action_spaces/webarena/view_all.txt``.
    """
    src = Path("prompts/action_spaces/webarena/view_all.txt")
    if not src.exists():
        return description
    body = src.read_text(encoding="utf-8").rstrip("\n")
    anchor = "Only a single action can be provided at once."
    if anchor not in description:
        return description + "\n" + body + "\n"
    out = re.sub(
        r"(\d+) different types of actions are available\.",
        lambda m: f"{int(m.group(1)) + 1} different types of actions are available.",
        description, count=1,
    )
    return out.replace(anchor, body + "\n\n" + anchor, 1)


def _override_send_msg_examples(description: str) -> str:
    """Replace browsergym's verbose send_msg_to_user example with concise
    value-only examples aligned with WebArena's string_match evaluator."""
    base = Path("prompts/action_spaces")
    old_path = base / "send_msg_old.txt"
    new_path = base / "send_msg_new.txt"
    if not (old_path.exists() and new_path.exists()):
        return description
    old = old_path.read_text(encoding="utf-8").rstrip("\n")
    new = new_path.read_text(encoding="utf-8").rstrip("\n")
    return description.replace(old, new)


def _enrich_axtree_nodes(raw_nodes: List[Dict[str, Any]], page: Any) -> None:
    """Fill DOM src/href on unnamed image/link nodes (parity with training)."""
    from extraction.axtree import enrich_raw_nodes_from_dom
    cdp = page.context.new_cdp_session(page)
    try:
        enrich_raw_nodes_from_dom(raw_nodes, cdp)
    finally:
        cdp.detach()


def _concat_selected_content(observation: Any) -> str:
    if observation is None:
        return ""
    skeleton = getattr(observation, "selected_text", "") or ""
    if skeleton:
        return skeleton
    parts: List[str] = []
    for rid in observation.selected_region_ids:
        content = observation.selected_content.get(rid, "")
        if content:
            parts.append(content)
    return "\n".join(parts)


def _unescape_rebuilt_tree(tree: Any) -> None:
    """Reverse the repr-round-trip artifact left by ``_rebuild_tree_from_text``
    so node names match the training-time escape level (single backslash) for
    every PUA / unicode-escaped name."""
    import ast

    def _try(s: str) -> str:
        if not s or "\\" not in s:
            return s
        for q in ("'", '"'):
            if q in s:
                continue
            try:
                return ast.literal_eval(f"{q}{s}{q}")
            except (ValueError, SyntaxError):
                continue
        return s

    for n in tree.nodes.values():
        if n.name:
            n.name = _try(n.name)
        if n.value:
            n.value = _try(n.value)


# ---------------------------------------------------------------------------
# Dump builders
# ---------------------------------------------------------------------------


def _build_decomposition_dump(step, page_entry_idx, url, partition, tree, elapsed_ms) -> DecompositionDump:
    from extraction.serialization import serialize_region_subtree
    regions: List[RegionDump] = []
    for r in partition.regions:
        root = tree.nodes[r.root_node_id]
        regions.append(RegionDump(
            region_id=r.region_id,
            root_backend_id=root.backend_node_id,
            root_bid=root.bid,
            root_role=root.role,
            root_name=root.name,
            member_count=len(r.member_node_ids),
            subtree_text=serialize_region_subtree(tree, r),
        ))
    return DecompositionDump(
        step=step, page_entry_idx=page_entry_idx, url=url,
        regions=regions, total_regions=len(regions), elapsed_ms=elapsed_ms,
    )


def _build_abstraction_dump(step, page_entry_idx, url, region_reprs, elapsed_ms) -> AbstractionDump:
    dumps: List[RegionReprDump] = []
    for rr in region_reprs:
        meta = rr.metadata or {}
        dumps.append(RegionReprDump(
            region_id=rr.region_id,
            purpose=rr.purpose,
            state_summary=rr.state_summary,
            slm_prompt=meta.get("slm_prompt", ""),
            slm_raw_output=meta.get("slm_raw_output", ""),
        ))
    return AbstractionDump(
        step=step, page_entry_idx=page_entry_idx, url=url,
        region_reprs=dumps, total_slm_calls=len(dumps), elapsed_ms=elapsed_ms,
    )


_API_PROVIDER_PREFIXES: Tuple[str, ...] = (
    "openai/", "anthropic/", "deepseek/", "gemini/",
)


def _pricing_slug(model_string: str) -> Optional[str]:
    """Return the PRICING key for an API-backed backbone, or None for vLLM /
    local models so cost fields are deliberately omitted from the dump."""
    if not model_string or not any(model_string.startswith(p) for p in _API_PROVIDER_PREFIXES):
        return None
    return model_string.split("/", 1)[1]


def _build_selection_dump(step, page_entry_idx, url, selection_call, elapsed_ms,
                          pricing_slug: Optional[str] = None) -> Optional[SelectionDump]:
    if not selection_call:
        return None
    usage = dict(selection_call.get("usage", {}))
    cost = calculate_cost(usage, pricing_slug) if pricing_slug else None
    return SelectionDump(
        step=step, page_entry_idx=page_entry_idx, url=url,
        prompt=selection_call.get("prompt", ""),
        raw_output=selection_call.get("raw_output", ""),
        parsed_region_ids=list(selection_call.get("parsed_region_ids", [])),
        fallback_used=bool(selection_call.get("fallback_used", False)),
        llm_usage=usage,
        elapsed_ms=elapsed_ms,
        llm_cost=cost,
    )


def _build_transition_dump(step, url, trans_rec, serialize_fn, history_size) -> TransitionDump:
    entries = [
        TransitionEntryDump(
            change_type=e.change_type,
            backend_node_id=e.backend_node_id,
            role=e.role, name=e.name, detail=e.detail,
        )
        for e in trans_rec.entries
    ]
    return TransitionDump(
        step=step, url=url,
        delta_text=serialize_fn([trans_rec]) if trans_rec.entries else "",
        entries=entries,
        cumulative_history_size=history_size,
    )


def _build_actor_dump(step, records, think, action_parsed, elapsed_ms,
                      pricing_slug: Optional[str] = None) -> ActorDump:
    attempts: List[ActorAttempt] = []
    for r in records:
        usage = dict(r.usage)
        cost = calculate_cost(usage, pricing_slug) if pricing_slug else None
        attempts.append(ActorAttempt(
            attempt=r.attempt, prompt=r.prompt, raw_output=r.raw_output,
            parse_ok=r.parse_ok, usage=usage, cost=cost,
        ))
    return ActorDump(
        step=step, attempts=attempts, think=think,
        action_parsed=action_parsed, elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Single-task run
# ---------------------------------------------------------------------------


def run_single_task(
    task_name: str, seed: int, config: Dict[str, Any], mode: str,
    action_mapping: Callable[[str], str], actor: Actor, obs_mgr: Any,
) -> TaskSummary:
    import gymnasium as gym

    bc = config["benchmark"]
    wa = bc["webarena"]
    max_steps = wa.get("max_steps", 30)

    env = gym.make(
        f"browsergym/{task_name}",
        action_mapping=action_mapping,
        max_episode_steps=max_steps,
    )

    output_dir = Path(bc.get("output_dir", "results/"))
    backbone = bc.get("llm", {}).get("model", "")
    run_dir = build_run_dir(
        output_dir, "webarena", backbone, "base", mode, task_name,
    )
    logger_ = TrajectoryLogger(run_dir)
    logger_.write_config({
        "framework": config.get("framework", {}),
        "benchmark": config.get("benchmark", {}),
        "_runtime": {"mode": mode},
    })

    try:
        return _run_task_loop(
            env=env, task_name=task_name, seed=seed,
            config=config, mode=mode, max_steps=max_steps,
            actor=actor, obs_mgr=obs_mgr, logger_=logger_,
        )
    finally:
        logger_.close()
        env.close()


def _run_task_loop(
    env: Any, task_name: str, seed: int, config: Dict[str, Any],
    mode: str, max_steps: int, actor: Actor, obs_mgr: Any,
    logger_: TrajectoryLogger,
) -> TaskSummary:
    from extraction.axtree import build_axtree
    from extraction.serialization import serialize_full_tree
    from pipeline.transition import serialize_transitions
    from training.train_decomposition import _rebuild_tree_from_text

    bc = config["benchmark"]
    backbone_model = bc.get("llm", {}).get("model", "")
    pricing_slug = _pricing_slug(backbone_model)
    running_cost: Optional[Dict[str, Any]] = None
    if pricing_slug is not None:
        running_cost = {
            "total": 0.0,
            "by_component": defaultdict(float),
            "in_tokens": 0,
            "cached_tokens": 0,
            "out_tokens": 0,
        }

    if obs_mgr is not None:
        obs_mgr.reset_for_new_task()

    t_task_start = time.time()
    obs, info = env.reset(seed=seed)
    task_cfg = getattr(env.unwrapped.task, "config", {}) or {}
    logger_.write_meta({
        "task_name": task_name,
        "seed": seed,
        "goal": obs.get("goal", ""),
        "sites": task_cfg.get("sites", []),
        "eval_types": (task_cfg.get("eval") or {}).get("eval_types", []),
        "start_url": task_cfg.get("start_url", ""),
        "mode": mode,
        "backbone": backbone_model,
        "max_steps": max_steps,
    })

    action_history: List[str] = []
    last_url: Optional[str] = None
    last_action: str = ""
    page_entry_counter = 0
    total_reward = 0.0
    terminated = False
    truncated = False
    exit_reason = "truncated"
    step_idx = 0

    for step_idx in range(max_steps):
        url_t = obs.get("url", "")
        goal_t = obs.get("goal", "")
        logger.info("  [step %d/%d] url=%s", step_idx, max_steps, _short_url(url_t))

        # 1. AXTree extraction + rebuild (training-distribution match)
        raw_nodes = obs["axtree_object"].get("nodes", [])
        _enrich_axtree_nodes(raw_nodes, env.unwrapped.page)
        raw_tree = build_axtree(raw_nodes)
        full_axtree_text = serialize_full_tree(raw_tree)
        tree = _rebuild_tree_from_text(full_axtree_text)
        _unescape_rebuilt_tree(tree)
        logger.info("    AXTree: %d nodes (raw %d)", len(tree.nodes), len(raw_tree.nodes))
        logger_.log_axtree(AxtreeDump(
            step=step_idx, full_axtree_text=full_axtree_text,
            num_tree_nodes=len(tree.nodes),
        ))

        # 2. Pipeline (page entry by URL identity) — only when obs_mgr is on
        page_entry = (last_url or "") != url_t
        decomp_ms = abs_ms = sel_ms = None

        if obs_mgr is None:
            observation_text = full_axtree_text
            logger.info("    [backbone] passing full AXTree to actor (%d chars)",
                        len(full_axtree_text))
        else:
            if page_entry:
                page_entry_counter += 1
                logger.info("    [page entry #%d] running decomp → abstract → select",
                            page_entry_counter)
                observation = obs_mgr.on_url_change(url_t, tree, goal_t)
                diag = obs_mgr.last_diagnostics
                decomp_ms = diag.get("decomposition_ms")
                abs_ms = diag.get("abstraction_ms")
                sel_ms = diag.get("selection_ms")
                n_regions = len(observation.region_reprs)
                n_selected = len(observation.selected_region_ids)
                logger.info("      decomp %dms → %d regions | abstract %dms | select %dms → %d/%d regions",
                            decomp_ms or 0, n_regions, abs_ms or 0, sel_ms or 0,
                            n_selected, n_regions)

                logger_.log_decomposition(_build_decomposition_dump(
                    step_idx, page_entry_counter - 1, url_t,
                    obs_mgr.current_partition, tree, decomp_ms or 0,
                ))
                logger_.log_abstraction(_build_abstraction_dump(
                    step_idx, page_entry_counter - 1, url_t,
                    observation.region_reprs, abs_ms or 0,
                ))
                sel_dump = _build_selection_dump(
                    step_idx, page_entry_counter - 1, url_t,
                    diag.get("selection_call"), sel_ms or 0,
                    pricing_slug=pricing_slug,
                )
                if sel_dump is not None:
                    logger_.log_selection(sel_dump)
                    if running_cost is not None and sel_dump.llm_cost:
                        running_cost["total"] += sel_dump.llm_cost["total_cost"]
                        running_cost["by_component"]["selection_input"]  += sel_dump.llm_cost["input_cost"]
                        running_cost["by_component"]["selection_cached"] += sel_dump.llm_cost["cached_input_cost"]
                        running_cost["by_component"]["selection_output"] += sel_dump.llm_cost["output_cost"]
                    if running_cost is not None:
                        u = sel_dump.llm_usage
                        running_cost["in_tokens"]     += u.get("prompt_tokens", 0) or 0
                        running_cost["cached_tokens"] += (u.get("cached_tokens", 0) or u.get("prompt_cache_hit_tokens", 0) or 0)
                        running_cost["out_tokens"]    += u.get("completion_tokens", 0) or 0
            else:
                trans_rec = obs_mgr.on_action_performed(last_action, tree)
                logger.info("    [same page] transition diff: %d changes",
                            len(trans_rec.entries))
                logger_.log_transition(_build_transition_dump(
                    step_idx, url_t, trans_rec, serialize_transitions,
                    len(obs_mgr.current_observation.transition_history)
                    if obs_mgr.current_observation else 0,
                ))

            parts: List[str] = []
            selected_text = _concat_selected_content(obs_mgr.current_observation)
            if selected_text:
                parts.append(selected_text)
            observation_text = "\n\n".join(parts) if parts else ""

        # 3. Actor
        logger.info("    [actor] calling LLM (observation %d chars, %d prior actions)",
                    len(observation_text), len(action_history))
        t_actor0 = time.time()
        action_parsed, think, records = actor.act(
            task=goal_t,
            observation=observation_text,
            action_history=action_history,
        )
        actor_ms = int((time.time() - t_actor0) * 1000)
        total_tokens = sum(r.usage.get("total_tokens", 0) for r in records)
        logger.info("      actor %dms → %s (attempts=%d, tokens=%d)",
                    actor_ms, (action_parsed or "<parse_fail>")[:80],
                    len(records), total_tokens)
        actor_dump = _build_actor_dump(
            step_idx, records, think, action_parsed, actor_ms,
            pricing_slug=pricing_slug,
        )
        logger_.log_actor(actor_dump)
        if running_cost is not None:
            for a in actor_dump.attempts:
                if a.cost:
                    running_cost["total"] += a.cost["total_cost"]
                    running_cost["by_component"]["actor_input"]  += a.cost["input_cost"]
                    running_cost["by_component"]["actor_cached"] += a.cost["cached_input_cost"]
                    running_cost["by_component"]["actor_output"] += a.cost["output_cost"]
                running_cost["in_tokens"]     += a.usage.get("prompt_tokens", 0) or 0
                running_cost["cached_tokens"] += (a.usage.get("cached_tokens", 0) or a.usage.get("prompt_cache_hit_tokens", 0) or 0)
                running_cost["out_tokens"]    += a.usage.get("completion_tokens", 0) or 0

        # 4. Env step (with view_all() interception in pagedigest)
        action_executed = False
        action_error: Optional[str] = None
        reward = 0.0
        t_env0 = time.time()
        if action_parsed is None:
            action_error = "parse_fail_after_retries"
            exit_reason = "parse_fail"
            terminated = True
            logger.warning("    [env] skipped — parse failed after retries")
        elif action_parsed.strip() == "view_all()":
            if obs_mgr is None:
                action_error = "view_all() requires pagedigest mode"
                exit_reason = "error"
                terminated = True
                logger.error("    [env] view_all() not allowed in backbone mode")
            else:
                try:
                    obs_mgr.on_view_all(tree, goal_t)
                    action_executed = True
                    action_history.append(action_parsed)
                    logger.info("    [view_all] all regions re-exposed; transitions baseline reset")
                except Exception as e:
                    action_error = f"{type(e).__name__}: {e}"
                    exit_reason = "error"
                    terminated = True
                    logger.error("    [view_all] failed: %s", action_error)
        else:
            try:
                obs, reward, terminated, truncated, info = env.step(action_parsed)
                action_executed = True
                action_history.append(action_parsed)
                total_reward += reward
                logger.info("    [env] reward=%.2f term=%s trunc=%s cum_reward=%.2f",
                            reward, terminated, truncated, total_reward)
            except Exception as e:
                action_error = f"{type(e).__name__}: {e}"
                exit_reason = "error"
                terminated = True
                logger.error("    [env] step failed: %s", action_error)
        env_step_ms = int((time.time() - t_env0) * 1000)

        last_url = url_t
        if action_parsed is not None:
            last_action = action_parsed

        logger_.log_step(StepSummary(
            step=step_idx, url=url_t,
            page_entry=page_entry,
            page_entry_idx=(page_entry_counter - 1) if page_entry and obs_mgr else None,
            action_parsed=action_parsed,
            action_executed=action_executed,
            action_error=action_error,
            reward=reward, terminated=terminated, truncated=truncated,
            observation_chars=len(observation_text),
            env_step_ms=env_step_ms,
            decomposition_ms=decomp_ms,
            abstraction_ms=abs_ms,
            selection_ms=sel_ms,
            actor_ms=actor_ms,
        ))

        if terminated or truncated:
            break

    if exit_reason == "truncated":
        exit_reason = "success" if total_reward > 0 else (
            "truncated" if truncated else "terminated"
        )
    final_answer: Optional[str] = None
    for act in reversed(action_history):
        m = re.match(r"send_msg_to_user\s*\(\s*(['\"])(.*)\1\s*\)\s*$", act or "", re.DOTALL)
        if m:
            final_answer = m.group(2)
            break

    cost_kwargs: Dict[str, Any] = {}
    if running_cost is not None:
        cost_kwargs = {
            "total_cost_usd":      round(running_cost["total"], 6),
            "cost_by_component":   {k: round(v, 6) for k, v in running_cost["by_component"].items()},
            "total_input_tokens":  running_cost["in_tokens"],
            "total_cached_tokens": running_cost["cached_tokens"],
            "total_output_tokens": running_cost["out_tokens"],
        }

    summary = TaskSummary(
        task_id=task_name, seed=seed,
        success=total_reward > 0,
        final_reward=total_reward,
        num_steps=step_idx + 1,
        num_page_entries=page_entry_counter,
        terminated=terminated, truncated=truncated,
        wall_time_s=time.time() - t_task_start,
        exit_reason=exit_reason,
        goal=obs.get("goal", ""),
        reference_answers=(task_cfg.get("eval") or {}).get("reference_answers"),
        final_answer=final_answer,
        **cost_kwargs,
    )
    logger_.write_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="WebArena benchmark runner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--backbone", default=None,
                        help="Backbone name; loads configs/<name>.yaml and deep-merges into --config")
    parser.add_argument("--task-list", default=None,
                        help="Override benchmark.webarena.task_list")
    parser.add_argument("--mode", default=None, choices=_MODE_CHOICES,
                        help="backbone (pure LLM baseline) or pagedigest (framework)")
    parser.add_argument("--task-slice", default=None,
                        help="Run subset, e.g. '0:10' for first 10 tasks")
    for site in ("shopping", "shopping-admin", "reddit", "gitlab", "wikipedia", "map", "homepage"):
        parser.add_argument(f"--url-{site}", default=None,
                            help=f"Override benchmark.webarena.urls.{site.replace('-', '_')}")
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.backbone:
        _deep_merge(config, _load_config(f"configs/{args.backbone}.yaml"))

    secrets_path = Path(config.get("secrets_file", "secrets.yaml"))
    if secrets_path.exists():
        secrets = yaml.safe_load(secrets_path.read_text(encoding="utf-8")) or {}
        for secret_key, env_var in (
            ("openai_api_key", "OPENAI_API_KEY"),
            ("anthropic_api_key", "ANTHROPIC_API_KEY"),
            ("huggingface_api_key", "HF_TOKEN"),
        ):
            val = secrets.get(secret_key)
            if val and not os.environ.get(env_var):
                os.environ[env_var] = val

    bc = config.setdefault("benchmark", {})
    wa = bc.setdefault("webarena", {})
    urls = wa.setdefault("urls", {})

    for site in ("shopping", "shopping_admin", "reddit", "gitlab", "wikipedia", "map", "homepage"):
        v = getattr(args, f"url_{site}", None)
        if v is not None:
            urls[site] = v

    mode = args.mode or bc.get("mode", "pagedigest")
    if mode not in _MODE_CHOICES:
        raise SystemExit(f"unknown mode {mode!r}; expected one of {_MODE_CHOICES}")
    bc["mode"] = mode
    task_list_path = args.task_list or wa.get("task_list", "benchmarks/task_lists/full.txt")

    tasks = _load_task_list(task_list_path)
    if args.task_slice:
        start, end = map(int, args.task_slice.split(":"))
        tasks = tasks[start:end]

    from browsergym.core.action.highlevel import HighLevelActionSet
    as_cfg = bc.get("action_set", {})
    action_set = HighLevelActionSet(
        subsets=as_cfg.get("subsets", ["webarena"]),
        strict=as_cfg.get("strict", False),
        multiaction=as_cfg.get("multiaction", False),
    )
    action_space_description = action_set.describe()
    action_space_description = _override_send_msg_examples(action_space_description)
    if mode == "pagedigest":
        action_space_description = _inject_view_all_action(action_space_description)

    configure_webarena_env(wa.get("urls", {}))
    import browsergym.webarena  # noqa: F401

    # libwebarena's llm_fuzzy_match / llm_ua_match call generate_from_openai_chat_completion
    # with a hardcoded gpt-4-1106-preview which OpenAI retired. Rewrite the model arg
    # so the evaluator keeps working with identical prompts/temperature.
    from webarena.evaluation_harness import helper_functions as _wa_helper
    _orig_gen = _wa_helper.generate_from_openai_chat_completion
    def _patched_gen(*args, **kwargs):
        if kwargs.get("model") == "gpt-4-1106-preview":
            kwargs["model"] = "gpt-4o"
        return _orig_gen(*args, **kwargs)
    _wa_helper.generate_from_openai_chat_completion = _patched_gen

    actor = _build_actor(config, action_space_description, mode=mode)
    obs_mgr = _build_observation_manager(config, mode)

    logger.info("Running %d tasks | mode=%s | backbone=%s",
                len(tasks), mode, bc.get("llm", {}).get("model", "?"))

    seed = wa.get("seed", 42)
    for task_name in tasks:
        logger.info("→ %s", task_name)
        try:
            summary = run_single_task(
                task_name=task_name, seed=seed, config=config, mode=mode,
                action_mapping=action_set.to_python_code,
                actor=actor, obs_mgr=obs_mgr,
            )
            cost_str = (
                f"  cost=${summary.total_cost_usd:.4f}"
                if summary.total_cost_usd is not None else ""
            )
            logger.info("  %s  reward=%.2f  steps=%d  page_entries=%d%s  exit=%s",
                        "OK" if summary.success else "FAIL",
                        summary.final_reward, summary.num_steps,
                        summary.num_page_entries, cost_str, summary.exit_reason)
        except Exception as e:
            logger.exception("  run failed: %s", e)


if __name__ == "__main__":
    main()
