"""Format selection input/output."""

from __future__ import annotations

import re
from typing import List, Optional

from core.types import RegionRepr


def format_regions(region_reprs: List[RegionRepr]) -> str:
    """Format all regions' purpose + state summary for the selection prompt."""
    blocks: List[str] = []
    for rr in region_reprs:
        lines = [f"<R{rr.region_id}>", f"purpose: {rr.purpose}"]
        if rr.state_summary:
            lines.append(f"state summary: {rr.state_summary}")
        lines.append(f"</R{rr.region_id}>")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_selection_input(
    region_reprs: List[RegionRepr],
    task: str,
    action_history: Optional[List[str]] = None,
    prompt_template: str = "",
) -> str:
    """Assemble the full selection prompt."""
    regions_text = format_regions(region_reprs)

    if action_history:
        action_history_text = "\n".join(
            f"{i}. {a}" for i, a in enumerate(action_history, 1)
        )
    else:
        action_history_text = "(none)"

    if prompt_template:
        prompt = prompt_template.replace("{region_abstractions}", regions_text)
        prompt = prompt.replace("{task_instruction}", task)
        prompt = prompt.replace("{action_history}", action_history_text)
        return prompt

    # Fallback: simple prompt
    parts = [
        f"Task: {task}",
        "",
        "Action history:",
        action_history_text,
        "",
        "Regions:",
        regions_text,
        "",
        "Select the region IDs most relevant to the current task step. Output a JSON list of integers.",
    ]
    return "\n".join(parts)


def parse_region_ids(llm_output: str, valid_ids: List[int]) -> List[int]:
    """Parse region IDs from LLM output, filtering to valid IDs.

    The prompt asks for a trailing comma-separated list like ``R3, R7``.
    We drop any leading ``<think>...</think>`` chain-of-thought so that
    region references inside the reasoning are not mis-parsed as answers.
    Deduplicates while preserving first-seen order.
    """
    # Strip chain-of-thought: keep only text after the last </think>.
    tail = re.split(r"</think>\s*", llm_output, flags=re.IGNORECASE)[-1]

    # Prefer R-prefixed tokens (matches the format we ask for). Fall back to
    # bare integers only if the model answered without the "R" prefix.
    matches = re.findall(r"R(\d+)", tail)
    if not matches:
        matches = re.findall(r"\d+", tail)

    seen: List[int] = []
    for m in matches:
        try:
            rid = int(m)
        except ValueError:
            continue
        if rid in valid_ids and rid not in seen:
            seen.append(rid)
    return seen
