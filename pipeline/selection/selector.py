"""Task-relevant region selection."""

from __future__ import annotations

from typing import List, Optional, Protocol

from core.types import RegionRepr
from pipeline.selection.formatting import format_selection_input, parse_region_ids


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class RegionSelector:
    """Select task-relevant regions using an LLM.

    After each ``select()`` call, the full prompt + raw output + usage stats
    are available on ``self.last_call`` for the benchmark runner to dump to
    selections.jsonl (avoids changing the return signature used by the pipeline
    manager).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: str = "",
    ) -> None:
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.last_call: Optional[dict] = None

    def select(
        self,
        region_reprs: List[RegionRepr],
        task: str,
        action_history: Optional[List[str]] = None,
    ) -> List[int]:
        """Select relevant region IDs."""
        prompt = format_selection_input(
            region_reprs, task,
            action_history=action_history,
            prompt_template=self.prompt_template,
        )
        raw_output, usage = self.llm_client.generate(prompt)
        valid_ids = [rr.region_id for rr in region_reprs]
        selected = parse_region_ids(raw_output, valid_ids)

        fallback_used = False
        if not selected:
            selected = valid_ids
            fallback_used = True

        self.last_call = {
            "prompt": prompt,
            "raw_output": raw_output,
            "usage": usage,
            "parsed_region_ids": list(selected),
            "fallback_used": fallback_used,
        }
        return selected
