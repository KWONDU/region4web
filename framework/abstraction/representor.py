"""Semantic Abstraction: interpret each region as purpose + state summary.

Uses a fine-tuned SLM (e.g. Qwen3-0.6B) to generate JSON {purpose, state_summary}
for each region's serialized AXTree subtree.

Model and checkpoint configured via config.yaml:
    framework.abstraction.base_model
    framework.abstraction.checkpoint_path
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import yaml

from core.types import RegionPartition, RegionRepr, Tree
from extraction.serialization import serialize_region_subtree


def load_prompt_template(path: str) -> str:
    """Load a prompt template file."""
    return Path(path).read_text(encoding="utf-8")


class RegionRepresentor:
    """SLM-based semantic abstraction: generates purpose + state summary per region."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prompt_template: str,
        batch_size: int = 16,
        max_new_tokens: int = 256,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        # Left-padding is required for decoder-only batched generation so the
        # generated tokens come from the correct position per row.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def represent_all(
        self, partition: RegionPartition, tree: Tree,
    ) -> List[RegionRepr]:
        """Generate representations for all regions, batched.

        Runs the SLM on ``batch_size`` regions at a time rather than one by
        one. Keeps the same metadata fields (slm_prompt / slm_raw_output /
        subtree_text) per region.
        """
        import torch

        regions = list(partition.regions)
        subtrees = [serialize_region_subtree(tree, r) for r in regions]
        prompts = [self.prompt_template.replace("{region_axtree}", s) for s in subtrees]

        outputs_text: List[str] = [""] * len(prompts)
        for start in range(0, len(prompts), self.batch_size):
            chunk_prompts = prompts[start:start + self.batch_size]
            enc = self.tokenizer(
                chunk_prompts, return_tensors="pt",
                padding=True, truncation=False,
            ).to(self.model.device)
            with torch.no_grad():
                out_ids = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            input_len = enc["input_ids"].shape[1]
            generated_ids = out_ids[:, input_len:]
            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for i, text in enumerate(decoded):
                outputs_text[start + i] = text

        results: List[RegionRepr] = []
        for region, prompt, subtree_text, generated in zip(
            regions, prompts, subtrees, outputs_text,
        ):
            purpose, state_summary = _parse_output(generated)
            results.append(RegionRepr(
                region_id=region.region_id,
                purpose=purpose,
                state_summary=state_summary,
                metadata={
                    "slm_prompt": prompt,
                    "slm_raw_output": generated,
                    "subtree_text": subtree_text,
                },
            ))
        return results


def load_representor(
    config_path: str = "config.yaml",
    prompt_path: str = "prompts/data_generation/abstraction.txt",
) -> RegionRepresentor:
    """Load a RegionRepresentor from config.

    Uses the same prompt used in SFT training (prompts/data_generation/abstraction.txt)
    since the SLM was fine-tuned to that exact formatting.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    abs_config = config.get("framework", {}).get("abstraction", {})
    base_model = abs_config.get("base_model", "Qwen/Qwen3-0.6B")
    checkpoint_path = abs_config.get("checkpoint_path", "")
    batch_size = int(abs_config.get("batch_size", 16))

    model_path = checkpoint_path if checkpoint_path and Path(checkpoint_path).exists() else base_model

    # transformers 4.57 logs a Mistral-regex warning for Qwen3 tokenizers even
    # though the issue doesn't apply — silence it to keep the runner log clean.
    import logging as _logging
    _logging.getLogger("transformers.tokenization_utils_base").setLevel(_logging.ERROR)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    prompt_template = load_prompt_template(prompt_path)
    return RegionRepresentor(
        model, tokenizer, prompt_template, batch_size=batch_size,
    )


def _parse_output(raw: str) -> tuple[str, str]:
    """Parse SLM output into (purpose, state_summary).

    Expects JSON: {"purpose": "...", "state_summary": "..."}
    Falls back to treating entire output as purpose if parsing fails.
    """
    try:
        data = json.loads(raw.strip())
        return data.get("purpose", raw), data.get("state_summary", "")
    except (json.JSONDecodeError, AttributeError):
        return raw.strip(), ""
