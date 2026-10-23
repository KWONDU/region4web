"""Generate abstraction batch input from verification annotations.

Reads verified region partitions and generates a batch JSONL where each
request asks the LLM to produce {purpose, state_summary} for one region.

Input:
  - data/annotations/verification/*.json (pages with pure valid regions)

Output:
  - data/api_requests/abstraction_input.jsonl

Usage:
    python training/data_generation/generate_abstraction_data.py \
        --annotations data/annotations/verification/ \
        --prompt prompts/data_generation/abstraction.txt \
        --config config.yaml \
        --output data/api_requests/abstraction_input.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> Dict[str, Any]:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "region_abstraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "purpose": {"type": "string"},
                "state_summary": {"type": "string"},
            },
            "required": ["purpose", "state_summary"],
            "additionalProperties": False,
        },
    },
}


def _build_api_body(
    messages: List[Dict[str, str]],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    from agent.llm_client import _parse_model_string
    _, model_name, _ = _parse_model_string(model_config.get("model", "openai/gpt-5-mini"))
    body: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "response_format": RESPONSE_FORMAT,
    }
    for key, value in model_config.get("params", {}).items():
        body[key] = value
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate abstraction batch input")
    parser.add_argument("--annotations", default="data/annotations/verification/")
    parser.add_argument("--prompt", default="prompts/data_generation/abstraction.txt")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="data/api_requests/abstraction_input.jsonl")
    args = parser.parse_args()

    config = _load_config(args.config)
    model_config = config.get("training", {}).get("data_generation", {}).get("annotation", {})
    prompt_template = Path(args.prompt).read_text(encoding="utf-8")

    ann_dir = Path(args.annotations)
    ann_files = sorted(ann_dir.glob("*.json"))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_requests = 0
    total_regions = 0

    with open(output, "w") as out_f:
        for ann_file in tqdm(ann_files, desc="Generating abstraction batch"):
            stem = ann_file.stem

            with open(ann_file) as f:
                data = json.load(f)

            for region in data.get("regions", []):
                subtree = region.get("subtree", "")
                if not subtree.strip():
                    continue

                prompt = prompt_template.replace("{region_axtree}", subtree)
                messages = [{"role": "user", "content": prompt}]
                body = _build_api_body(messages, model_config)

                request = {
                    "custom_id": f"abstr-{stem}-r{region['id']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                out_f.write(json.dumps(request, ensure_ascii=False) + "\n")
                total_requests += 1

            total_regions += len(data.get("regions", []))

    logger.info("Generated %d requests from %d pages (%d total regions) -> %s",
                total_requests, len(ann_files), total_regions, output)


if __name__ == "__main__":
    main()
