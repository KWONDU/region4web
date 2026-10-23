"""Generate verification batch input from decomposition annotations.

For each annotated page, builds the region partition in XML tag format
and creates a batch request for the verification prompt.

Input:
  - data/annotations/decomposition/*.json (annotations with region_root_ids)
  - data/axtrees/ (raw AXTree files)

Output:
  - data/api_requests/verification_input.jsonl (batch JSONL)

Usage:
    python training/data_generation/generate_verification_data.py \
        --annotations data/annotations/decomposition/ \
        --axtree-dir data/axtrees/ \
        --prompt prompts/data_generation/verification.txt \
        --config config.yaml \
        --output data/api_requests/verification_input.jsonl
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


def _strip_url_scheme(url: str) -> str:
    for prefix in ("https://www.", "http://www.", "https://", "http://"):
        if url.startswith(prefix):
            url = url[len(prefix):]
            break
    return url.rstrip("/")


def _build_api_body(
    messages: List[Dict[str, str]],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    from agent.llm_client import _parse_model_string
    _, model_name, _ = _parse_model_string(model_config.get("model", "openai/gpt-5-mini"))
    body: Dict[str, Any] = {"model": model_name, "messages": messages}
    for key, value in model_config.get("params", {}).items():
        body[key] = value
    return body


def _build_region_partition_text(tree, partition) -> str:
    """Serialize partition as XML-tagged region subtrees."""
    from extraction.serialization import serialize_region_subtree

    parts = []
    for region in partition.regions:
        text = serialize_region_subtree(tree, region)
        if text.strip():
            parts.append(f"<R{region.region_id}>\n{text}\n</R{region.region_id}>")
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate verification batch input")
    parser.add_argument("--annotations", default="data/annotations/decomposition/")
    parser.add_argument("--axtree-dir", default="data/axtrees/")
    parser.add_argument("--prompt", default="prompts/data_generation/verification.txt")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="data/api_requests/verification_input.jsonl")
    args = parser.parse_args()

    from extraction.axtree import build_axtree
    from framework.decomposition.partition import extract_regions

    config = _load_config(args.config)
    model_config = config.get("training", {}).get("data_generation", {}).get("verification", {})
    prompt_template = Path(args.prompt).read_text(encoding="utf-8")

    annotation_dir = Path(args.annotations)
    axtree_dir = Path(args.axtree_dir)
    annotation_files = sorted(annotation_dir.glob("*.json"))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_requests = 0
    skipped = 0

    with open(output, "w") as out_f:
        for ann_file in tqdm(annotation_files, desc="Generating verification batch"):
            stem = ann_file.stem

            # Load annotation
            with open(ann_file) as f:
                ann = json.load(f)

            region_root_ids = ann.get("region_root_ids", [])
            url = ann.get("url", "")

            # Merge fallback sub_ids if present
            if ann.get("fallback_applied") and ann.get("fallback_sub_ids"):
                region_root_ids = region_root_ids + ann["fallback_sub_ids"]

            # Find AXTree file
            matches = list(axtree_dir.rglob(f"{stem}.json"))
            if not matches:
                skipped += 1
                continue

            with open(matches[0]) as f:
                data = json.load(f)

            raw_nodes = data.get("raw_nodes", [])
            if not raw_nodes:
                skipped += 1
                continue

            # Build tree and partition
            tree = build_axtree(raw_nodes)

            valid_ids = []
            for rid in region_root_ids:
                if rid in tree.backend_id_map:
                    valid_ids.append(tree.backend_id_map[rid])

            cut_edges = set()
            for nid in valid_ids:
                node = tree.nodes[nid]
                if node.parent is not None:
                    cut_edges.add((node.parent.id, nid))

            partition = extract_regions(tree, cut_edges)

            # Build region partition text with XML tags
            region_text = _build_region_partition_text(tree, partition)
            if not region_text.strip():
                skipped += 1
                continue

            # Build prompt
            prompt = prompt_template.replace("{region_partition}", region_text)
            prompt = prompt.replace("{url}", _strip_url_scheme(url))

            messages = [{"role": "user", "content": prompt}]
            body = _build_api_body(messages, model_config)

            request = {
                "custom_id": f"verify-{stem}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            out_f.write(json.dumps(request, ensure_ascii=False) + "\n")
            total_requests += 1

    logger.info("Generated %d requests (skipped %d) -> %s", total_requests, skipped, output)


if __name__ == "__main__":
    main()
