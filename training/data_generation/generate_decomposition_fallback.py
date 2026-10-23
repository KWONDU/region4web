"""Generate fallback decomposition batch input.

Reads the initial batch output, identifies regions that need further
decomposition, and generates a new batch JSONL.

Fallback trigger criteria:
  - Region has >= 50% of total visible nodes AND
  - Region is >= 10x the median region size

Usage:
    python training/data_generation/generate_decomposition_fallback.py \
        --batch-output data/api_requests/decomposition_output.jsonl \
        --axtree-dir data/axtrees/ \
        --prompt prompts/data_generation/decomposition.txt \
        --config config.yaml \
        --output data/api_requests/decomposition_fallback_input.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FALLBACK_FRACTION = 0.5   # region >= 50% of total visible nodes
FALLBACK_RATIO = 10.0     # region >= 10x median region size


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


def _parse_ids(content: str) -> List[int]:
    match = re.search(r"\[[\d\s,]+\]", content)
    if match:
        return [int(x) for x in re.findall(r"\d+", match.group())]
    return [int(x) for x in re.findall(r"\d+", content)]


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


def find_fallback_regions(
    batch_output_path: str,
    axtree_dir: str,
    fraction_threshold: float = FALLBACK_FRACTION,
    ratio_threshold: float = FALLBACK_RATIO,
) -> List[Tuple[str, Path, int]]:
    """Find (custom_id, axtree_path, region_root_backend_id) tuples needing fallback."""

    from extraction.axtree import build_axtree
    from extraction.serialization import serialize_full_tree, serialize_region_subtree
    from framework.decomposition.partition import extract_regions

    with open(batch_output_path) as f:
        results = [json.loads(l) for l in f if l.strip()]

    axtree_base = Path(axtree_dir)
    fallback_targets = []
    total_checked = 0
    total_triggered = 0

    from tqdm import tqdm

    for item in tqdm(results, desc="Finding fallback targets"):
        if item.get("response", {}).get("status_code") != 200:
            continue

        cid = item["custom_id"]
        stem = cid.replace("decomp-", "")

        # Find AXTree file
        matches = list(axtree_base.rglob(f"{stem}.json"))
        if not matches:
            continue

        axtree_path = matches[0]
        with open(axtree_path) as f:
            data = json.load(f)

        raw_nodes = data.get("raw_nodes", [])
        if not raw_nodes:
            continue

        tree = build_axtree(raw_nodes)
        total_visible = len([l for l in serialize_full_tree(tree).split("\n") if l.strip()])

        # Parse region IDs from LLM output
        content = item["response"]["body"]["choices"][0]["message"]["content"]
        ids = _parse_ids(content)

        valid_ids = []
        for rid in ids:
            if rid in tree.backend_id_map:
                valid_ids.append(tree.backend_id_map[rid])

        cut_edges = set()
        for nid in valid_ids:
            node = tree.nodes[nid]
            if node.parent is not None:
                cut_edges.add((node.parent.id, nid))

        partition = extract_regions(tree, cut_edges)
        if len(partition.regions) < 2:
            continue

        # Compute visible node count per region
        region_visible = []
        for r in partition.regions:
            text = serialize_region_subtree(tree, r)
            visible = len([l for l in text.split("\n") if l.strip()])
            region_visible.append((r, visible))

        total_checked += 1
        median_size = float(np.median([v for _, v in region_visible]))

        # Check fallback criteria
        for region, visible in region_visible:
            fraction = visible / max(total_visible, 1)
            ratio = visible / max(median_size, 1)

            if fraction >= fraction_threshold and ratio >= ratio_threshold:
                root_node = tree.nodes[region.root_node_id]
                fallback_targets.append((cid, axtree_path, root_node.backend_node_id))
                total_triggered += 1
                break  # one fallback per page

    logger.info("Checked %d pages, %d triggered fallback (%.1f%%)",
                total_checked, total_triggered,
                total_triggered / max(total_checked, 1) * 100)

    return fallback_targets


def generate_fallback_jsonl(
    fallback_targets: List[Tuple[str, Path, int]],
    prompt_path: str,
    model_config: Dict[str, Any],
    output_path: str,
) -> None:
    """Generate batch JSONL for fallback decomposition."""

    from extraction.axtree import build_axtree
    from extraction.serialization import serialize_region_subtree

    prompt_template = Path(prompt_path).read_text(encoding="utf-8")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_requests = 0

    from tqdm import tqdm

    with open(output, "w") as out_f:
        for cid, axtree_path, target_backend_id in tqdm(fallback_targets, desc="Generating fallback batch"):
            with open(axtree_path) as f:
                data = json.load(f)

            raw_nodes = data.get("raw_nodes", [])
            if not raw_nodes:
                continue

            tree = build_axtree(raw_nodes)
            page_url = _strip_url_scheme(data.get("url", ""))

            # Re-parse original partition to get the target region
            # We need to rebuild the partition from the original batch output
            # For simplicity, find the region containing the target backend_id as root
            if target_backend_id not in tree.backend_id_map:
                continue

            target_nid = tree.backend_id_map[target_backend_id]
            target_node = tree.nodes[target_nid]

            # Build a subtree serialization from this node downward
            # Create a temporary region covering the full subtree
            from core.types import Region
            member_ids = set()

            def _collect(node):
                member_ids.add(node.id)
                for child in node.children:
                    _collect(child)

            _collect(target_node)

            region = Region(
                region_id=0,
                root_node_id=target_nid,
                root_backend_id=target_backend_id,
                member_node_ids=member_ids,
            )

            subtree_text = serialize_region_subtree(tree, region)
            if not subtree_text.strip():
                continue

            # Build prompt
            prompt = prompt_template.replace("{axtree}", subtree_text).replace("{url}", page_url)
            messages = [{"role": "user", "content": prompt}]
            body = _build_api_body(messages, model_config)

            request = {
                "custom_id": f"fallback-{cid.replace('decomp-', '')}-{target_backend_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            out_f.write(json.dumps(request, ensure_ascii=False) + "\n")
            total_requests += 1

    logger.info("Generated %d fallback requests -> %s", total_requests, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fallback decomposition batch")
    parser.add_argument("--batch-output", default="data/api_requests/decomposition_output.jsonl")
    parser.add_argument("--axtree-dir", default="data/axtrees/")
    parser.add_argument("--prompt", default="prompts/data_generation/decomposition.txt")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="data/api_requests/decomposition_fallback_input.jsonl")
    parser.add_argument("--fraction", type=float, default=FALLBACK_FRACTION,
                        help="Fallback trigger: min fraction of total visible nodes")
    parser.add_argument("--ratio", type=float, default=FALLBACK_RATIO,
                        help="Fallback trigger: min ratio vs median region size")
    args = parser.parse_args()

    fraction = args.fraction
    ratio = args.ratio

    config = _load_config(args.config)
    model_config = config.get("training", {}).get("data_generation", {}).get("annotation", {})

    logger.info("Finding fallback targets (fraction >= %.2f, ratio >= %.1f)...",
                fraction, ratio)

    targets = find_fallback_regions(args.batch_output, args.axtree_dir,
                                    fraction_threshold=fraction, ratio_threshold=ratio)

    if not targets:
        logger.info("No fallback targets found.")
        return

    generate_fallback_jsonl(targets, args.prompt, model_config, args.output)


if __name__ == "__main__":
    main()
