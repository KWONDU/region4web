"""Generate decomposition training data via LLM.

Input: serialized AXTree per page
Output: region root node IDs -> edge labels
Fallback: 2nd pass on the largest region (by visible node count)

Modes:
  --mode direct  : Call API directly (default)
  --mode batch   : Generate JSONL for OpenAI Batch API

Usage:
    python training/data_generation/generate_decomposition_data.py \
        --input data/axtrees/ \
        --output data/decomposition/ \
        --config config.yaml \
        --mode direct
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load config.yaml for model settings."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _load_prompt_template(prompt_path: str) -> str:
    return Path(prompt_path).read_text(encoding="utf-8")


def _strip_url_scheme(url: str) -> str:
    """Strip https://www., https://, http://www., http:// and trailing slash from URL."""
    for prefix in ("https://www.", "http://www.", "https://", "http://"):
        if url.startswith(prefix):
            url = url[len(prefix):]
            break
    return url.rstrip("/")


def _build_messages(axtree_text: str, prompt_template: str, url: str = "") -> List[Dict[str, str]]:
    prompt = prompt_template.replace("{axtree}", axtree_text).replace("{url}", _strip_url_scheme(url))
    return [{"role": "user", "content": prompt}]


def _build_api_body(
    messages: List[Dict[str, str]],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build OpenAI-compatible request body from config."""
    from agent.llm_client import _parse_model_string
    _, model_name, _ = _parse_model_string(model_config.get("model", "openai/gpt-5.4-mini"))
    body: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    params = model_config.get("params", {})
    for key, value in params.items():
        body[key] = value
    return body


def _count_tokens_approx(text: str) -> int:
    """Approximate token count (words * 1.3)."""
    return int(len(text.split()) * 1.3)


def generate_batch_jsonl(
    axtree_dir: str,
    output_path: str,
    prompt_path: str,
    model_config: Dict[str, Any],
) -> None:
    """Generate JSONL file for OpenAI Batch API."""
    prompt_template = _load_prompt_template(prompt_path)
    axtree_files = sorted(Path(axtree_dir).rglob("*.json"))
    # Exclude non-data files
    axtree_files = [f for f in axtree_files if f.name not in ("progress.json", "failed.json")]

    total_input_tokens = 0
    total_requests = 0

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    with open(output, "w") as f:
        for i, axtree_file in enumerate(tqdm(axtree_files, desc="Generating batch")):
            with open(axtree_file) as af:
                data = json.load(af)

            # Build tree and serialize
            raw_nodes = data.get("raw_nodes", [])
            if not raw_nodes:
                continue

            from extraction.axtree import build_axtree
            from extraction.serialization import serialize_full_tree
            tree = build_axtree(raw_nodes)
            axtree_text = serialize_full_tree(tree)
            page_url = data.get("url", "")

            messages = _build_messages(axtree_text, prompt_template, page_url)
            body = _build_api_body(messages, model_config)

            request = {
                "custom_id": f"decomp-{axtree_file.stem}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

            total_input_tokens += _count_tokens_approx(axtree_text)
            total_requests += 1

    logger.info("Generated %d requests -> %s", total_requests, output)
    logger.info("Approx input tokens: %d", total_input_tokens)


def generate_direct(
    axtree_dir: str,
    output_dir: str,
    prompt_path: str,
    model_config: Dict[str, Any],
    fallback_prompt_path: Optional[str] = None,
) -> None:
    """Generate data by calling API directly."""
    from agent.llm_client import create_llm_client

    prompt_template = _load_prompt_template(prompt_path)
    client = create_llm_client(
        model_string=model_config.get("model", "openai/gpt-5.4-mini"),
        params=model_config.get("params", {}),
    )

    axtree_files = sorted(Path(axtree_dir).rglob("*.json"))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_input_tokens = 0
    total_output_tokens = 0
    total_pages = 0

    for i, axtree_file in enumerate(axtree_files):
        with open(axtree_file) as af:
            data = json.load(af)

        raw_nodes = data.get("raw_nodes", [])
        if not raw_nodes:
            continue

        from extraction.axtree import build_axtree
        from extraction.serialization import serialize_full_tree
        tree = build_axtree(raw_nodes)
        axtree_text = serialize_full_tree(tree)
        page_url = data.get("url", "")

        # 1st pass
        messages = _build_messages(axtree_text, prompt_template, page_url)
        prompt_text = messages[0]["content"]
        total_input_tokens += _count_tokens_approx(prompt_text)

        raw_output, _ = client.generate(prompt_text)
        total_output_tokens += _count_tokens_approx(raw_output)

        region_root_ids = _parse_ids(raw_output)

        # Verify coverage
        from framework.decomposition.partition import extract_regions
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
        covered = sum(len(r.member_node_ids) for r in partition.regions)

        # Fallback: 2nd pass on the largest region
        if fallback_prompt_path:
            from extraction.serialization import serialize_region_subtree
            fallback_template = _load_prompt_template(fallback_prompt_path)

            # Find the region with the most visible nodes
            largest_region = max(
                partition.regions,
                key=lambda r: len([
                    l for l in serialize_region_subtree(tree, r).split("\n") if l.strip()
                ]),
            )
            largest_text = serialize_region_subtree(tree, largest_region)
            largest_visible = len([l for l in largest_text.split("\n") if l.strip()])

            if largest_visible > 1:
                fallback_prompt = fallback_template.replace("{axtree}", largest_text)
                total_input_tokens += _count_tokens_approx(fallback_prompt)

                fallback_output, _ = client.generate(fallback_prompt)
                total_output_tokens += _count_tokens_approx(fallback_output)

                fallback_ids = _parse_ids(fallback_output)
                for rid in fallback_ids:
                    if rid in tree.backend_id_map:
                        nid = tree.backend_id_map[rid]
                        node = tree.nodes[nid]
                        if node.parent is not None:
                            cut_edges.add((node.parent.id, nid))

                # Re-partition with merged cut edges
                partition = extract_regions(tree, cut_edges)
                region_root_ids = region_root_ids + fallback_ids

        # Build edge labels
        region_root_set = {r.root_node_id for r in partition.regions}
        edge_labels = {}
        for node in tree.nodes.values():
            for child in node.children:
                edge_labels[f"{node.id}->{child.id}"] = (
                    1 if child.id in region_root_set else 0
                )

        result = {
            "url": data.get("url", ""),
            "axtree": axtree_text,
            "region_root_ids": region_root_ids,
            "edge_labels": edge_labels,
            "num_regions": len(partition.regions),
            "coverage": covered / max(len(tree.nodes), 1),
        }

        fname = f"page_{i:04d}.json"
        with open(out_dir / fname, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        total_pages += 1
        if (i + 1) % 50 == 0:
            logger.info("Processed %d/%d pages", i + 1, len(axtree_files))

    logger.info("Done. %d pages processed.", total_pages)
    logger.info("Token usage: input=%d, output=%d, total=%d",
                total_input_tokens, total_output_tokens,
                total_input_tokens + total_output_tokens)


def _parse_ids(llm_output: str) -> List[int]:
    """Parse region root IDs from LLM output."""
    import re
    match = re.search(r"\[[\d\s,]+\]", llm_output)
    if match:
        return json.loads(match.group())
    return [int(x) for x in re.findall(r"\d+", llm_output)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate decomposition training data")
    parser.add_argument("--input", default="data/axtrees/", help="AXTree JSON directory")
    parser.add_argument("--output", default="data/decomposition/", help="Output directory")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--prompt", default="prompts/data_generation/decomposition.txt", help="Prompt template")
    parser.add_argument("--mode", choices=["direct", "batch"], default="direct")
    parser.add_argument("--batch-output", default="data/api_requests/decomposition_input.jsonl")
    args = parser.parse_args()

    config = _load_config(args.config)
    model_config = config.get("training", {}).get("data_generation", {}).get("annotation", {})

    if args.mode == "batch":
        generate_batch_jsonl(args.input, args.batch_output, args.prompt, model_config)
    else:
        generate_direct(args.input, args.output, args.prompt, model_config)


if __name__ == "__main__":
    main()
