"""Build verification annotations for pages with pure valid decomposition.

Filters to pages where all regions passed verification (invalid=0),
then stores each page's region partition with serialized subtrees.

Input:
  - data/api_requests/verification_output.jsonl (+ retry if exists)
  - data/annotations/decomposition/*.json
  - data/axtrees/

Output:
  - data/annotations/verification/{stem}.json per valid page

Usage:
    python training/data_generation/build_verification_annotations.py \
        --verification-output data/api_requests/verification_output.jsonl \
        --verification-retry data/api_requests/verification_retry_output.jsonl \
        --decomposition-dir data/annotations/decomposition/ \
        --axtree-dir data/axtrees/ \
        --output-dir data/annotations/verification/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Set

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_verification_outputs(*paths: str) -> Set[str]:
    """Return stems where all regions are valid (output = 'none')."""
    valid_stems: Set[str] = set()
    total = 0

    for path in paths:
        if not Path(path).exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("response", {}).get("status_code") != 200:
                    continue
                stem = item["custom_id"].replace("verify-", "")
                content = item["response"]["body"]["choices"][0]["message"]["content"].strip().lower()
                total += 1
                if content in ("none", "none."):
                    valid_stems.add(stem)

    logger.info("Verification: %d total, %d valid (%.1f%%)",
                total, len(valid_stems), len(valid_stems) / max(total, 1) * 100)
    return valid_stems


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verification annotations")
    parser.add_argument("--verification-output", default="data/api_requests/verification_output.jsonl")
    parser.add_argument("--verification-retry", default="data/api_requests/verification_retry_output.jsonl")
    parser.add_argument("--decomposition-dir", default="data/annotations/decomposition/")
    parser.add_argument("--axtree-dir", default="data/axtrees/")
    parser.add_argument("--output-dir", default="data/annotations/verification/")
    args = parser.parse_args()

    from extraction.axtree import build_axtree
    from extraction.serialization import serialize_region_subtree
    from framework.decomposition.partition import extract_regions

    # Step 1: Find valid stems
    valid_stems = _parse_verification_outputs(args.verification_output, args.verification_retry)

    # Step 2: Build annotations for valid pages
    decomp_dir = Path(args.decomposition_dir)
    axtree_dir = Path(args.axtree_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0

    for stem in tqdm(sorted(valid_stems), desc="Building verification annotations"):
        # Load decomposition annotation
        ann_path = decomp_dir / f"{stem}.json"
        if not ann_path.exists():
            skipped += 1
            continue

        with open(ann_path) as f:
            ann = json.load(f)

        url = ann.get("url", "")
        region_root_ids = ann.get("region_root_ids", [])
        if ann.get("fallback_applied") and ann.get("fallback_sub_ids"):
            region_root_ids = region_root_ids + ann["fallback_sub_ids"]

        # Load AXTree
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

        # Build partition
        tree = build_axtree(raw_nodes)
        valid_ids = [tree.backend_id_map[rid] for rid in region_root_ids if rid in tree.backend_id_map]
        cut_edges = set()
        for nid in valid_ids:
            node = tree.nodes[nid]
            if node.parent is not None:
                cut_edges.add((node.parent.id, nid))

        partition = extract_regions(tree, cut_edges)

        # Serialize each region
        regions = []
        for region in partition.regions:
            subtree = serialize_region_subtree(tree, region)
            if not subtree.strip():
                continue
            regions.append({
                "id": region.region_id,
                "root_backend_id": region.root_backend_id,
                "subtree": subtree,
            })

        result = {
            "url": url,
            "regions": regions,
        }

        with open(out_dir / f"{stem}.json", "w") as f:
            json.dump(result, f, ensure_ascii=False)

        total += 1

    logger.info("Saved %d annotations (skipped %d) -> %s", total, skipped, out_dir)


if __name__ == "__main__":
    main()
