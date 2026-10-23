"""Build final training data files for decomposition and abstraction.

Both files use pages whose partition has no invalid region (invalid=0).
``--decomp-max-invalid`` can relax this for decomposition data.

Output:
  - data/decomposition.jsonl (1 line = 1 page)
  - data/abstraction.jsonl (1 line = 1 region)

Usage:
    python training/data_generation/build_training_data.py \
        --verification-output data/api_requests/verification_output.jsonl \
        --verification-retry data/api_requests/verification_retry_output.jsonl \
        --verification-dir data/annotations/verification/ \
        --abstraction-dir data/annotations/abstraction/ \
        --decomposition-dir data/annotations/decomposition/ \
        --axtree-dir data/axtrees/ \
        --output-dir data/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Set

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_invalid_counts(*paths: str) -> Dict[str, int]:
    """Parse verification outputs -> {stem: invalid_count}."""
    result: Dict[str, int] = {}
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
                content = item["response"]["body"]["choices"][0]["message"]["content"].strip()
                if content.lower() in ("none", "none."):
                    result[stem] = 0
                else:
                    result[stem] = len(re.findall(r"R\d+", content, re.IGNORECASE))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final training data")
    parser.add_argument("--verification-output", default="data/api_requests/verification_output.jsonl")
    parser.add_argument("--verification-retry", default="data/api_requests/verification_retry_output.jsonl")
    parser.add_argument("--verification-dir", default="data/annotations/verification/")
    parser.add_argument("--abstraction-dir", default="data/annotations/abstraction/")
    parser.add_argument("--decomposition-dir", default="data/annotations/decomposition/")
    parser.add_argument("--axtree-dir", default="data/axtrees/")
    parser.add_argument("--output-dir", default="data/")
    parser.add_argument("--decomp-max-invalid", type=int, default=0,
                        help="Max invalid regions for decomposition data (default: 0)")
    args = parser.parse_args()

    from extraction.axtree import build_axtree
    from extraction.serialization import serialize_full_tree

    ver_dir = Path(args.verification_dir)
    abs_dir = Path(args.abstraction_dir)
    decomp_dir = Path(args.decomposition_dir)
    axtree_dir = Path(args.axtree_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse verification results for invalid counts
    invalid_counts = _parse_invalid_counts(args.verification_output, args.verification_retry)
    logger.info("Verification results: %d pages", len(invalid_counts))

    # Decomposition: invalid <= max_invalid
    decomp_stems: Set[str] = {s for s, c in invalid_counts.items() if c <= args.decomp_max_invalid}
    logger.info("Decomposition pages (invalid<=%d): %d", args.decomp_max_invalid, len(decomp_stems))

    # Abstraction: invalid = 0 only (from verification annotations)
    abstr_stems: Set[str] = {f.stem for f in abs_dir.glob("*.json")}
    logger.info("Abstraction pages (invalid=0): %d", len(abstr_stems))

    # Load abstraction annotations
    abs_map: Dict[str, Dict[int, Dict[str, str]]] = {}
    for f in abs_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        region_map = {}
        for r in data.get("regions", []):
            region_map[r["id"]] = {
                "purpose": r["purpose"],
                "state_summary": r["state_summary"],
            }
        abs_map[f.stem] = region_map

    # Build both files
    decomp_out = out_dir / "decomposition.jsonl"
    abstr_out = out_dir / "abstraction.jsonl"

    decomp_count = 0
    abstr_count = 0
    skipped = 0

    with open(decomp_out, "w") as d_f, open(abstr_out, "w") as a_f:
        # All stems to process (union of decomp and abstr)
        all_stems = decomp_stems | abstr_stems
        for stem in tqdm(sorted(all_stems), desc="Building training data"):
            # Load decomposition annotation
            decomp_path = decomp_dir / f"{stem}.json"
            if not decomp_path.exists():
                skipped += 1
                continue
            with open(decomp_path) as f:
                decomp_data = json.load(f)

            # Load AXTree
            matches = list(axtree_dir.rglob(f"{stem}.json"))
            if not matches:
                skipped += 1
                continue
            with open(matches[0]) as f:
                raw_data = json.load(f)
            raw_nodes = raw_data.get("raw_nodes", [])
            if not raw_nodes:
                skipped += 1
                continue

            url = decomp_data.get("url", "")
            region_root_ids = decomp_data.get("region_root_ids", [])
            if decomp_data.get("fallback_applied") and decomp_data.get("fallback_sub_ids"):
                region_root_ids = region_root_ids + decomp_data["fallback_sub_ids"]

            # --- decomposition.jsonl (invalid <= max_invalid) ---
            if stem in decomp_stems:
                tree = build_axtree(raw_nodes)
                axtree_text = serialize_full_tree(tree)

                d_f.write(json.dumps({
                    "uid": stem,
                    "url": url,
                    "axtree": axtree_text,
                    "region_root_backend_ids": region_root_ids,
                }, ensure_ascii=False) + "\n")
                decomp_count += 1

            # --- abstraction.jsonl (invalid = 0 only) ---
            if stem in abstr_stems:
                ver_path = ver_dir / f"{stem}.json"
                if ver_path.exists():
                    with open(ver_path) as f:
                        ver_data = json.load(f)
                    abstr_regions = abs_map.get(stem, {})
                    for r in ver_data.get("regions", []):
                        rid = r["id"]
                        abstr = abstr_regions.get(rid, {})
                        a_f.write(json.dumps({
                            "uid": stem,
                            "region_id": rid,
                            "root_backend_id": r["root_backend_id"],
                            "subtree": r["subtree"],
                            "purpose": abstr.get("purpose", ""),
                            "state_summary": abstr.get("state_summary", ""),
                        }, ensure_ascii=False) + "\n")
                        abstr_count += 1

    logger.info("decomposition.jsonl: %d pages (skipped %d) -> %s", decomp_count, skipped, decomp_out)
    logger.info("abstraction.jsonl: %d regions -> %s", abstr_count, abstr_out)


if __name__ == "__main__":
    main()
