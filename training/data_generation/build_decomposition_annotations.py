"""Build decomposition annotations from batch output.

Merges 1st pass + fallback results into per-page annotation files.

Input:
  - data/api_requests/decomposition_output.jsonl (1st pass)
  - data/api_requests/decomposition_fallback_output.jsonl (optional, fallback)
  - data/axtrees/ (for URL mapping)

Output:
  - data/annotations/decomposition/{stem}.json per page

Usage:
    python training/data_generation/build_decomposition_annotations.py \
        --batch-output data/api_requests/decomposition_output.jsonl \
        --fallback-output data/api_requests/decomposition_fallback_output.jsonl \
        --axtree-dir data/axtrees/ \
        --output-dir data/annotations/decomposition/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_ids(content: str) -> List[int]:
    match = re.search(r"\[[\d\s,]+\]", content)
    if match:
        return [int(x) for x in re.findall(r"\d+", match.group())]
    return [int(x) for x in re.findall(r"\d+", content)]


def _parse_batch_output(path: str) -> Dict[str, List[int]]:
    """Parse batch output JSONL -> {custom_id: [region_root_ids]}."""
    results = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("response", {}).get("status_code") != 200:
                continue
            cid = item["custom_id"]
            content = item["response"]["body"]["choices"][0]["message"]["content"]
            results[cid] = _parse_ids(content)
    return results


def _load_url_mapping(axtree_dir: str, stems: set[str] | None = None) -> Dict[str, str]:
    """Build stem -> url mapping from AXTree files."""
    stem_to_url: Dict[str, str] = {}
    axtree_base = Path(axtree_dir)
    axtree_files = [f for f in axtree_base.rglob("*.json")
                     if f.name not in ("progress.json", "failed.json")
                     and (stems is None or f.stem in stems)]
    for axtree_file in tqdm(axtree_files, desc="Loading URLs"):
        try:
            with open(axtree_file) as f:
                data = json.load(f)
            stem_to_url[axtree_file.stem] = data.get("url", "")
        except Exception:
            pass
    return stem_to_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Build decomposition annotations")
    parser.add_argument("--batch-output", default="data/api_requests/decomposition_output.jsonl")
    parser.add_argument("--fallback-output", default=None,
                        help="Fallback batch output (optional)")
    parser.add_argument("--axtree-dir", default="data/axtrees/")
    parser.add_argument("--output-dir", default="data/annotations/decomposition/")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse 1st pass
    logger.info("Parsing 1st pass: %s", args.batch_output)
    first_pass = _parse_batch_output(args.batch_output)
    logger.info("  %d pages", len(first_pass))

    # Load URL mapping (only for pages in batch output)
    first_pass_stems = {cid.replace("decomp-", "") for cid in first_pass}
    stem_to_url = _load_url_mapping(args.axtree_dir, first_pass_stems)
    logger.info("  %d URL mappings", len(stem_to_url))

    # Parse fallback (keyed by "fallback-{stem}-{backend_id}")
    fallback: Dict[str, Dict[str, Any]] = {}
    if args.fallback_output and Path(args.fallback_output).exists():
        logger.info("Parsing fallback: %s", args.fallback_output)
        raw_fallback = _parse_batch_output(args.fallback_output)
        for cid, ids in raw_fallback.items():
            # cid = "fallback-{stem}-{backend_id}"
            parts = cid.split("-", 1)[1]  # "{stem}-{backend_id}"
            last_dash = parts.rfind("-")
            stem = parts[:last_dash]
            backend_id = int(parts[last_dash + 1:])
            fallback[stem] = {
                "region_backend_id": backend_id,
                "sub_ids": ids,
            }
        logger.info("  %d fallback pages", len(fallback))

    # Build annotations
    total = 0
    total_with_fallback = 0

    for cid, region_ids in tqdm(first_pass.items(), desc="Building annotations"):
        stem = cid.replace("decomp-", "")

        annotation: Dict[str, Any] = {
            "url": stem_to_url.get(stem, ""),
            "region_root_ids": region_ids,
            "fallback_applied": False,
        }

        if stem in fallback:
            fb = fallback[stem]
            annotation["fallback_applied"] = True
            annotation["fallback_region"] = fb["region_backend_id"]
            annotation["fallback_sub_ids"] = fb["sub_ids"]
            total_with_fallback += 1

        with open(out_dir / f"{stem}.json", "w") as f:
            json.dump(annotation, f, ensure_ascii=False)

        total += 1

    logger.info("Saved %d annotations (%d with fallback) -> %s",
                total, total_with_fallback, out_dir)


if __name__ == "__main__":
    main()
