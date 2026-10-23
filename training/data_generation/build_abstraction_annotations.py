"""Build abstraction annotations from batch output.

Parses the abstraction batch output and stores per-page JSON files
with each region's purpose and state_summary.

Input:
  - data/api_requests/abstraction_output.jsonl
  - data/annotations/verification/*.json (for region metadata)

Output:
  - data/annotations/abstraction/{stem}.json per page

Usage:
    python training/data_generation/build_abstraction_annotations.py \
        --batch-output data/api_requests/abstraction_output.jsonl \
        --verification-dir data/annotations/verification/ \
        --output-dir data/annotations/abstraction/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_batch_output(path: str) -> Dict[str, Dict[str, str]]:
    """Parse batch output -> {custom_id: {purpose, state_summary}}."""
    results = {}
    parse_fail = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("response", {}).get("status_code") != 200:
                continue
            cid = item["custom_id"]
            content = item["response"]["body"]["choices"][0]["message"]["content"].strip()
            try:
                data = json.loads(content)
                results[cid] = {
                    "purpose": data.get("purpose", ""),
                    "state_summary": data.get("state_summary", ""),
                }
            except json.JSONDecodeError:
                parse_fail += 1
    if parse_fail:
        logger.warning("%d responses failed JSON parse", parse_fail)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build abstraction annotations")
    parser.add_argument("--batch-output", default="data/api_requests/abstraction_output.jsonl")
    parser.add_argument("--verification-dir", default="data/annotations/verification/")
    parser.add_argument("--output-dir", default="data/annotations/abstraction/")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse batch output
    logger.info("Parsing batch output: %s", args.batch_output)
    abstr_map = _parse_batch_output(args.batch_output)
    logger.info("  %d results", len(abstr_map))

    # Build per-page annotations from verification annotations
    ver_dir = Path(args.verification_dir)
    ver_files = sorted(ver_dir.glob("*.json"))

    total_pages = 0
    total_regions = 0
    missing = 0

    for ver_file in tqdm(ver_files, desc="Building abstraction annotations"):
        stem = ver_file.stem

        with open(ver_file) as f:
            ver_data = json.load(f)

        regions = []
        for r in ver_data.get("regions", []):
            rid = r["id"]
            cid = f"abstr-{stem}-r{rid}"

            abstr = abstr_map.get(cid)
            if abstr:
                regions.append({
                    "id": rid,
                    "root_backend_id": r["root_backend_id"],
                    "purpose": abstr["purpose"],
                    "state_summary": abstr["state_summary"],
                })
                total_regions += 1
            else:
                missing += 1

        annotation = {
            "url": ver_data.get("url", ""),
            "regions": regions,
        }

        with open(out_dir / f"{stem}.json", "w") as f:
            json.dump(annotation, f, ensure_ascii=False)

        total_pages += 1

    logger.info("Saved %d pages, %d regions (missing %d) -> %s",
                total_pages, total_regions, missing, out_dir)


if __name__ == "__main__":
    main()
