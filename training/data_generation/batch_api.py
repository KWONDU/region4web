"""OpenAI Batch API: split, upload, create, poll, download.

Generic tool for any batch API job (decomposition, verification, abstraction, etc.).

Usage:
    # Split large JSONL into <200MB parts
    python training/data_generation/batch_api.py split --input data/api_requests/decomposition_input.jsonl

    # Upload and start batch jobs
    python training/data_generation/batch_api.py upload --input data/api_requests/decomposition_input.jsonl

    # Check status
    python training/data_generation/batch_api.py status --input data/api_requests/decomposition_input.jsonl

    # Download results (auto-merges parts)
    python training/data_generation/batch_api.py download --input data/api_requests/decomposition_input.jsonl

    # Cancel a batch
    python training/data_generation/batch_api.py cancel --batch-id batch_xxx

File naming convention:
    Input:  {name}.jsonl
    Parts:  {name}_part0.jsonl, {name}_part1.jsonl, ...
    Output: {name}_output.jsonl (merged)
    State:  {name}_state.json (tracks batch IDs)
    Errors: {name}_errors_part0.jsonl, ...

Limits (OpenAI Batch API):
    - Max file size: 200MB per file
    - Max requests: 50,000 per batch
    - Completion window: 24 hours
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 190 * 1024 * 1024  # 190MB (safety margin from 200MB limit)
MAX_REQUESTS = 50_000


def _get_client() -> OpenAI:
    """Load OpenAI client from secrets.yaml or env."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            import yaml
            with open("secrets.yaml") as f:
                secrets = yaml.safe_load(f) or {}
            api_key = secrets.get("openai_api_key", "")
        except FileNotFoundError:
            pass
    if not api_key:
        logger.error("No OpenAI API key found (secrets.yaml or OPENAI_API_KEY)")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def _derive_paths(input_path: Path) -> Dict[str, Any]:
    """Derive all file paths from the input JSONL path."""
    stem = input_path.stem  # e.g. decomposition_input
    parent = input_path.parent
    return {
        "input": input_path,
        "parent": parent,
        "stem": stem,
        "state": parent / f"{stem}_state.json",
        "output": parent / f"{stem.replace('_input', '_output')}.jsonl",
    }


def _load_state(state_path: Path) -> Dict[str, Any]:
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {"batches": []}


def _save_state(state: Dict[str, Any], state_path: Path) -> None:
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def _find_parts(parent: Path, stem: str) -> List[Path]:
    return sorted(parent.glob(f"{stem}_part*.jsonl"))


# ---- Commands ----

def split(args: argparse.Namespace) -> None:
    """Split a large JSONL file into parts under 200MB / 50K requests."""
    input_path = Path(args.input)
    paths = _derive_paths(input_path)

    file_size = input_path.stat().st_size
    with open(input_path) as f:
        lines = f.readlines()

    if file_size <= MAX_FILE_SIZE and len(lines) <= MAX_REQUESTS:
        logger.info("File is within limits (%dMB, %d requests). No split needed.",
                     file_size // (1024*1024), len(lines))
        # Still create part0 for consistency
        part0 = paths["parent"] / f"{paths['stem']}_part0.jsonl"
        if not part0.exists():
            import shutil
            shutil.copy(input_path, part0)
            logger.info("Copied -> %s", part0)
        return

    chunks: List[List[str]] = []
    current: List[str] = []
    current_size = 0

    for line in lines:
        line_size = len(line.encode("utf-8"))
        if (current_size + line_size > MAX_FILE_SIZE) or (len(current) >= MAX_REQUESTS):
            chunks.append(current)
            current = []
            current_size = 0
        current.append(line)
        current_size += line_size

    if current:
        chunks.append(current)

    for i, chunk in enumerate(chunks):
        out_path = paths["parent"] / f"{paths['stem']}_part{i}.jsonl"
        with open(out_path, "w") as f:
            f.writelines(chunk)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        logger.info("Part %d: %d requests, %.1fMB -> %s", i, len(chunk), size_mb, out_path)

    logger.info("Split into %d parts", len(chunks))


def upload(args: argparse.Namespace) -> None:
    """Upload parts and create batch jobs."""
    input_path = Path(args.input)
    paths = _derive_paths(input_path)
    parts = _find_parts(paths["parent"], paths["stem"])

    if not parts:
        logger.info("No parts found. Running split first...")
        split(args)
        parts = _find_parts(paths["parent"], paths["stem"])

    client = _get_client()
    state = _load_state(paths["state"])
    existing = {b["part"] for b in state["batches"]}

    for part_path in parts:
        part_name = part_path.stem
        if part_name in existing:
            logger.info("Skipping %s (already uploaded)", part_name)
            continue

        size_mb = part_path.stat().st_size / 1024 / 1024
        logger.info("Uploading %s (%.1fMB)...", part_path.name, size_mb)
        with open(part_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        logger.info("File ID: %s", file_obj.id)

        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        logger.info("Batch ID: %s, status: %s", batch.id, batch.status)

        state["batches"].append({
            "part": part_name,
            "file_id": file_obj.id,
            "batch_id": batch.id,
            "status": batch.status,
        })
        _save_state(state, paths["state"])

    logger.info("Done. Run 'status' to check progress.")


def status(args: argparse.Namespace) -> None:
    """Check status of all batch jobs."""
    input_path = Path(args.input)
    paths = _derive_paths(input_path)
    state = _load_state(paths["state"])

    if not state["batches"]:
        logger.info("No batches found. Run 'upload' first.")
        return

    client = _get_client()

    for entry in state["batches"]:
        batch = client.batches.retrieve(entry["batch_id"])
        entry["status"] = batch.status

        counts = ""
        if batch.request_counts:
            counts = f" (completed={batch.request_counts.completed}, failed={batch.request_counts.failed}, total={batch.request_counts.total})"

        logger.info("%s: %s%s", entry["part"], batch.status, counts)

        if batch.output_file_id:
            entry["output_file_id"] = batch.output_file_id
        if batch.error_file_id:
            entry["error_file_id"] = batch.error_file_id

    _save_state(state, paths["state"])

    all_done = all(e["status"] in ("completed", "failed", "cancelled") for e in state["batches"])
    if all_done:
        logger.info("All batches finished. Run 'download' to get results.")


def download(args: argparse.Namespace) -> None:
    """Download results and merge into single output file."""
    input_path = Path(args.input)
    paths = _derive_paths(input_path)
    state = _load_state(paths["state"])

    if not state["batches"]:
        logger.info("No batches found.")
        return

    client = _get_client()

    for entry in state["batches"]:
        batch = client.batches.retrieve(entry["batch_id"])
        entry["status"] = batch.status

        if batch.status != "completed":
            logger.info("%s: status=%s, skipping", entry["part"], batch.status)
            continue

        if batch.output_file_id:
            part_idx = entry["part"].rsplit("part", 1)[-1]
            out_path = paths["parent"] / f"{paths['stem'].replace('_input', '_output')}_part{part_idx}.jsonl"

            if out_path.exists() and not args.force:
                logger.info("%s: already downloaded -> %s", entry["part"], out_path)
                continue

            logger.info("Downloading %s output...", entry["part"])
            content = client.files.content(batch.output_file_id).text
            with open(out_path, "w") as f:
                f.write(content)
            lines = [l for l in content.strip().split("\n") if l]
            logger.info("Saved %d results -> %s", len(lines), out_path)

        if batch.error_file_id:
            err_path = paths["parent"] / f"{paths['stem'].replace('_input', '_errors')}_part{entry['part'].rsplit('part', 1)[-1]}.jsonl"
            logger.info("Downloading %s errors...", entry["part"])
            content = client.files.content(batch.error_file_id).text
            with open(err_path, "w") as f:
                f.write(content)

    _save_state(state, paths["state"])

    # Merge
    output_stem = paths["stem"].replace("_input", "_output")
    part_files = sorted(paths["parent"].glob(f"{output_stem}_part*.jsonl"))
    if part_files:
        total = 0
        with open(paths["output"], "w") as out:
            for p in part_files:
                with open(p) as f:
                    for line in f:
                        if line.strip():
                            out.write(line)
                            total += 1
        logger.info("Merged %d results -> %s", total, paths["output"])


def cancel(args: argparse.Namespace) -> None:
    """Cancel a batch job."""
    client = _get_client()
    batch = client.batches.cancel(args.batch_id)
    logger.info("Cancelled %s, status: %s", args.batch_id, batch.status)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI Batch API tool")
    parser.add_argument("--input", "-i", help="Input JSONL file path")

    sub = parser.add_subparsers(dest="command")

    sp = sub.add_parser("split", help="Split large JSONL into parts under 200MB")
    sp.add_argument("--input", "-i", required=True, help="Input JSONL file")

    up = sub.add_parser("upload", help="Upload parts and create batch jobs")
    up.add_argument("--input", "-i", required=True, help="Input JSONL file")

    st = sub.add_parser("status", help="Check batch status")
    st.add_argument("--input", "-i", required=True, help="Input JSONL file")

    dl = sub.add_parser("download", help="Download results")
    dl.add_argument("--input", "-i", required=True, help="Input JSONL file")
    dl.add_argument("--force", action="store_true", help="Re-download existing files")

    cn = sub.add_parser("cancel", help="Cancel a batch")
    cn.add_argument("--batch-id", required=True)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    elif args.command == "split":
        split(args)
    elif args.command == "upload":
        upload(args)
    elif args.command == "status":
        status(args)
    elif args.command == "download":
        download(args)
    elif args.command == "cancel":
        cancel(args)


if __name__ == "__main__":
    main()
