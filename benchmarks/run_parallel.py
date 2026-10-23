"""Parallel WebArena runner.

Splits the input task list by single-site domain, then spawns one
``benchmarks.runner`` subprocess per domain in parallel. All other arguments
are forwarded verbatim to the runner, including ``--mode`` (backbone or
pagedigest), ``--config``, ``--backbone``, and any ``--url-*`` overrides.

Usage:
  python -m benchmarks.run_parallel \\
      --task-list benchmarks/task_lists/full.txt \\
      --backbone gpt-5.1 --mode pagedigest

When ``--mode pagedigest`` is used, parallelism is capped at the number of
visible GPUs (one runner per GPU). ``--mode backbone`` does not load any
local models, so all selected domains run in parallel by default.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

DOMAINS = ("shopping", "shopping_admin", "reddit", "gitlab", "map")

# Candidate locations for WebArena's task dataset (used to look up each
# task's ``sites`` so we can split the list by single-site domain).
_WA_TASK_SOURCE_CANDIDATES = [
    ".webarena-verified/lib/python3.11/site-packages/webarena_verified/assets/dataset/test.raw.json",
    ".conda/lib/python3.11/site-packages/webarena/test.raw.json",
    ".conda/lib/python3.10/site-packages/webarena/test.raw.json",
]


def _load_task_sites() -> Dict[str, List[str]]:
    src = next((p for p in _WA_TASK_SOURCE_CANDIDATES if Path(p).exists()), None)
    if src is None:
        sys.exit(
            "WebArena task dataset not found. Looked at:\n  "
            + "\n  ".join(_WA_TASK_SOURCE_CANDIDATES)
        )
    out: Dict[str, List[str]] = {}
    data = json.load(open(src))
    for d in data:
        out[f"webarena.{d['task_id']}"] = d.get("sites") or []
    return out


def _split_task_list(task_list_path: str, domains: Tuple[str, ...]) -> Dict[str, List[str]]:
    task_sites = _load_task_sites()
    tasks = [t.strip() for t in Path(task_list_path).read_text().splitlines() if t.strip()]
    by_domain: Dict[str, List[str]] = {d: [] for d in domains}
    skipped_multi = skipped_unknown = skipped_other = 0
    for t in tasks:
        sites = task_sites.get(t)
        if sites is None:
            skipped_unknown += 1; continue
        if len(sites) != 1:
            skipped_multi += 1; continue
        s = sites[0]
        if s not in by_domain:
            skipped_other += 1; continue
        by_domain[s].append(t)
    if skipped_multi or skipped_unknown or skipped_other:
        print(f"[parallel] dropped {skipped_multi} multi-site, "
              f"{skipped_unknown} unknown, {skipped_other} out-of-scope tasks",
              file=sys.stderr)
    return by_domain


def _visible_gpu_ids() -> List[str]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return [x.strip() for x in cvd.split(",") if x.strip()]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()
        return [x.strip() for x in out if x.strip()]
    except Exception:
        return []


def _mode_from(forward_args: List[str]) -> str:
    for i, a in enumerate(forward_args):
        if a == "--mode" and i + 1 < len(forward_args):
            return forward_args[i + 1]
        if a.startswith("--mode="):
            return a.split("=", 1)[1]
    return "pagedigest"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--task-list", required=True,
                        help="Input task list (one webarena.N per line). Will be split by domain.")
    parser.add_argument("--max-parallel", type=int, default=-1,
                        help="Override max concurrent runners.")
    parser.add_argument("--domains", default=",".join(DOMAINS),
                        help="Comma-separated subset of domains to run (default: all 5).")
    parser.add_argument("--logdir", default="logs/parallel",
                        help="Directory for per-domain stdout/stderr logs.")
    parser.add_argument("--dry-print", action="store_true",
                        help="Print spawn plan without launching subprocesses.")
    args, forward_args = parser.parse_known_args()

    selected = tuple(d.strip() for d in args.domains.split(",") if d.strip())
    for d in selected:
        if d not in DOMAINS:
            sys.exit(f"unknown domain {d!r}; valid: {DOMAINS}")

    by_domain = _split_task_list(args.task_list, selected)
    nonempty = [(d, ts) for d, ts in by_domain.items() if ts]
    if not nonempty:
        sys.exit("no tasks to run after domain split")

    mode = _mode_from(forward_args)
    needs_gpu = (mode == "pagedigest")
    visible = _visible_gpu_ids()

    if needs_gpu:
        if not visible:
            sys.exit(f"mode={mode!r} needs a GPU but none visible. "
                     "Set CUDA_VISIBLE_DEVICES or check nvidia-smi.")
        max_par = len(visible)
    else:
        max_par = len(nonempty)
    if args.max_parallel > 0:
        max_par = min(max_par, args.max_parallel)

    print(f"[parallel] mode={mode!r}  needs_gpu={needs_gpu}  "
          f"visible_gpus={visible or '-'}  max_parallel={max_par}  "
          f"domains={[d for d,_ in nonempty]}", file=sys.stderr)

    log_root = Path(args.logdir) / mode
    log_root.mkdir(parents=True, exist_ok=True)
    sublist_root = log_root / "sublists"
    sublist_root.mkdir(parents=True, exist_ok=True)

    runs: List[Dict] = []
    for domain, tasks in nonempty:
        sub = sublist_root / f"{domain}.txt"
        sub.write_text("\n".join(tasks) + "\n")
        cmd = [sys.executable, "-m", "benchmarks.runner",
               *forward_args, "--task-list", str(sub)]
        runs.append({
            "domain": domain,
            "n_tasks": len(tasks),
            "cmd": cmd,
            "log_path": log_root / f"{domain}.log",
        })

    batches: List[List[Dict]] = [runs[i:i + max_par] for i in range(0, len(runs), max_par)]

    print(f"\n[parallel] plan: {len(batches)} batch(es) x <={max_par} runners:",
          file=sys.stderr)
    for bi, batch in enumerate(batches):
        for j, r in enumerate(batch):
            gpu = visible[j] if (needs_gpu and visible) else "-"
            print(f"  batch{bi+1}  {r['domain']:16s} n={r['n_tasks']:>4d}  gpu={gpu}  "
                  f"log={r['log_path']}", file=sys.stderr)
    if args.dry_print:
        for bi, batch in enumerate(batches):
            print(f"\n# === batch {bi+1} ===")
            for r in batch:
                print(f"# {r['domain']}\n" + " ".join(r["cmd"]) + "\n")
        return 0

    overall_rcs: List[int] = []
    for bi, batch in enumerate(batches):
        print(f"\n[parallel] === batch {bi+1}/{len(batches)} ===", file=sys.stderr)
        procs: List[Tuple[Dict, subprocess.Popen]] = []
        log_handles: List = []
        try:
            for j, r in enumerate(batch):
                env = os.environ.copy()
                if needs_gpu and visible:
                    env["CUDA_VISIBLE_DEVICES"] = visible[j]
                f = open(r["log_path"], "w")
                log_handles.append(f)
                p = subprocess.Popen(
                    r["cmd"], env=env, stdout=f, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                procs.append((r, p))
                gpu_str = env.get("CUDA_VISIBLE_DEVICES", "-")
                print(f"[parallel] spawned pid={p.pid:>8d}  {r['domain']:16s} gpu={gpu_str}",
                      file=sys.stderr)

            outstanding = list(procs)
            while outstanding:
                time.sleep(15)
                still = []
                for r, p in outstanding:
                    if p.poll() is None:
                        still.append((r, p))
                    else:
                        overall_rcs.append(p.returncode)
                        print(f"[parallel] done   pid={p.pid:>8d}  {r['domain']:16s} "
                              f"rc={p.returncode}", file=sys.stderr)
                outstanding = still
        except KeyboardInterrupt:
            print("\n[parallel] Ctrl-C — killing children", file=sys.stderr)
            for _, p in procs:
                try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except ProcessLookupError: pass
            for _, p in procs:
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except ProcessLookupError: pass
            return 130
        finally:
            for f in log_handles:
                try: f.close()
                except Exception: pass

    print(f"\n[parallel] all batches done. rcs={overall_rcs}", file=sys.stderr)
    return max(overall_rcs) if overall_rcs else 0


if __name__ == "__main__":
    sys.exit(main())
