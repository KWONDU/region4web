"""Train the abstraction model (full SFT for purpose + state_summary).

Supports single-GPU and multi-GPU (DDP) training.
Tracks both epoch and global step. save/eval are step-based.
Supports resume from checkpoint with exact data position recovery.

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=7 python training/train_abstraction.py --config config.yaml

    # Multi-GPU (DDP)
    CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 training/train_abstraction.py --config config.yaml

    # Resume from latest checkpoint
    CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 training/train_abstraction.py --config config.yaml --resume
"""

from __future__ import annotations

import os
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import csv
import json
import logging
import random
from datetime import timedelta
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from training.dataset import AbstractionDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

for _name in ("httpx", "transformers", "huggingface_hub", "huggingface_hub.utils"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)


def _is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main() -> bool:
    return not _is_ddp() or dist.get_rank() == 0


def _load_config(path: str) -> Dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _format_example(subtree: str, purpose: str, state_summary: str, prompt_template: str, eos_token: str = "") -> tuple[str, str]:
    """Return (full_text, input_text) for loss masking."""
    input_text = prompt_template.replace("{region_axtree}", subtree)
    output_text = json.dumps({"purpose": purpose, "state_summary": state_summary})
    return f"{input_text}{output_text}{eos_token}", input_text


def _next_run_dir(base: Path) -> Path:
    existing = sorted(base.glob("run_*"))
    if not existing:
        return base / "run_001"
    num = int(existing[-1].name.split("_")[1]) + 1
    return base / f"run_{num:03d}"


def _latest_run_dir(base: Path) -> Path | None:
    existing = sorted(base.glob("run_*"))
    return existing[-1] if existing else None


def _latest_checkpoint(run_dir: Path) -> tuple[Path | None, int]:
    ckpts = sorted(run_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    if not ckpts:
        return None, 0
    latest = ckpts[-1]
    return latest, int(latest.name.split("_")[1])


def _configs_match(run_dir: Path, current_tc: Dict) -> bool:
    """Strict match: every key in current tc must equal saved (except output_dir)."""
    import yaml
    saved_path = run_dir / "config.yaml"
    if not saved_path.exists():
        return False
    with open(saved_path) as f:
        saved = yaml.safe_load(f)
    saved_tc = saved.get("training", {})
    for key in current_tc:
        if key == "output_dir":
            continue
        if current_tc.get(key) != saved_tc.get(key):
            return False
    return True


def train(config_path: str, resume: bool = False) -> None:
    config = _load_config(config_path)
    tc = config.get("training", {}).get("abstraction", {})

    base_model = tc.get("base_model", "Qwen/Qwen3-0.6B")
    data_path = tc.get("data_path", "data/abstraction.jsonl")
    prompt_path = tc.get("prompt_path", "prompts/data_generation/abstraction.txt")
    output_base = Path(tc.get("output_dir", "checkpoints/abstraction/"))
    max_steps = tc.get("max_steps", 100000)
    lr = float(tc.get("lr", 2e-5))
    batch_size = tc.get("batch_size", 1)
    effective_batch_size = tc.get("effective_batch_size", 48)
    max_seq_len = tc.get("max_seq_len", 8192)
    val_ratio = tc.get("val_ratio", 0.1)
    save_every = tc.get("save_every_steps", 500)
    eval_every = tc.get("eval_every_steps", 500)
    warmup_steps = tc.get("warmup_steps", 0)
    lr_schedule = tc.get("lr_schedule", "constant")

    # --- DDP setup ---
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        world_size = 1
        rank = 0

    # Compute grad_accum from effective_batch_size (reproducibility across GPU counts)
    denom = batch_size * world_size
    if effective_batch_size % denom != 0:
        raise ValueError(
            f"effective_batch_size ({effective_batch_size}) must be divisible by "
            f"batch_size ({batch_size}) * world_size ({world_size}) = {denom}"
        )
    grad_accum = effective_batch_size // denom
    if _is_main():
        logger.info("Effective batch=%d, per-device=%d, world=%d → grad_accum=%d",
                    effective_batch_size, batch_size, world_size, grad_accum)

    prompt_template = Path(prompt_path).read_text(encoding="utf-8")

    # --- Dataset & split ---
    dataset = AbstractionDataset(data_path)
    if _is_main():
        logger.info("Loaded %d samples from %s", len(dataset), data_path)

    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)
    val_size = int(len(indices) * val_ratio)
    val_indices = indices[:val_size]
    train_indices_all = indices[val_size:]

    # DDP partition
    per_rank_train = len(train_indices_all) // world_size
    train_indices = train_indices_all[rank * per_rank_train:(rank + 1) * per_rank_train]
    per_rank_val = len(val_indices) // world_size
    val_indices = val_indices[rank * per_rank_val:(rank + 1) * per_rank_val]

    if _is_main():
        logger.info("Split: %d train/rank (%d total), %d val/rank (%d total)",
                     len(train_indices), per_rank_train * world_size,
                     len(val_indices), per_rank_val * world_size)

    # --- Resume check ---
    resume_step = 0
    ckpt_load_path = None
    run_dir = output_base / "run_001"  # default, overwritten below

    if resume:
        found_dir = _latest_run_dir(output_base)
        if found_dir:
            if not _configs_match(found_dir, tc):
                if _is_main():
                    logger.error("Config mismatch with %s. Cannot resume.", found_dir)
                if ddp:
                    dist.destroy_process_group()
                return
            run_dir = found_dir
            ckpt_load_path, resume_step = _latest_checkpoint(run_dir)
            if _is_main():
                if ckpt_load_path:
                    logger.info("Resuming from %s (step %d)", ckpt_load_path, resume_step)
                else:
                    logger.info("No checkpoint found in %s, starting fresh", run_dir)

    # --- Model ---
    if _is_main():
        logger.info("Loading model: %s", base_model)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_load_path) if ckpt_load_path else base_model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = model.to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def _lr_at(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        if lr_schedule == "cosine":
            import math
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            return lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr

    # --- Run directory (all ranks need run_dir for barrier sync) ---
    if _is_main():
        if not resume or not _latest_run_dir(output_base):
            run_dir = _next_run_dir(output_base)
            run_dir.mkdir(parents=True, exist_ok=True)

            import yaml
            # Snapshot: full training config (incl. base_model) + runtime-determined world_size.
            # grad_accum is derived from effective_batch_size / (batch_size * world_size).
            run_config = {
                "training": dict(tc),
                "world_size": world_size,
            }
            with open(run_dir / "config.yaml", "w") as f:
                yaml.dump(run_config, f, default_flow_style=False)

        logger.info("Run directory: %s", run_dir)

    # Broadcast run_dir to all ranks
    if ddp:
        if _is_main():
            run_dir_str = str(run_dir)
            run_dir_len = torch.tensor([len(run_dir_str)], dtype=torch.long, device=device)
        else:
            run_dir_len = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(run_dir_len, src=0)
        if _is_main():
            run_dir_bytes = torch.tensor(list(run_dir_str.encode()), dtype=torch.uint8, device=device)
        else:
            run_dir_bytes = torch.zeros(run_dir_len.item(), dtype=torch.uint8, device=device)
        dist.broadcast(run_dir_bytes, src=0)
        if not _is_main():
            run_dir = Path(bytes(run_dir_bytes.cpu().tolist()).decode())

    # --- Metrics CSV (main only) ---
    metrics_file = None
    metrics_writer = None
    if _is_main():
        metrics_path = run_dir / "metrics.csv"
        if resume and metrics_path.exists():
            metrics_file = open(metrics_path, "a", newline="")
            metrics_writer = csv.writer(metrics_file)
        else:
            metrics_file = open(metrics_path, "w", newline="")
            metrics_writer = csv.writer(metrics_file)
            metrics_writer.writerow(["step", "epoch", "train_loss", "val_loss"])

    # --- Compute resume position ---
    samples_per_step = grad_accum * batch_size  # per rank
    samples_per_epoch = len(train_indices)       # per rank
    total_samples_done = resume_step * samples_per_step
    resume_epoch = total_samples_done // samples_per_epoch
    resume_sample_idx = total_samples_done % samples_per_epoch

    if _is_main() and resume_step > 0:
        logger.info("Resume position: epoch %d, sample %d/%d (per rank)",
                     resume_epoch + 1, resume_sample_idx, samples_per_epoch)

    # --- Training loop (infinite data cycling, step-based) ---
    global_step = resume_step
    accum_count = 0
    running_loss = 0.0
    running_count = 0
    epoch = resume_epoch
    sample_ptr = resume_sample_idx

    optimizer.zero_grad()
    model.train()

    # Shuffle for current epoch
    random.seed(42 + epoch)
    random.shuffle(train_indices)

    target_steps = max_steps if max_steps else float("inf")

    pbar = None
    if _is_main():
        pbar = tqdm(total=samples_per_epoch - sample_ptr,
                    desc=f"Epoch {epoch+1} | Step {global_step}", leave=True)

    while global_step < target_steps:
        # Get next batch
        batch_indices = train_indices[sample_ptr:sample_ptr + batch_size]
        sample_ptr += batch_size

        # Epoch boundary: reshuffle and reset
        if sample_ptr >= samples_per_epoch:
            epoch += 1
            sample_ptr = 0
            random.seed(42 + epoch)
            random.shuffle(train_indices)

            if pbar:
                pbar.close()
            if _is_main():
                pbar = tqdm(total=samples_per_epoch,
                            desc=f"Epoch {epoch+1} | Step {global_step}", leave=True)

        if not batch_indices:
            continue

        batch_results = [
            _format_example(
                dataset[j]["subtree"], dataset[j]["purpose"],
                dataset[j]["state_summary"], prompt_template,
                eos_token=tokenizer.eos_token,
            )
            for j in batch_indices
        ]
        batch_texts = [r[0] for r in batch_results]
        batch_inputs = [r[1] for r in batch_results]

        encoded = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
        ).to(device)

        if encoded["input_ids"].shape[1] > max_seq_len:
            if pbar:
                pbar.update(len(batch_indices))
            continue

        labels = encoded["input_ids"].clone()
        # Mask padding (but not EOS — use attention_mask instead of token ID)
        labels[encoded["attention_mask"] == 0] = -100
        # Mask input/prompt portion — only compute loss on output
        input_lens = [len(tokenizer(inp, add_special_tokens=False)["input_ids"]) for inp in batch_inputs]
        for bi, inp_len in enumerate(input_lens):
            labels[bi, :inp_len] = -100

        outputs = model(**encoded, labels=labels)
        loss = outputs.loss / grad_accum
        loss.backward()

        running_loss += outputs.loss.item()
        running_count += 1
        accum_count += 1

        if accum_count % grad_accum == 0:
            current_lr = _lr_at(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # --- Eval (all ranks) ---
            if global_step % eval_every == 0:
                avg_train_loss = running_loss / max(running_count, 1)

                if _is_main():
                    logger.info("Step %d (Epoch %d): train_loss=%.4f | evaluating...",
                                global_step, epoch + 1, avg_train_loss)

                eval_model = model.module if ddp else model
                eval_model.eval()
                val_loss = 0.0
                val_count = 0
                with torch.no_grad():
                    for vi in val_indices:
                        s = dataset[vi]
                        text, inp_text = _format_example(s["subtree"], s["purpose"],
                                                         s["state_summary"], prompt_template,
                                                         eos_token=tokenizer.eos_token)
                        enc = tokenizer(text, return_tensors="pt", padding=False).to(device)
                        if enc["input_ids"].shape[1] > max_seq_len:
                            continue
                        lab = enc["input_ids"].clone()
                        inp_len = len(tokenizer(inp_text, add_special_tokens=False)["input_ids"])
                        lab[0, :inp_len] = -100
                        out = eval_model(**enc, labels=lab)
                        val_loss += out.loss.item()
                        val_count += 1

                val_loss_local = val_loss / max(val_count, 1)

                if ddp:
                    val_tensor = torch.tensor([val_loss_local], device=device)
                    dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
                    val_loss_avg = val_tensor.item()
                else:
                    val_loss_avg = val_loss_local

                if _is_main():
                    logger.info("Step %d (Epoch %d): train_loss=%.4f | val_loss=%.4f",
                                global_step, epoch + 1, avg_train_loss, val_loss_avg)
                    metrics_writer.writerow([global_step, epoch + 1,
                                             f"{avg_train_loss:.4f}", f"{val_loss_avg:.4f}"])
                    metrics_file.flush()

                running_loss = 0.0
                running_count = 0
                model.train()

            # --- Save ---
            if global_step % save_every == 0 and _is_main():
                save_model = model.module if ddp else model
                ckpt_path = run_dir / f"step_{global_step}"
                save_model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)

        if pbar:
            pbar.update(len(batch_indices))
            pbar.set_description(f"Epoch {epoch+1} | Step {global_step}")

    if pbar:
        pbar.close()

    # Flush remaining
    if accum_count % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad()

    if _is_main():
        if metrics_file:
            metrics_file.close()
        logger.info("Training complete at step %d. Results in %s", global_step, run_dir)

    if ddp:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train abstraction model (full SFT)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    train(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
