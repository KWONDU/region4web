"""Train the decomposition model (RegionPartitioner).

Reads from data/decomposition.jsonl. Edge labels are derived
from region_root_backend_ids at training time.

Usage:
    python training/train_decomposition.py --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import torch
import torch.optim as optim

from framework.decomposition.features import extract_node_features
from framework.decomposition.model import RegionPartitioner
from training.dataset import DecompositionDataset
from training.losses import focal_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _rebuild_tree_from_text(axtree_text: str):
    """Rebuild a Tree from serialized (processed) AXTree text."""
    from core.types import Tree, TreeNode

    lines = axtree_text.split("\n")
    nodes: Dict[int, TreeNode] = {}
    stack: List[Tuple[int, TreeNode]] = []
    counter = 0

    for line in lines:
        if not line.strip():
            continue
        indent = 0
        for ch in line:
            if ch == "\t":
                indent += 1
            else:
                break
        stripped = line.strip()

        backend_id = 0
        rest = stripped
        id_match = re.match(r"\[(\d+)\]\s*(.*)", stripped)
        if id_match:
            backend_id = int(id_match.group(1))
            rest = id_match.group(2)

        parts = rest.split(" ", 1)
        role = parts[0] if parts else "generic"

        name = ""
        name_match = re.search(r"""(?<!value=)(['"])(.*?)\1""", rest)
        if name_match:
            name = name_match.group(2)

        value = ""
        value_match = re.search(r"""value=(['"])(.*?)\1""", rest)
        if value_match:
            value = value_match.group(2)

        properties: Dict[str, Any] = {}
        prop_match = re.search(r"""(?:['"]\)?)?\s*,\s*(.+)$""", rest)
        if prop_match:
            for attr in prop_match.group(1).split(","):
                attr = attr.strip()
                if not attr:
                    continue
                if "=" in attr:
                    k, v = attr.split("=", 1)
                    properties[k.strip()] = v.strip().strip("'\"")
                else:
                    properties[attr] = "true"

        node = TreeNode(
            id=counter, backend_node_id=backend_id, role=role,
            name=name, value=value, properties=properties,
        )
        nodes[counter] = node

        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stack:
            parent = stack[-1][1]
            node.parent = parent
            parent.children.append(node)

        stack.append((indent, node))
        counter += 1

    root = nodes[0] if nodes else TreeNode(id=0, backend_node_id=0, role="root", name="")
    backend_map = {n.backend_node_id: n.id for n in nodes.values() if n.backend_node_id}
    return Tree(root=root, nodes=nodes, backend_id_map=backend_map)


def _build_edge_labels(tree, region_root_backend_ids: List[int]) -> Dict[Tuple[int, int], int]:
    """Build edge labels from region root backend IDs. cut=1, merge=0."""
    cut_nids: Set[int] = set()
    for bid in region_root_backend_ids:
        if bid in tree.backend_id_map:
            cut_nids.add(tree.backend_id_map[bid])

    edge_labels: Dict[Tuple[int, int], int] = {}
    for node in tree.nodes.values():
        for child in node.children:
            edge_labels[(node.id, child.id)] = 1 if child.id in cut_nids else 0
    return edge_labels


def _evaluate(model, dataset, indices, device, fl_alpha, fl_gamma):
    """Evaluate on a subset of the dataset. Returns (avg_loss, precision, recall, f1)."""
    model.eval()
    total_loss = 0.0
    total_tp, total_fp, total_fn = 0, 0, 0
    processed = 0

    from tqdm import tqdm
    with torch.no_grad():
        for i in tqdm(indices, desc="[val]", leave=False):
            sample = dataset[i]
            tree = _rebuild_tree_from_text(sample["axtree"])
            if len(tree.nodes) < 2:
                continue

            features = extract_node_features(tree)
            edge_labels = _build_edge_labels(tree, sample["region_root_backend_ids"])
            _, edge_logits = model(tree, features, edge_labels=edge_labels)

            if not edge_logits:
                continue

            logit_list = []
            target_list = []
            for (pid, cid), logit in edge_logits.items():
                logit_list.append(logit)
                target_list.append(float(edge_labels.get((pid, cid), 0)))

            logits_t = torch.stack(logit_list)
            targets_t = torch.tensor(target_list, device=device)

            loss = focal_loss(logits_t, targets_t, alpha=fl_alpha, gamma=fl_gamma)
            total_loss += loss.item()
            processed += 1

            preds = (torch.sigmoid(logits_t) >= 0.5).float()
            total_tp += ((preds == 1) & (targets_t == 1)).sum().item()
            total_fp += ((preds == 1) & (targets_t == 0)).sum().item()
            total_fn += ((preds == 0) & (targets_t == 1)).sum().item()

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    avg_loss = total_loss / max(processed, 1)
    return avg_loss, precision, recall, f1


def _load_config(path: str) -> Dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _next_run_dir(base: Path) -> Path:
    """Find next run_NNN directory."""
    existing = sorted(base.glob("run_*"))
    if not existing:
        return base / "run_001"
    last = existing[-1].name
    num = int(last.split("_")[1]) + 1
    return base / f"run_{num:03d}"


def train(config_path: str) -> None:
    config = _load_config(config_path)
    tc = config.get("training", {}).get("decomposition", {})
    mc = config.get("framework", {}).get("decomposition", {}).get("model", {})

    data_path = tc.get("data_path", "data/decomposition.jsonl")
    output_base = Path(tc.get("output_dir", "checkpoints/decomposition/"))
    epochs = tc.get("epochs", 1000)
    lr = float(tc.get("lr", 1e-3))
    val_ratio = tc.get("val_ratio", 0.1)
    save_every = tc.get("save_every", 10)
    grad_clip = tc.get("grad_clip", 0.0)
    fl = tc.get("focal_loss", {})
    fl_alpha = fl.get("alpha", 0.75)
    fl_gamma = fl.get("gamma", 2.0)
    hidden_dim = mc.get("hidden_dim", 128)
    role_embed_dim = mc.get("role_embed_dim", 13)
    num_layers = mc.get("num_layers", 2)
    wd = tc.get("weight_decay", 0.0)
    warmup = tc.get("warmup_epochs", 0)
    schedule = tc.get("lr_schedule", "constant")
    drop = mc.get("dropout", 0.0)

    dataset = DecompositionDataset(data_path)
    logger.info("Loaded %d pages from %s", len(dataset), data_path)

    # Train/val split (page-level, deterministic)
    import random
    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)
    val_size = int(len(indices) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    logger.info("Split: %d train, %d val", len(train_indices), len(val_indices))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegionPartitioner(hidden_dim=hidden_dim, role_embed_dim=role_embed_dim,
                              num_layers=num_layers, dropout=drop).to(device)
    if wd > 0:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        logger.info("Optimizer: AdamW(lr=%g, weight_decay=%g)", lr, wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        logger.info("Optimizer: Adam(lr=%g)", lr)

    def _lr_at(ep: int) -> float:
        if warmup > 0 and ep < warmup:
            return lr * (ep + 1) / warmup
        if schedule == "cosine":
            import math
            progress = (ep - warmup) / max(epochs - warmup, 1)
            progress = min(max(progress, 0.0), 1.0)
            return lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr

    if warmup > 0 or schedule != "constant":
        logger.info("LR schedule: warmup=%d epochs, %s decay", warmup, schedule)
    if drop > 0:
        logger.info("Dropout: %g", drop)
    logger.info("Focal loss alpha=%g gamma=%g", fl_alpha, fl_gamma)
    logger.info("Hidden dim=%d num_layers=%d", hidden_dim, num_layers)

    # Run directory
    run_dir = _next_run_dir(output_base)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    # Save config snapshot
    import yaml
    run_config = {"model": dict(mc), "training": dict(tc)}
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)

    # Metrics CSV
    metrics_path = run_dir / "metrics.csv"
    metrics_file = open(metrics_path, "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["epoch", "train_loss", "train_P", "train_R", "train_F1",
                              "val_loss", "val_P", "val_R", "val_F1"])

    for epoch in range(epochs):
        # --- LR schedule ---
        current_lr = _lr_at(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # --- Train ---
        model.train()
        epoch_loss = 0.0
        total_tp, total_fp, total_fn = 0, 0, 0
        processed = 0

        random.shuffle(train_indices)

        from tqdm import tqdm
        for i in tqdm(train_indices, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False):
            sample = dataset[i]
            tree = _rebuild_tree_from_text(sample["axtree"])
            if len(tree.nodes) < 2:
                continue

            features = extract_node_features(tree)
            edge_labels = _build_edge_labels(tree, sample["region_root_backend_ids"])
            _, edge_logits = model(tree, features, edge_labels=edge_labels)

            if not edge_logits:
                continue

            logit_list = []
            target_list = []
            for (pid, cid), logit in edge_logits.items():
                logit_list.append(logit)
                target_list.append(float(edge_labels.get((pid, cid), 0)))

            logits_t = torch.stack(logit_list)
            targets_t = torch.tensor(target_list, device=device)

            loss = focal_loss(logits_t, targets_t, alpha=fl_alpha, gamma=fl_gamma)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            processed += 1

            preds = (torch.sigmoid(logits_t) >= 0.5).float()
            total_tp += ((preds == 1) & (targets_t == 1)).sum().item()
            total_fp += ((preds == 1) & (targets_t == 0)).sum().item()
            total_fn += ((preds == 0) & (targets_t == 1)).sum().item()

        train_loss = epoch_loss / max(processed, 1)
        train_p = total_tp / max(total_tp + total_fp, 1)
        train_r = total_tp / max(total_tp + total_fn, 1)
        train_f1 = 2 * train_p * train_r / max(train_p + train_r, 1e-8)

        # --- Val ---
        val_loss, val_p, val_r, val_f1 = _evaluate(
            model, dataset, val_indices, device, fl_alpha, fl_gamma)

        logger.info(
            "Epoch %d/%d: train_loss=%.4f train_F1=%.3f | val_loss=%.4f val_F1=%.3f",
            epoch + 1, epochs, train_loss, train_f1, val_loss, val_f1,
        )

        # Save metrics
        metrics_writer.writerow([
            epoch + 1, f"{train_loss:.4f}", f"{train_p:.3f}", f"{train_r:.3f}", f"{train_f1:.3f}",
            f"{val_loss:.4f}", f"{val_p:.3f}", f"{val_r:.3f}", f"{val_f1:.3f}",
        ])
        metrics_file.flush()

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch + 1},
                        run_dir / f"epoch_{epoch + 1}.pt")

    metrics_file.close()
    logger.info("Training complete. Results in %s", run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train decomposition model")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
