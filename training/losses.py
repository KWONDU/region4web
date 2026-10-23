"""Loss functions for training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for binary classification with class imbalance.

    Designed for decomposition where <5% of edges are cut (positive).

    Args:
        logits: Raw logits, shape (N,).
        targets: Binary labels (0=merge, 1=cut), shape (N,).
        alpha: Weighting factor for positive class.
        gamma: Focusing parameter (higher = more focus on hard examples).

    Returns:
        Scalar loss.
    """
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")

    # Focal modulation
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    # Alpha weighting
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    loss = alpha_t * focal_weight * bce
    return loss.mean()
