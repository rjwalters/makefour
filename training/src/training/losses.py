"""
Loss Functions for Connect Four Neural Network Training

Provides loss functions for policy (move prediction) and value (position evaluation) heads.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def policy_loss(
    policy_logits: Tensor,
    target_move: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Cross-entropy loss for move prediction.

    Args:
        policy_logits: (batch, 7) raw logits from the policy head.
        target_move: (batch,) integer column indices (0-6).
        mask: (batch, 7) legal move mask. If provided, illegal moves are masked
              before computing softmax to prevent the model from learning to
              predict illegal moves.

    Returns:
        Scalar loss tensor.
    """
    if mask is not None:
        # Mask illegal moves by setting their logits to very negative values
        # This ensures softmax assigns ~0 probability to illegal moves
        policy_logits = policy_logits.masked_fill(~mask.bool(), -1e9)

    return F.cross_entropy(policy_logits, target_move)


def value_loss(value_pred: Tensor, target_value: Tensor) -> Tensor:
    """
    MSE loss for position evaluation.

    Args:
        value_pred: (batch, 1) or (batch,) predicted value in [-1, 1].
        target_value: (batch, 1) or (batch,) actual game outcome.

    Returns:
        Scalar loss tensor.
    """
    # Ensure consistent shapes
    value_pred = value_pred.view(-1)
    target_value = target_value.view(-1)

    return F.mse_loss(value_pred, target_value)


def combined_loss(
    policy_logits: Tensor,
    value_pred: Tensor,
    policy_target: Tensor,
    value_target: Tensor,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    legal_mask: Tensor | None = None,
    entropy_weight: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    """
    AlphaZero-style combined loss for dual-headed networks.

    Combines policy cross-entropy loss with value MSE loss, optionally
    including entropy regularization to encourage exploration.

    Args:
        policy_logits: (batch, 7) raw logits from policy head.
        value_pred: (batch, 1) or (batch,) predicted position value.
        policy_target: (batch,) target move indices.
        value_target: (batch, 1) or (batch,) target game outcomes.
        policy_weight: Weight for policy loss component.
        value_weight: Weight for value loss component.
        legal_mask: (batch, 7) optional mask for legal moves.
        entropy_weight: Weight for entropy regularization (0 to disable).

    Returns:
        Tuple of (total_loss, loss_components_dict) where loss_components_dict
        contains individual loss values for logging.
    """
    p_loss = policy_loss(policy_logits, policy_target, legal_mask)
    v_loss = value_loss(value_pred, value_target)

    total = policy_weight * p_loss + value_weight * v_loss

    components = {
        "policy_loss": p_loss.item(),
        "value_loss": v_loss.item(),
    }

    # Optional entropy regularization
    if entropy_weight > 0:
        e_loss = entropy_loss(policy_logits, legal_mask)
        total = total - entropy_weight * e_loss  # Subtract to maximize entropy
        components["entropy_loss"] = e_loss.item()

    components["total_loss"] = total.item()

    return total, components


def entropy_loss(policy_logits: Tensor, mask: Tensor | None = None) -> Tensor:
    """
    Computes the entropy of the policy distribution.

    Higher entropy means more uniform (exploratory) policy. This can be used
    as a regularization term to prevent the policy from becoming too confident
    too early in training.

    Args:
        policy_logits: (batch, 7) raw logits.
        mask: (batch, 7) optional legal move mask.

    Returns:
        Mean entropy over the batch (scalar tensor).
    """
    if mask is not None:
        policy_logits = policy_logits.masked_fill(~mask.bool(), -1e9)

    # Compute softmax probabilities
    probs = F.softmax(policy_logits, dim=-1)
    log_probs = F.log_softmax(policy_logits, dim=-1)

    # Entropy = -sum(p * log(p))
    # Mask out the contribution from illegal moves (their prob is ~0)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    return entropy.mean()


def policy_accuracy(
    policy_logits: Tensor,
    target_move: Tensor,
    mask: Tensor | None = None,
) -> float:
    """
    Computes policy accuracy (percentage of correct move predictions).

    Args:
        policy_logits: (batch, 7) raw logits.
        target_move: (batch,) target move indices.
        mask: (batch, 7) optional legal move mask.

    Returns:
        Accuracy as a float in [0, 1].
    """
    if mask is not None:
        policy_logits = policy_logits.masked_fill(~mask.bool(), -1e9)

    predictions = policy_logits.argmax(dim=-1)
    correct = (predictions == target_move).float().mean()
    return correct.item()


def policy_top_k_accuracy(
    policy_logits: Tensor,
    target_move: Tensor,
    k: int = 3,
    mask: Tensor | None = None,
) -> float:
    """
    Computes top-k policy accuracy (target in top k predictions).

    Args:
        policy_logits: (batch, 7) raw logits.
        target_move: (batch,) target move indices.
        k: Number of top predictions to consider.
        mask: (batch, 7) optional legal move mask.

    Returns:
        Top-k accuracy as a float in [0, 1].
    """
    if mask is not None:
        policy_logits = policy_logits.masked_fill(~mask.bool(), -1e9)

    # Get top-k predictions
    _, top_k_indices = policy_logits.topk(k, dim=-1)

    # Check if target is in top-k
    target_expanded = target_move.unsqueeze(-1)
    correct = (top_k_indices == target_expanded).any(dim=-1).float().mean()
    return correct.item()


def value_mae(value_pred: Tensor, target_value: Tensor) -> float:
    """
    Computes mean absolute error for value predictions.

    Args:
        value_pred: (batch, 1) or (batch,) predicted values.
        target_value: (batch, 1) or (batch,) target values.

    Returns:
        MAE as a float.
    """
    value_pred = value_pred.view(-1)
    target_value = target_value.view(-1)
    return F.l1_loss(value_pred, target_value).item()
