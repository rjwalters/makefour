"""Tests for loss functions."""

import pytest
import torch

from src.training.losses import (
    combined_loss,
    entropy_loss,
    policy_accuracy,
    policy_loss,
    policy_top_k_accuracy,
    value_loss,
    value_mae,
)


class TestPolicyLoss:
    """Tests for policy_loss function."""

    def test_basic_policy_loss(self):
        """Test that policy loss computes correctly."""
        batch_size = 4
        policy_logits = torch.randn(batch_size, 7, requires_grad=True)
        target_move = torch.randint(0, 7, (batch_size,))

        loss = policy_loss(policy_logits, target_move)

        assert loss.shape == ()
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_policy_loss_with_mask(self):
        """Test policy loss with legal move mask."""
        batch_size = 4
        policy_logits = torch.randn(batch_size, 7)
        target_move = torch.zeros(batch_size, dtype=torch.long)  # Column 0

        # Only column 0 is legal
        mask = torch.zeros(batch_size, 7, dtype=torch.bool)
        mask[:, 0] = True

        loss = policy_loss(policy_logits, target_move, mask=mask)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_policy_loss_perfect_prediction(self):
        """Test that perfect predictions give low loss."""
        batch_size = 4
        # Create logits where the correct column has highest value
        target_move = torch.tensor([0, 1, 2, 3])
        policy_logits = torch.zeros(batch_size, 7)
        for i, col in enumerate(target_move):
            policy_logits[i, col] = 10.0  # High logit for correct column

        loss = policy_loss(policy_logits, target_move)
        assert loss.item() < 0.1  # Should be very low


class TestValueLoss:
    """Tests for value_loss function."""

    def test_basic_value_loss(self):
        """Test that value loss computes correctly."""
        batch_size = 4
        value_pred = torch.randn(batch_size, 1, requires_grad=True)
        target_value = torch.randn(batch_size, 1)

        loss = value_loss(value_pred, target_value)

        assert loss.shape == ()
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_value_loss_perfect_prediction(self):
        """Test that perfect predictions give zero loss."""
        value_pred = torch.tensor([[0.5], [-0.3], [0.0]])
        target_value = torch.tensor([[0.5], [-0.3], [0.0]])

        loss = value_loss(value_pred, target_value)
        assert loss.item() < 1e-6

    def test_value_loss_flat_tensors(self):
        """Test value loss works with flat tensors."""
        value_pred = torch.tensor([0.5, -0.3, 0.0])
        target_value = torch.tensor([0.5, -0.3, 0.0])

        loss = value_loss(value_pred, target_value)
        assert loss.item() < 1e-6


class TestCombinedLoss:
    """Tests for combined_loss function."""

    def test_combined_loss_returns_components(self):
        """Test that combined loss returns loss components."""
        batch_size = 4
        policy_logits = torch.randn(batch_size, 7)
        value_pred = torch.randn(batch_size, 1)
        policy_target = torch.randint(0, 7, (batch_size,))
        value_target = torch.randn(batch_size, 1)

        total_loss, components = combined_loss(
            policy_logits, value_pred, policy_target, value_target
        )

        assert total_loss.shape == ()
        assert "policy_loss" in components
        assert "value_loss" in components
        assert "total_loss" in components

    def test_combined_loss_with_weights(self):
        """Test that weights affect the total loss."""
        batch_size = 4
        policy_logits = torch.randn(batch_size, 7)
        value_pred = torch.randn(batch_size, 1)
        policy_target = torch.randint(0, 7, (batch_size,))
        value_target = torch.randn(batch_size, 1)

        # Loss with default weights
        loss1, _ = combined_loss(
            policy_logits, value_pred, policy_target, value_target,
            policy_weight=1.0, value_weight=1.0
        )

        # Loss with different weights
        loss2, _ = combined_loss(
            policy_logits, value_pred, policy_target, value_target,
            policy_weight=2.0, value_weight=0.5
        )

        # Losses should be different
        assert loss1.item() != loss2.item()

    def test_combined_loss_with_entropy(self):
        """Test combined loss with entropy regularization."""
        batch_size = 4
        policy_logits = torch.randn(batch_size, 7)
        value_pred = torch.randn(batch_size, 1)
        policy_target = torch.randint(0, 7, (batch_size,))
        value_target = torch.randn(batch_size, 1)

        _, components = combined_loss(
            policy_logits, value_pred, policy_target, value_target,
            entropy_weight=0.01
        )

        assert "entropy_loss" in components


class TestEntropyLoss:
    """Tests for entropy_loss function."""

    def test_uniform_has_max_entropy(self):
        """Test that uniform distribution has maximum entropy."""
        # Uniform logits (all same) should give max entropy
        uniform_logits = torch.zeros(4, 7)
        entropy = entropy_loss(uniform_logits)

        # Max entropy for 7 choices is log(7)
        expected_max = torch.log(torch.tensor(7.0)).item()
        assert abs(entropy.item() - expected_max) < 0.01

    def test_peaked_has_low_entropy(self):
        """Test that peaked distribution has low entropy."""
        # Very peaked distribution (one high value)
        peaked_logits = torch.full((4, 7), -10.0)
        peaked_logits[:, 0] = 10.0

        entropy = entropy_loss(peaked_logits)
        assert entropy.item() < 0.1


class TestMetrics:
    """Tests for accuracy and MAE metrics."""

    def test_policy_accuracy_perfect(self):
        """Test policy accuracy with perfect predictions."""
        batch_size = 4
        policy_logits = torch.zeros(batch_size, 7)
        target_move = torch.tensor([0, 1, 2, 3])

        for i, col in enumerate(target_move):
            policy_logits[i, col] = 10.0

        acc = policy_accuracy(policy_logits, target_move)
        assert acc == 1.0

    def test_policy_accuracy_wrong(self):
        """Test policy accuracy with wrong predictions."""
        batch_size = 4
        policy_logits = torch.zeros(batch_size, 7)
        target_move = torch.tensor([0, 1, 2, 3])

        # Predict wrong columns
        for i, col in enumerate(target_move):
            policy_logits[i, (col + 1) % 7] = 10.0

        acc = policy_accuracy(policy_logits, target_move)
        assert acc == 0.0

    def test_policy_top_k_accuracy(self):
        """Test top-k accuracy."""
        batch_size = 4
        policy_logits = torch.zeros(batch_size, 7)
        target_move = torch.tensor([0, 1, 2, 3])

        # Target is in top 3 but not top 1
        for i, col in enumerate(target_move):
            policy_logits[i, (col + 1) % 7] = 10.0  # Highest
            policy_logits[i, (col + 2) % 7] = 9.0   # Second
            policy_logits[i, col] = 8.0             # Third (target)

        top1_acc = policy_accuracy(policy_logits, target_move)
        top3_acc = policy_top_k_accuracy(policy_logits, target_move, k=3)

        assert top1_acc == 0.0
        assert top3_acc == 1.0

    def test_value_mae(self):
        """Test value MAE computation."""
        value_pred = torch.tensor([0.5, -0.3, 0.0])
        target_value = torch.tensor([0.6, -0.2, 0.1])

        mae = value_mae(value_pred, target_value)
        expected = (0.1 + 0.1 + 0.1) / 3
        assert abs(mae - expected) < 1e-6
