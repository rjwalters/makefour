"""
Validation utilities for ONNX exported models.

Ensures exported ONNX models produce identical outputs to the original
PyTorch models within acceptable numerical tolerances.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ValidationResult:
    """Result of ONNX validation."""

    valid: bool
    policy_max_diff: float
    value_max_diff: float
    policy_mean_diff: float
    value_mean_diff: float
    num_test_cases: int
    error_message: str | None = None

    def __str__(self) -> str:
        if self.valid:
            return (
                f"Validation PASSED ({self.num_test_cases} test cases)\n"
                f"  Policy: max_diff={self.policy_max_diff:.2e}, mean_diff={self.policy_mean_diff:.2e}\n"
                f"  Value:  max_diff={self.value_max_diff:.2e}, mean_diff={self.value_mean_diff:.2e}"
            )
        else:
            return f"Validation FAILED: {self.error_message}"


def validate_onnx_model(
    onnx_path: str | Path,
    pytorch_model: nn.Module,
    input_shape: Tuple[int, ...],
    num_samples: int = 100,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> ValidationResult:
    """
    Validate that an ONNX model matches the PyTorch model outputs.

    Args:
        onnx_path: Path to ONNX model file
        pytorch_model: Original PyTorch model
        input_shape: Input shape (without batch dimension)
        num_samples: Number of random test samples
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        ValidationResult with comparison metrics
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return ValidationResult(
            valid=False,
            policy_max_diff=float("inf"),
            value_max_diff=float("inf"),
            policy_mean_diff=float("inf"),
            value_mean_diff=float("inf"),
            num_test_cases=0,
            error_message="onnxruntime not installed",
        )

    # Load ONNX session
    try:
        session = ort.InferenceSession(str(onnx_path))
    except Exception as e:
        return ValidationResult(
            valid=False,
            policy_max_diff=float("inf"),
            value_max_diff=float("inf"),
            policy_mean_diff=float("inf"),
            value_mean_diff=float("inf"),
            num_test_cases=0,
            error_message=f"Failed to load ONNX model: {e}",
        )

    # Set PyTorch model to eval mode
    pytorch_model.eval()

    # Generate test inputs
    test_inputs = [torch.randn(1, *input_shape) for _ in range(num_samples)]

    # Compare outputs
    policy_diffs = []
    value_diffs = []

    input_name = session.get_inputs()[0].name

    try:
        for test_input in test_inputs:
            # PyTorch inference
            with torch.no_grad():
                pt_policy, pt_value = pytorch_model(test_input)

            # ONNX inference
            onnx_outputs = session.run(None, {input_name: test_input.numpy()})
            onnx_policy, onnx_value = onnx_outputs

            # Compute differences
            policy_diff = np.abs(pt_policy.numpy() - onnx_policy)
            value_diff = np.abs(pt_value.numpy() - onnx_value)

            policy_diffs.append(policy_diff)
            value_diffs.append(value_diff)
    except Exception as e:
        return ValidationResult(
            valid=False,
            policy_max_diff=float("inf"),
            value_max_diff=float("inf"),
            policy_mean_diff=float("inf"),
            value_mean_diff=float("inf"),
            num_test_cases=0,
            error_message=f"Inference error: {e}",
        )

    # Aggregate statistics
    all_policy_diffs = np.concatenate(policy_diffs)
    all_value_diffs = np.concatenate(value_diffs)

    policy_max_diff = float(np.max(all_policy_diffs))
    value_max_diff = float(np.max(all_value_diffs))
    policy_mean_diff = float(np.mean(all_policy_diffs))
    value_mean_diff = float(np.mean(all_value_diffs))

    # Check tolerances
    policy_valid = np.allclose(
        np.zeros_like(all_policy_diffs), all_policy_diffs, rtol=rtol, atol=atol
    )
    value_valid = np.allclose(
        np.zeros_like(all_value_diffs), all_value_diffs, rtol=rtol, atol=atol
    )

    valid = policy_valid and value_valid
    error_message = None
    if not valid:
        if not policy_valid:
            error_message = f"Policy outputs differ: max_diff={policy_max_diff:.2e}"
        if not value_valid:
            msg = f"Value outputs differ: max_diff={value_max_diff:.2e}"
            error_message = msg if error_message is None else f"{error_message}; {msg}"

    return ValidationResult(
        valid=valid,
        policy_max_diff=policy_max_diff,
        value_max_diff=value_max_diff,
        policy_mean_diff=policy_mean_diff,
        value_mean_diff=value_mean_diff,
        num_test_cases=num_samples,
        error_message=error_message,
    )


def compare_outputs(
    pytorch_output: Tuple[torch.Tensor, torch.Tensor],
    onnx_output: List[np.ndarray],
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Tuple[bool, dict]:
    """
    Compare PyTorch and ONNX outputs for a single inference.

    Args:
        pytorch_output: Tuple of (policy, value) tensors from PyTorch
        onnx_output: List of [policy, value] arrays from ONNX
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Tuple of (matches, details_dict)
    """
    pt_policy, pt_value = pytorch_output
    onnx_policy, onnx_value = onnx_output

    # Convert PyTorch to numpy
    pt_policy_np = pt_policy.detach().numpy()
    pt_value_np = pt_value.detach().numpy()

    # Compare policy
    policy_match = np.allclose(pt_policy_np, onnx_policy, rtol=rtol, atol=atol)
    policy_max_diff = float(np.max(np.abs(pt_policy_np - onnx_policy)))

    # Compare value
    value_match = np.allclose(pt_value_np, onnx_value, rtol=rtol, atol=atol)
    value_max_diff = float(np.max(np.abs(pt_value_np - onnx_value)))

    details = {
        "policy_match": policy_match,
        "value_match": value_match,
        "policy_max_diff": policy_max_diff,
        "value_max_diff": value_max_diff,
        "pytorch_policy": pt_policy_np.tolist(),
        "onnx_policy": onnx_policy.tolist(),
        "pytorch_value": float(pt_value_np[0, 0]),
        "onnx_value": float(onnx_value[0, 0]),
    }

    return policy_match and value_match, details


def validate_with_game_positions(
    onnx_path: str | Path,
    pytorch_model: nn.Module,
    encoding_func,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> ValidationResult:
    """
    Validate using realistic game positions instead of random inputs.

    Args:
        onnx_path: Path to ONNX model file
        pytorch_model: Original PyTorch model
        encoding_func: Function to encode board positions
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        ValidationResult with comparison metrics
    """
    from ..data import board_from_moves

    # Representative game positions to test
    test_positions = [
        # Empty board
        [],
        # Single move
        [3],
        # Early game
        [3, 3, 4, 2],
        # Mid game
        [3, 3, 4, 4, 5, 2, 6, 0, 1],
        # Near full column
        [3, 3, 3, 3, 3, 3],
        # Horizontal threat
        [0, 6, 1, 6, 2, 6],
        # Diagonal threat
        [0, 1, 1, 2, 2, 3, 2, 3, 3],
    ]

    try:
        import onnxruntime as ort
    except ImportError:
        return ValidationResult(
            valid=False,
            policy_max_diff=float("inf"),
            value_max_diff=float("inf"),
            policy_mean_diff=float("inf"),
            value_mean_diff=float("inf"),
            num_test_cases=0,
            error_message="onnxruntime not installed",
        )

    session = ort.InferenceSession(str(onnx_path))
    pytorch_model.eval()

    policy_diffs = []
    value_diffs = []
    input_name = session.get_inputs()[0].name

    for moves in test_positions:
        board, player = board_from_moves(moves)
        encoded = encoding_func(board, player)

        # Add batch dimension and convert to tensor
        test_input = torch.from_numpy(encoded).unsqueeze(0)

        # PyTorch inference
        with torch.no_grad():
            pt_policy, pt_value = pytorch_model(test_input)

        # ONNX inference
        onnx_outputs = session.run(None, {input_name: test_input.numpy()})
        onnx_policy, onnx_value = onnx_outputs

        # Compute differences
        policy_diff = np.abs(pt_policy.numpy() - onnx_policy)
        value_diff = np.abs(pt_value.numpy() - onnx_value)

        policy_diffs.append(policy_diff)
        value_diffs.append(value_diff)

    # Aggregate
    all_policy_diffs = np.concatenate(policy_diffs)
    all_value_diffs = np.concatenate(value_diffs)

    policy_max_diff = float(np.max(all_policy_diffs))
    value_max_diff = float(np.max(all_value_diffs))
    policy_mean_diff = float(np.mean(all_policy_diffs))
    value_mean_diff = float(np.mean(all_value_diffs))

    valid = (policy_max_diff < atol + rtol) and (value_max_diff < atol + rtol)

    return ValidationResult(
        valid=valid,
        policy_max_diff=policy_max_diff,
        value_max_diff=value_max_diff,
        policy_mean_diff=policy_mean_diff,
        value_mean_diff=value_mean_diff,
        num_test_cases=len(test_positions),
        error_message=None if valid else "Outputs differ beyond tolerance",
    )
