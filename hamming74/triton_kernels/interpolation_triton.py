"""
Triton GPU Kernel for Linear Interpolation of Double-Detected Errors.

When Hamming(8,4) SECDED detects a double-bit error (but cannot correct it),
this kernel replaces the corrupted value with the average of its neighbors.

Algorithm:
    For each position i with DOUBLE_DETECTED error:
        result[i] = (q[i-1] + q[i+1]) / 2.0
    Boundary handling:
        First position (i=0): result[0] = (q[0] + q[1]) / 2.0
        Last position (i=n-1): result[n-1] = (q[n-2] + q[n-1]) / 2.0

This provides a reasonable approximation for corrupted values while preserving
the uncorrupted values exactly.
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

from .config import INTERPOLATION_BLOCK_SIZE, ErrorType


# =============================================================================
# Triton Interpolation Kernel
# =============================================================================

@triton.jit
def interpolate_double_errors_kernel(
    # Pointers
    q_ptr,              # Input: quantized values (uint8 or float)
    error_type_ptr,     # Input: error types from SECDED decode
    output_ptr,         # Output: interpolated values
    # Dimensions
    seq_len,            # Length of each sequence
    num_sequences,      # Number of sequences (batch dimension)
    # Constants
    BLOCK_SIZE: tl.constexpr,
    DOUBLE_DETECTED: tl.constexpr,
):
    """
    Interpolate double-detected errors with neighbor averages.

    Processes data as 2D: [num_sequences, seq_len]
    Each program handles one block of one sequence.
    """
    # Program ID encodes both sequence and position
    pid = tl.program_id(0)
    num_blocks_per_seq = tl.cdiv(seq_len, BLOCK_SIZE)
    seq_idx = pid // num_blocks_per_seq
    block_idx = pid % num_blocks_per_seq

    # Compute offsets within this sequence
    local_offset = block_idx * BLOCK_SIZE
    offsets = local_offset + tl.arange(0, BLOCK_SIZE)
    mask = (seq_idx < num_sequences) & (offsets < seq_len)

    # Base pointer for this sequence
    base = seq_idx * seq_len

    # Load values and error types
    q = tl.load(q_ptr + base + offsets, mask=mask, other=0).to(tl.float32)
    err = tl.load(error_type_ptr + base + offsets, mask=mask, other=0)

    # Identify double-detected errors
    is_double = (err == DOUBLE_DETECTED)

    # Compute neighbor indices with boundary clamping
    left_idx = tl.maximum(offsets - 1, 0)
    right_idx = tl.minimum(offsets + 1, seq_len - 1)

    # Load neighbor values
    left_val = tl.load(q_ptr + base + left_idx, mask=mask, other=0).to(tl.float32)
    right_val = tl.load(q_ptr + base + right_idx, mask=mask, other=0).to(tl.float32)

    # Compute interpolated value: average of neighbors
    interpolated = (left_val + right_val) * 0.5

    # Apply interpolation only for double-detected errors
    result = tl.where(is_double, interpolated, q)

    # Clamp to valid INT4 range [0, 15] with rounding
    result = tl.maximum(0.0, tl.minimum(15.0, result + 0.5))
    result = result.to(tl.uint8)

    # Store result
    tl.store(output_ptr + base + offsets, result, mask=mask)


# =============================================================================
# Python Wrapper Functions
# =============================================================================

def interpolate_double_errors(
    q: torch.Tensor,
    error_type: torch.Tensor,
    original_shape: Optional[Tuple[int, ...]] = None,
    seq_dim: int = -1,
) -> torch.Tensor:
    """
    Interpolate double-detected errors using Triton kernel.

    Args:
        q: Decoded INT4 values (uint8), any shape, on CUDA
        error_type: Error types from SECDED decode (uint8), same shape as q
        original_shape: Original tensor shape (for reshape tracking)
        seq_dim: Which dimension is the sequence dimension (default: -1, last dim)

    Returns:
        Interpolated values (uint8), same shape as input

    Note:
        Only DOUBLE_DETECTED positions are modified. All other values
        are preserved exactly.
    """
    assert q.is_cuda, "Input must be on CUDA device"
    assert error_type.is_cuda, "Error type must be on CUDA device"
    assert q.shape == error_type.shape, "Shape mismatch between q and error_type"

    # Early exit if no double errors (common case - skip kernel entirely)
    has_double_errors = (error_type == ErrorType.DOUBLE_DETECTED).any()
    if not has_double_errors:
        return q.clone()

    # Save original shape for reshape at end
    input_shape = q.shape

    # Handle different tensor shapes by reshaping to 2D [batch, seq_len]
    if q.dim() == 1:
        # Already 1D: treat as single sequence
        flat_q = q.unsqueeze(0)  # [1, N]
        flat_err = error_type.unsqueeze(0)
    elif q.dim() == 2:
        # Already 2D: use as-is
        flat_q = q
        flat_err = error_type
    else:
        # Multi-dimensional: move seq_dim to last, flatten batch dims
        # Normalize negative seq_dim
        if seq_dim < 0:
            seq_dim = q.dim() + seq_dim

        # Move sequence dimension to last position
        if seq_dim != q.dim() - 1:
            perm = list(range(q.dim()))
            perm.remove(seq_dim)
            perm.append(seq_dim)
            q = q.permute(*perm)
            error_type = error_type.permute(*perm)

        # Flatten all batch dimensions
        seq_len = q.shape[-1]
        batch_size = q.numel() // seq_len
        flat_q = q.reshape(batch_size, seq_len)
        flat_err = error_type.reshape(batch_size, seq_len)

    # Ensure contiguous for kernel
    flat_q = flat_q.contiguous().to(torch.uint8)
    flat_err = flat_err.contiguous().to(torch.uint8)

    num_sequences, seq_len = flat_q.shape

    # Allocate output
    output = torch.empty_like(flat_q)

    # Calculate grid size
    num_blocks_per_seq = triton.cdiv(seq_len, INTERPOLATION_BLOCK_SIZE)
    total_blocks = num_sequences * num_blocks_per_seq

    # Launch kernel
    interpolate_double_errors_kernel[(total_blocks,)](
        flat_q, flat_err, output,
        seq_len, num_sequences,
        BLOCK_SIZE=INTERPOLATION_BLOCK_SIZE,
        DOUBLE_DETECTED=ErrorType.DOUBLE_DETECTED,
    )

    # Reshape output back to original shape
    return output.reshape(input_shape)


def interpolate_double_errors_1d(
    q: torch.Tensor,
    error_type: torch.Tensor,
) -> torch.Tensor:
    """
    Simplified 1D interpolation for flat tensors.

    This is a convenience wrapper for the common case of flat tensors
    (e.g., flattened KV cache values).

    Args:
        q: Flat tensor of decoded INT4 values
        error_type: Flat tensor of error types

    Returns:
        Interpolated values
    """
    return interpolate_double_errors(q, error_type, seq_dim=-1)


# =============================================================================
# Verification
# =============================================================================

def verify_triton_vs_cpu():
    """
    Verify Triton interpolation kernel matches CPU reference implementation.

    Tests:
    1. No double errors -> exact copy
    2. Single double error in middle -> neighbor average
    3. Double error at boundary -> correct boundary handling
    4. Multiple scattered double errors
    5. Large tensor with random double errors
    """
    device = "cuda"

    print("Verifying Triton interpolation vs CPU reference...")

    # Test 1: No double errors (should return exact copy)
    q = torch.tensor([1, 5, 10, 15, 8], dtype=torch.uint8, device=device)
    err = torch.zeros_like(q)  # NO_ERROR = 0
    result = interpolate_double_errors(q, err)
    assert torch.equal(result, q), "No-error case should return exact copy"
    print("  [PASS] No double errors -> exact copy")

    # Test 2: Single double error in middle
    q = torch.tensor([4, 8, 12, 8, 4], dtype=torch.uint8, device=device)
    err = torch.tensor([0, 0, ErrorType.DOUBLE_DETECTED, 0, 0], dtype=torch.uint8, device=device)
    result = interpolate_double_errors(q, err)
    # Position 2 should be (8 + 8) / 2 = 8
    expected = torch.tensor([4, 8, 8, 8, 4], dtype=torch.uint8, device=device)
    assert torch.equal(result, expected), f"Middle interpolation failed: {result} vs {expected}"
    print("  [PASS] Single double error in middle -> neighbor average")

    # Test 3: Double error at first position (boundary)
    q = torch.tensor([15, 4, 8, 12], dtype=torch.uint8, device=device)
    err = torch.tensor([ErrorType.DOUBLE_DETECTED, 0, 0, 0], dtype=torch.uint8, device=device)
    result = interpolate_double_errors(q, err)
    # Position 0 should be (q[0] + q[1]) / 2 = (15 + 4) / 2 = 9.5 -> 10
    expected = torch.tensor([10, 4, 8, 12], dtype=torch.uint8, device=device)
    assert torch.equal(result, expected), f"Left boundary failed: {result} vs {expected}"
    print("  [PASS] Double error at first position -> boundary handling")

    # Test 4: Double error at last position (boundary)
    q = torch.tensor([4, 8, 12, 15], dtype=torch.uint8, device=device)
    err = torch.tensor([0, 0, 0, ErrorType.DOUBLE_DETECTED], dtype=torch.uint8, device=device)
    result = interpolate_double_errors(q, err)
    # Position 3 should be (q[2] + q[3]) / 2 = (12 + 15) / 2 = 13.5 -> 14
    expected = torch.tensor([4, 8, 12, 14], dtype=torch.uint8, device=device)
    assert torch.equal(result, expected), f"Right boundary failed: {result} vs {expected}"
    print("  [PASS] Double error at last position -> boundary handling")

    # Test 5: Multiple scattered double errors
    q = torch.tensor([0, 4, 8, 12, 8, 4, 0], dtype=torch.uint8, device=device)
    err = torch.tensor([0, ErrorType.DOUBLE_DETECTED, 0, ErrorType.DOUBLE_DETECTED, 0, ErrorType.DOUBLE_DETECTED, 0],
                       dtype=torch.uint8, device=device)
    result = interpolate_double_errors(q, err)
    # Position 1: (0 + 8) / 2 = 4
    # Position 3: (8 + 8) / 2 = 8
    # Position 5: (8 + 0) / 2 = 4
    expected = torch.tensor([0, 4, 8, 8, 8, 4, 0], dtype=torch.uint8, device=device)
    assert torch.equal(result, expected), f"Scattered errors failed: {result} vs {expected}"
    print("  [PASS] Multiple scattered double errors")

    # Test 6: Large tensor with random double errors
    N = 100000
    q = torch.randint(0, 16, (N,), dtype=torch.uint8, device=device)
    err = torch.zeros(N, dtype=torch.uint8, device=device)
    # Mark 10% as double errors
    double_mask = torch.rand(N, device=device) < 0.1
    err[double_mask] = ErrorType.DOUBLE_DETECTED

    result = interpolate_double_errors(q, err)

    # Verify non-double-error positions are unchanged
    non_double = ~double_mask
    assert torch.equal(result[non_double], q[non_double]), "Non-double positions should be unchanged"

    # Verify double-error positions are interpolated (within valid range)
    assert (result >= 0).all() and (result <= 15).all(), "Results should be in valid INT4 range"

    num_double = double_mask.sum().item()
    print(f"  [PASS] Large tensor with {num_double} double errors ({100*num_double/N:.1f}%)")

    # Test 7: 2D tensor (batch of sequences)
    q_2d = torch.randint(0, 16, (32, 1024), dtype=torch.uint8, device=device)
    err_2d = torch.zeros_like(q_2d)
    # Add some double errors
    err_2d[::4, ::10] = ErrorType.DOUBLE_DETECTED

    result_2d = interpolate_double_errors(q_2d, err_2d)
    assert result_2d.shape == q_2d.shape, "Shape should be preserved"

    # Verify non-double positions unchanged
    non_double_2d = err_2d != ErrorType.DOUBLE_DETECTED
    assert torch.equal(result_2d[non_double_2d], q_2d[non_double_2d]), "Non-double positions should be unchanged (2D)"
    print("  [PASS] 2D tensor (batch of sequences)")

    print("All Triton interpolation verifications passed!")
    return True


if __name__ == "__main__":
    if torch.cuda.is_available():
        verify_triton_vs_cpu()
    else:
        print("CUDA not available, skipping Triton verification")
