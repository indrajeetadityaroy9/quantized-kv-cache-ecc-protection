"""
Triton GPU Kernels for Deterministic Fault Injection.

GPU-native implementation of bit error injection using Philox counter-based RNG.
Eliminates CPU-GPU transfer overhead while maintaining reproducibility.

Key features:
- Philox RNG: Deterministic, parallel-friendly PRNG
- Per-element seeding: seed + offset ensures reproducibility
- Bitwise XOR: Standard binary channel model
- No CPU roundtrip: All operations on GPU
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional

from .config import FAULT_INJECTION_BLOCK_SIZE


# =============================================================================
# Triton Fault Injection Kernels
# =============================================================================

@triton.jit
def fault_inject_uint8_kernel(
    # Pointers
    data_ptr,           # Input: codewords [N], dtype=uint8
    output_ptr,         # Output: corrupted codewords [N], dtype=uint8
    error_count_ptr,    # Output: error counts per element [N], dtype=uint8
    # Parameters
    N,                  # Total number of elements
    n_bits: tl.constexpr,  # Number of active bits per element (1-8)
    seed,               # Random seed (combined with offset for determinism)
    ber_threshold,      # Threshold for bit flip (ber * 2^31)
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Inject bit errors into uint8 tensor using Philox RNG.

    For each bit position in each element:
    - Generate random value using tl.rand (Philox-based)
    - Flip bit if random < BER threshold

    The seed is combined with element offset for reproducibility:
    random[element, bit] = philox(seed + element * n_bits + bit)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input data
    data = tl.load(data_ptr + offsets, mask=mask, other=0).to(tl.uint8)

    # Initialize error mask and count
    error_mask = tl.zeros([BLOCK_SIZE], dtype=tl.uint8)
    error_count = tl.zeros([BLOCK_SIZE], dtype=tl.uint8)

    # Process each bit position
    # Using tl.rand with unique seed per (element, bit) pair
    # Philox counter: seed * N * n_bits + offset * n_bits + bit
    base_seed = seed * (N * n_bits) + offsets * n_bits

    # Bit 0
    rand_val_0 = tl.rand(base_seed.to(tl.int32), offsets.to(tl.int32))
    flip_0 = (rand_val_0 < ber_threshold).to(tl.uint8)
    error_mask = error_mask | (flip_0 << 0)
    error_count = error_count + flip_0

    # Bit 1 (if n_bits >= 2)
    if n_bits >= 2:
        rand_val_1 = tl.rand((base_seed + 1).to(tl.int32), offsets.to(tl.int32))
        flip_1 = (rand_val_1 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_1 << 1)
        error_count = error_count + flip_1

    # Bit 2 (if n_bits >= 3)
    if n_bits >= 3:
        rand_val_2 = tl.rand((base_seed + 2).to(tl.int32), offsets.to(tl.int32))
        flip_2 = (rand_val_2 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_2 << 2)
        error_count = error_count + flip_2

    # Bit 3 (if n_bits >= 4)
    if n_bits >= 4:
        rand_val_3 = tl.rand((base_seed + 3).to(tl.int32), offsets.to(tl.int32))
        flip_3 = (rand_val_3 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_3 << 3)
        error_count = error_count + flip_3

    # Bit 4 (if n_bits >= 5)
    if n_bits >= 5:
        rand_val_4 = tl.rand((base_seed + 4).to(tl.int32), offsets.to(tl.int32))
        flip_4 = (rand_val_4 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_4 << 4)
        error_count = error_count + flip_4

    # Bit 5 (if n_bits >= 6)
    if n_bits >= 6:
        rand_val_5 = tl.rand((base_seed + 5).to(tl.int32), offsets.to(tl.int32))
        flip_5 = (rand_val_5 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_5 << 5)
        error_count = error_count + flip_5

    # Bit 6 (if n_bits >= 7)
    if n_bits >= 7:
        rand_val_6 = tl.rand((base_seed + 6).to(tl.int32), offsets.to(tl.int32))
        flip_6 = (rand_val_6 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_6 << 6)
        error_count = error_count + flip_6

    # Bit 7 (if n_bits >= 8)
    if n_bits >= 8:
        rand_val_7 = tl.rand((base_seed + 7).to(tl.int32), offsets.to(tl.int32))
        flip_7 = (rand_val_7 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_7 << 7)
        error_count = error_count + flip_7

    # Apply error mask via XOR
    corrupted = data ^ error_mask

    # Store outputs
    tl.store(output_ptr + offsets, corrupted, mask=mask)
    tl.store(error_count_ptr + offsets, error_count, mask=mask)


@triton.jit
def fault_inject_int32_kernel(
    # Pointers
    data_ptr,           # Input: codewords [N], dtype=int32
    output_ptr,         # Output: corrupted codewords [N], dtype=int32
    error_count_ptr,    # Output: error counts per element [N], dtype=uint8
    # Parameters
    N,                  # Total number of elements
    n_bits: tl.constexpr,  # Number of active bits per element (1-24 for Golay)
    seed,               # Random seed
    ber_threshold,      # Threshold for bit flip
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Inject bit errors into int32 tensor using Philox RNG.

    Designed for Golay(24,12) codewords stored as int32.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input data
    data = tl.load(data_ptr + offsets, mask=mask, other=0).to(tl.int32)

    # Initialize error mask and count
    error_mask = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    error_count = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    # Base seed for this block
    base_seed = seed * (N * n_bits) + offsets * n_bits

    # Process each bit position (unrolled for common cases)
    # For Golay(24,12), n_bits = 24
    for bit in range(24):  # Max 24 bits for Golay
        if bit < n_bits:
            rand_val = tl.rand((base_seed + bit).to(tl.int32), offsets.to(tl.int32))
            flip = (rand_val < ber_threshold).to(tl.int32)
            error_mask = error_mask | (flip << bit)
            error_count = error_count + flip

    # Apply error mask via XOR
    corrupted = data ^ error_mask

    # Store outputs
    tl.store(output_ptr + offsets, corrupted, mask=mask)
    tl.store(error_count_ptr + offsets, error_count.to(tl.uint8), mask=mask)


# =============================================================================
# Python Wrapper Functions
# =============================================================================

def inject_bit_errors_triton(
    data: torch.Tensor,
    ber: float,
    n_bits: int,
    seed: int = 0,
    return_stats: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """
    Inject bit errors using Triton GPU kernels with Philox RNG.

    Args:
        data: Input tensor (uint8 or int32) on CUDA
        ber: Bit Error Rate (0.0 to 1.0)
        n_bits: Number of active bits per element
        seed: Random seed for reproducibility
        return_stats: If True, return error statistics

    Returns:
        corrupted: Tensor with errors injected
        stats: (optional) Tuple of (total_errors, elements_with_errors)
    """
    assert data.is_cuda, "Input must be on CUDA device"

    if ber <= 0:
        if return_stats:
            return data, (0, 0)
        return data

    N = data.numel()
    device = data.device
    original_shape = data.shape

    # Flatten input
    flat_data = data.flatten()

    # Allocate outputs
    if flat_data.dtype == torch.uint8:
        corrupted = torch.empty_like(flat_data)
        error_counts = torch.empty(N, dtype=torch.uint8, device=device)

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        fault_inject_uint8_kernel[grid](
            flat_data, corrupted, error_counts,
            N, n_bits, seed, ber,
            BLOCK_SIZE=FAULT_INJECTION_BLOCK_SIZE,
        )

    elif flat_data.dtype == torch.int32:
        corrupted = torch.empty_like(flat_data)
        error_counts = torch.empty(N, dtype=torch.uint8, device=device)

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        fault_inject_int32_kernel[grid](
            flat_data, corrupted, error_counts,
            N, n_bits, seed, ber,
            BLOCK_SIZE=FAULT_INJECTION_BLOCK_SIZE,
        )

    else:
        raise ValueError(f"Unsupported dtype: {flat_data.dtype}. Use uint8 or int32.")

    # Reshape to original
    corrupted = corrupted.view(original_shape)

    if return_stats:
        total_errors = int(error_counts.sum())
        elements_with_errors = int((error_counts > 0).sum())
        return corrupted, (total_errors, elements_with_errors)

    return corrupted


def inject_bit_errors_triton_batched(
    data: torch.Tensor,
    ber: float,
    n_bits: int,
    seed: int = 0,
) -> Tuple[torch.Tensor, int]:
    """
    Inject bit errors and return corrupted data with error count.

    Convenience wrapper matching the CPU API.

    Args:
        data: Input tensor on CUDA
        ber: Bit Error Rate
        n_bits: Number of active bits per element
        seed: Random seed

    Returns:
        corrupted: Data with errors injected
        total_errors: Total number of bit flips
    """
    corrupted, (total_errors, _) = inject_bit_errors_triton(
        data, ber, n_bits, seed, return_stats=True
    )
    return corrupted, total_errors


# =============================================================================
# Verification
# =============================================================================

def verify_triton_fault_injection(
    target_ber: float = 0.05,
    n_values: int = 100_000,
    n_bits: int = 8,
    seed: int = 42,
    tolerance: float = 0.01,
) -> dict:
    """
    Verify that Triton fault injection matches target BER.

    Args:
        target_ber: Target bit error rate
        n_values: Number of values to test
        n_bits: Bits per value
        seed: Random seed
        tolerance: Acceptable deviation from target

    Returns:
        Dictionary with verification results
    """
    device = "cuda"

    # Test with all-zeros (makes counting errors trivial)
    if n_bits <= 8:
        data = torch.zeros(n_values, dtype=torch.uint8, device=device)
    else:
        data = torch.zeros(n_values, dtype=torch.int32, device=device)

    corrupted, (total_errors, _) = inject_bit_errors_triton(
        data, target_ber, n_bits, seed, return_stats=True
    )

    total_bits = n_values * n_bits
    empirical_ber = total_errors / total_bits
    deviation = abs(empirical_ber - target_ber)

    result = {
        "target_ber": target_ber,
        "empirical_ber": empirical_ber,
        "deviation": deviation,
        "total_bits": total_bits,
        "total_errors": total_errors,
        "passed": deviation < tolerance,
    }

    return result


def verify_determinism(
    ber: float = 0.1,
    n_values: int = 10_000,
    n_bits: int = 8,
    seed: int = 42,
) -> bool:
    """
    Verify that Triton fault injection is deterministic.

    Same seed should produce identical error patterns.
    """
    device = "cuda"

    if n_bits <= 8:
        data = torch.randint(0, 256, (n_values,), dtype=torch.uint8, device=device)
    else:
        data = torch.randint(0, 2**24, (n_values,), dtype=torch.int32, device=device)

    # Run twice with same seed
    corrupted1 = inject_bit_errors_triton(data, ber, n_bits, seed)
    corrupted2 = inject_bit_errors_triton(data, ber, n_bits, seed)

    return torch.equal(corrupted1, corrupted2)


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Triton Fault Injection Verification")
        print("=" * 60)

        # Test BER fidelity
        print("\n1. BER Fidelity Test:")
        for target_ber in [0.01, 0.05, 0.10]:
            result = verify_triton_fault_injection(
                target_ber=target_ber,
                n_values=100_000,
                n_bits=8,
                seed=42,
            )
            status = "PASS" if result["passed"] else "FAIL"
            print(f"   BER={target_ber:.2f}: empirical={result['empirical_ber']:.4f} [{status}]")

        # Test determinism
        print("\n2. Determinism Test:")
        is_deterministic = verify_determinism(ber=0.1, n_values=10_000, seed=42)
        print(f"   Same seed produces same errors: {is_deterministic}")

        # Test with Golay (24-bit)
        print("\n3. Golay (24-bit) Test:")
        result = verify_triton_fault_injection(
            target_ber=0.05,
            n_values=10_000,
            n_bits=24,
            seed=42,
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(f"   BER=0.05 on 24-bit: empirical={result['empirical_ber']:.4f} [{status}]")

        print("\nAll verifications completed!")
    else:
        print("CUDA not available, skipping Triton verification")
