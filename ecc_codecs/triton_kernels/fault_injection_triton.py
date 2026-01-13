"""
GPU-Accelerated Bernoulli Bit-Flip Fault Injection.

This module implements deterministic bit error injection for simulating
memory corruption in BER (Bit Error Rate) sweep experiments. Each bit
position is independently flipped with probability `ber`.

Fault Model:
    - Bernoulli independent bit flips: P(bit_i flipped) = ber
    - Applied to individual bits within codewords or raw data
    - Supports variable codeword widths (4-bit INT4, 7-bit Hamming, 8-bit, 24-bit Golay)

Determinism:
    The RNG uses seed = base_seed * N + offset * n_bits + bit_idx, ensuring:
    - Same seed → identical bit flip pattern
    - Different positions get different random sequences
    - Different seeds → statistically independent patterns

Performance:
    Two kernel variants are provided:
    - Standard: One tl.rand() call per bit (simpler, baseline)
    - Vectorized: Uses tl.rand4x() for 4 random values per call (2-3x faster)

    For 8-bit data: 2 rand4x calls vs 8 rand calls
    For 24-bit data: 6 rand4x calls vs 24 rand calls

Supported Data Types:
    - uint8: For Hamming(7,4), Hamming(8,4), INT4, FP8 codewords
    - int32: For Golay(24,12) 24-bit codewords

Usage:
    # Inject errors with 1% BER into Hamming(8,4) codewords
    corrupted = inject_bit_errors_triton(
        encoded_data,   # uint8 tensor
        ber=0.01,       # 1% bit error rate
        n_bits=8,       # 8 bits per codeword
        seed=42,        # Deterministic seed
    )

    # With statistics
    corrupted, (total_flips, elements_affected) = inject_bit_errors_triton(
        encoded_data, ber=0.01, n_bits=8, seed=42, return_stats=True
    )

BER Verification:
    The verify_triton_fault_injection() function confirms that empirical
    BER matches target BER within tolerance, validating the RNG quality.
"""
import torch
import triton
import triton.language as tl

from .config import FAULT_INJECTION_BLOCK_SIZE


@triton.jit
def fault_inject_uint8_vectorized_kernel(
    data_ptr,
    output_ptr,
    error_count_ptr,
    N,
    n_bits: tl.constexpr,
    seed,
    ber_threshold,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Vectorized fault injection using rand4x to reduce RNG overhead.

    Uses 2 rand4x calls (8 random values) instead of 8 rand calls.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    data = tl.load(data_ptr + offsets, mask=mask, other=0).to(tl.uint8)

    error_mask = tl.zeros([BLOCK_SIZE], dtype=tl.uint8)
    error_count = tl.zeros([BLOCK_SIZE], dtype=tl.uint8)

    # Use different seed bases for the two rand4x calls
    base_seed = seed * N + offsets

    # First rand4x call - bits 0-3
    r0, r1, r2, r3 = tl.rand4x(base_seed.to(tl.int32), offsets.to(tl.int32))

    flip_0 = (r0 < ber_threshold).to(tl.uint8)
    error_mask = error_mask | (flip_0 << 0)
    error_count = error_count + flip_0

    if n_bits >= 2:
        flip_1 = (r1 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_1 << 1)
        error_count = error_count + flip_1

    if n_bits >= 3:
        flip_2 = (r2 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_2 << 2)
        error_count = error_count + flip_2

    if n_bits >= 4:
        flip_3 = (r3 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_3 << 3)
        error_count = error_count + flip_3

    # Second rand4x call - bits 4-7
    if n_bits >= 5:
        # Use offset + N as different seed for second batch
        r4, r5, r6, r7 = tl.rand4x((base_seed + N).to(tl.int32), offsets.to(tl.int32))

        flip_4 = (r4 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_4 << 4)
        error_count = error_count + flip_4

        if n_bits >= 6:
            flip_5 = (r5 < ber_threshold).to(tl.uint8)
            error_mask = error_mask | (flip_5 << 5)
            error_count = error_count + flip_5

        if n_bits >= 7:
            flip_6 = (r6 < ber_threshold).to(tl.uint8)
            error_mask = error_mask | (flip_6 << 6)
            error_count = error_count + flip_6

        if n_bits >= 8:
            flip_7 = (r7 < ber_threshold).to(tl.uint8)
            error_mask = error_mask | (flip_7 << 7)
            error_count = error_count + flip_7

    corrupted = data ^ error_mask

    tl.store(output_ptr + offsets, corrupted, mask=mask)
    tl.store(error_count_ptr + offsets, error_count, mask=mask)


@triton.jit
def fault_inject_int32_vectorized_kernel(
    data_ptr,
    output_ptr,
    error_count_ptr,
    N,
    n_bits: tl.constexpr,
    seed,
    ber_threshold,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Vectorized fault injection for int32 using rand4x.

    Uses 6 rand4x calls (24 random values) instead of 24 rand calls for 24-bit.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    data = tl.load(data_ptr + offsets, mask=mask, other=0).to(tl.int32)

    error_mask = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    error_count = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    base_seed = seed * N + offsets

    # Batch 0: bits 0-3
    r0, r1, r2, r3 = tl.rand4x(base_seed.to(tl.int32), offsets.to(tl.int32))
    for i, r in enumerate([r0, r1, r2, r3]):
        if i < n_bits:
            flip = (r < ber_threshold).to(tl.int32)
            error_mask = error_mask | (flip << i)
            error_count = error_count + flip

    # Batch 1: bits 4-7
    if n_bits > 4:
        r4, r5, r6, r7 = tl.rand4x((base_seed + N).to(tl.int32), offsets.to(tl.int32))
        for i, r in enumerate([r4, r5, r6, r7]):
            bit_idx = 4 + i
            if bit_idx < n_bits:
                flip = (r < ber_threshold).to(tl.int32)
                error_mask = error_mask | (flip << bit_idx)
                error_count = error_count + flip

    # Batch 2: bits 8-11
    if n_bits > 8:
        r8, r9, r10, r11 = tl.rand4x((base_seed + 2*N).to(tl.int32), offsets.to(tl.int32))
        for i, r in enumerate([r8, r9, r10, r11]):
            bit_idx = 8 + i
            if bit_idx < n_bits:
                flip = (r < ber_threshold).to(tl.int32)
                error_mask = error_mask | (flip << bit_idx)
                error_count = error_count + flip

    # Batch 3: bits 12-15
    if n_bits > 12:
        r12, r13, r14, r15 = tl.rand4x((base_seed + 3*N).to(tl.int32), offsets.to(tl.int32))
        for i, r in enumerate([r12, r13, r14, r15]):
            bit_idx = 12 + i
            if bit_idx < n_bits:
                flip = (r < ber_threshold).to(tl.int32)
                error_mask = error_mask | (flip << bit_idx)
                error_count = error_count + flip

    # Batch 4: bits 16-19
    if n_bits > 16:
        r16, r17, r18, r19 = tl.rand4x((base_seed + 4*N).to(tl.int32), offsets.to(tl.int32))
        for i, r in enumerate([r16, r17, r18, r19]):
            bit_idx = 16 + i
            if bit_idx < n_bits:
                flip = (r < ber_threshold).to(tl.int32)
                error_mask = error_mask | (flip << bit_idx)
                error_count = error_count + flip

    # Batch 5: bits 20-23
    if n_bits > 20:
        r20, r21, r22, r23 = tl.rand4x((base_seed + 5*N).to(tl.int32), offsets.to(tl.int32))
        for i, r in enumerate([r20, r21, r22, r23]):
            bit_idx = 20 + i
            if bit_idx < n_bits:
                flip = (r < ber_threshold).to(tl.int32)
                error_mask = error_mask | (flip << bit_idx)
                error_count = error_count + flip

    corrupted = data ^ error_mask

    tl.store(output_ptr + offsets, corrupted, mask=mask)
    tl.store(error_count_ptr + offsets, error_count.to(tl.uint8), mask=mask)


@triton.jit
def fault_inject_uint8_kernel(
    data_ptr,
    output_ptr,
    error_count_ptr,
    N,
    n_bits: tl.constexpr,
    seed,
    ber_threshold,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    data = tl.load(data_ptr + offsets, mask=mask, other=0).to(tl.uint8)

    error_mask = tl.zeros([BLOCK_SIZE], dtype=tl.uint8)
    error_count = tl.zeros([BLOCK_SIZE], dtype=tl.uint8)

    base_seed = seed * (N * n_bits) + offsets * n_bits

    rand_val_0 = tl.rand(base_seed.to(tl.int32), offsets.to(tl.int32))
    flip_0 = (rand_val_0 < ber_threshold).to(tl.uint8)
    error_mask = error_mask | (flip_0 << 0)
    error_count = error_count + flip_0

    if n_bits >= 2:
        rand_val_1 = tl.rand((base_seed + 1).to(tl.int32), offsets.to(tl.int32))
        flip_1 = (rand_val_1 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_1 << 1)
        error_count = error_count + flip_1

    if n_bits >= 3:
        rand_val_2 = tl.rand((base_seed + 2).to(tl.int32), offsets.to(tl.int32))
        flip_2 = (rand_val_2 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_2 << 2)
        error_count = error_count + flip_2

    if n_bits >= 4:
        rand_val_3 = tl.rand((base_seed + 3).to(tl.int32), offsets.to(tl.int32))
        flip_3 = (rand_val_3 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_3 << 3)
        error_count = error_count + flip_3

    if n_bits >= 5:
        rand_val_4 = tl.rand((base_seed + 4).to(tl.int32), offsets.to(tl.int32))
        flip_4 = (rand_val_4 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_4 << 4)
        error_count = error_count + flip_4

    if n_bits >= 6:
        rand_val_5 = tl.rand((base_seed + 5).to(tl.int32), offsets.to(tl.int32))
        flip_5 = (rand_val_5 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_5 << 5)
        error_count = error_count + flip_5

    if n_bits >= 7:
        rand_val_6 = tl.rand((base_seed + 6).to(tl.int32), offsets.to(tl.int32))
        flip_6 = (rand_val_6 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_6 << 6)
        error_count = error_count + flip_6

    if n_bits >= 8:
        rand_val_7 = tl.rand((base_seed + 7).to(tl.int32), offsets.to(tl.int32))
        flip_7 = (rand_val_7 < ber_threshold).to(tl.uint8)
        error_mask = error_mask | (flip_7 << 7)
        error_count = error_count + flip_7

    corrupted = data ^ error_mask

    tl.store(output_ptr + offsets, corrupted, mask=mask)
    tl.store(error_count_ptr + offsets, error_count, mask=mask)


@triton.jit
def fault_inject_int32_kernel(
    data_ptr,
    output_ptr,
    error_count_ptr,
    N,
    n_bits: tl.constexpr,
    seed,
    ber_threshold,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    data = tl.load(data_ptr + offsets, mask=mask, other=0).to(tl.int32)

    error_mask = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    error_count = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    base_seed = seed * (N * n_bits) + offsets * n_bits

    for bit in range(24):
        if bit < n_bits:
            rand_val = tl.rand((base_seed + bit).to(tl.int32), offsets.to(tl.int32))
            flip = (rand_val < ber_threshold).to(tl.int32)
            error_mask = error_mask | (flip << bit)
            error_count = error_count + flip

    corrupted = data ^ error_mask

    tl.store(output_ptr + offsets, corrupted, mask=mask)
    tl.store(error_count_ptr + offsets, error_count.to(tl.uint8), mask=mask)


def inject_bit_errors_triton(data, ber, n_bits, seed=0, return_stats=False):
    """
    Inject Bernoulli bit-flip errors into codewords at specified BER.

    Each of the lower `n_bits` bits in each element is independently flipped
    with probability `ber`. Uses GPU-accelerated RNG for high throughput.

    Args:
        data: Input tensor (uint8 or int32) containing codewords
        ber: Bit Error Rate in [0, 1]. Probability of flipping each bit.
        n_bits: Number of bits per codeword to corrupt (e.g., 4, 7, 8, 24)
        seed: RNG seed for deterministic injection. Same seed = same flips.
        return_stats: If True, return (corrupted, (total_flips, elements_affected))

    Returns:
        If return_stats=False: Corrupted tensor with same shape/dtype as input
        If return_stats=True: Tuple of (corrupted_tensor, (total_bit_flips, num_elements_with_errors))

    Determinism:
        The RNG is seeded deterministically per-element and per-bit:
            bit_seed = seed * N * n_bits + offset * n_bits + bit_idx
        This ensures reproducibility across runs with the same seed.

    Examples:
        # Hamming(8,4) with 1% BER
        corrupted = inject_bit_errors_triton(encoded, 0.01, 8, seed=42)

        # Golay(24,12) with statistics
        corrupted, (flips, affected) = inject_bit_errors_triton(
            golay_encoded, 0.001, 24, seed=42, return_stats=True
        )
    """
    assert data.is_cuda, "Input must be on CUDA device"

    if ber <= 0:
        if return_stats:
            return data, (0, 0)
        return data

    N = data.numel()
    device = data.device
    original_shape = data.shape

    flat_data = data.flatten()

    if flat_data.dtype == torch.uint8:
        corrupted = torch.empty_like(flat_data)
        error_counts = torch.empty(N, dtype=torch.uint8, device=device)

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        fault_inject_uint8_kernel[grid](
            flat_data,
            corrupted,
            error_counts,
            N,
            n_bits,
            seed,
            ber,
            BLOCK_SIZE=FAULT_INJECTION_BLOCK_SIZE,
        )

    elif flat_data.dtype == torch.int32:
        corrupted = torch.empty_like(flat_data)
        error_counts = torch.empty(N, dtype=torch.uint8, device=device)

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        fault_inject_int32_kernel[grid](
            flat_data,
            corrupted,
            error_counts,
            N,
            n_bits,
            seed,
            ber,
            BLOCK_SIZE=FAULT_INJECTION_BLOCK_SIZE,
        )

    else:
        raise ValueError(f"Unsupported dtype: {flat_data.dtype}. Use uint8 or int32.")

    corrupted = corrupted.view(original_shape)

    if return_stats:
        total_errors = int(error_counts.sum())
        elements_with_errors = int((error_counts > 0).sum())
        return corrupted, (total_errors, elements_with_errors)

    return corrupted


def inject_bit_errors_triton_batched(data, ber, n_bits, seed=0):
    corrupted, (total_errors, _) = inject_bit_errors_triton(
        data, ber, n_bits, seed, return_stats=True
    )
    return corrupted, total_errors


def inject_bit_errors_triton_vectorized(data, ber, n_bits, seed=0, return_stats=False):
    """
    Vectorized fault injection using rand4x for better performance.

    Uses 2 rand4x calls (8 random values) instead of 8 rand calls for uint8,
    and 6 rand4x calls instead of 24 for int32/24-bit.
    """
    assert data.is_cuda, "Input must be on CUDA device"

    if ber <= 0:
        if return_stats:
            return data, (0, 0)
        return data

    N = data.numel()
    device = data.device
    original_shape = data.shape

    flat_data = data.flatten()

    if flat_data.dtype == torch.uint8:
        corrupted = torch.empty_like(flat_data)
        error_counts = torch.empty(N, dtype=torch.uint8, device=device)

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        fault_inject_uint8_vectorized_kernel[grid](
            flat_data,
            corrupted,
            error_counts,
            N,
            n_bits,
            seed,
            ber,
            BLOCK_SIZE=FAULT_INJECTION_BLOCK_SIZE,
        )

    elif flat_data.dtype == torch.int32:
        corrupted = torch.empty_like(flat_data)
        error_counts = torch.empty(N, dtype=torch.uint8, device=device)

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        fault_inject_int32_vectorized_kernel[grid](
            flat_data,
            corrupted,
            error_counts,
            N,
            n_bits,
            seed,
            ber,
            BLOCK_SIZE=FAULT_INJECTION_BLOCK_SIZE,
        )

    else:
        raise ValueError(f"Unsupported dtype: {flat_data.dtype}. Use uint8 or int32.")

    corrupted = corrupted.view(original_shape)

    if return_stats:
        total_errors = int(error_counts.sum())
        elements_with_errors = int((error_counts > 0).sum())
        return corrupted, (total_errors, elements_with_errors)

    return corrupted


def verify_triton_fault_injection(target_ber=0.05, n_values=100_000, n_bits=8, seed=42, tolerance=0.01):
    device = "cuda"

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


def verify_determinism(ber=0.1, n_values=10_000, n_bits=8, seed=42):
    device = "cuda"

    if n_bits <= 8:
        data = torch.randint(0, 256, (n_values,), dtype=torch.uint8, device=device)
    else:
        data = torch.randint(0, 2**24, (n_values,), dtype=torch.int32, device=device)

    corrupted1 = inject_bit_errors_triton(data, ber, n_bits, seed)
    corrupted2 = inject_bit_errors_triton(data, ber, n_bits, seed)

    return torch.equal(corrupted1, corrupted2)


if __name__ == "__main__":
    print("Triton Fault Injection Verification")
    print("=" * 60)

    print("\n1. BER Fidelity Test:")
    for target_ber in [0.01, 0.05, 0.10]:
        result = verify_triton_fault_injection(
            target_ber=target_ber,
            n_values=100_000,
            n_bits=8,
            seed=42,
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"   BER={target_ber:.2f}: empirical={result['empirical_ber']:.4f} [{status}]"
        )

    print("\n2. Determinism Test:")
    is_deterministic = verify_determinism(ber=0.1, n_values=10_000, seed=42)
    print(f"   Same seed produces same errors: {is_deterministic}")

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
