import torch
import triton
import triton.language as tl

from .config import FAULT_INJECTION_BLOCK_SIZE


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
