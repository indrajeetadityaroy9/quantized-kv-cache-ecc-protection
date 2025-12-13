import torch
import triton
import triton.language as tl
from typing import Tuple

from .config import (
    HAMMING74_BLOCK_SIZE,
    SYNDROME_LUT_HAMMING74,
)


@triton.jit
def hamming74_encode_kernel(
    int4_ptr,
    codeword_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    int4_vals = tl.load(int4_ptr + offsets, mask=mask, other=0).to(tl.uint8)

    d0 = (int4_vals >> 0) & 1
    d1 = (int4_vals >> 1) & 1
    d2 = (int4_vals >> 2) & 1
    d3 = (int4_vals >> 3) & 1

    p0 = d0 ^ d1 ^ d3
    p1 = d0 ^ d2 ^ d3
    p2 = d1 ^ d2 ^ d3

    codeword = (
        (d0 << 0)
        | (d1 << 1)
        | (d2 << 2)
        | (d3 << 3)
        | (p0 << 4)
        | (p1 << 5)
        | (p2 << 6)
    ).to(tl.uint8)

    tl.store(codeword_ptr + offsets, codeword, mask=mask)


@triton.jit
def hamming74_decode_kernel(
    codeword_ptr,
    decoded_ptr,
    error_detected_ptr,
    lut0: tl.constexpr,
    lut1: tl.constexpr,
    lut2: tl.constexpr,
    lut3: tl.constexpr,
    lut4: tl.constexpr,
    lut5: tl.constexpr,
    lut6: tl.constexpr,
    lut7: tl.constexpr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    codewords = tl.load(codeword_ptr + offsets, mask=mask, other=0).to(tl.uint8)

    c0 = (codewords >> 0) & 1
    c1 = (codewords >> 1) & 1
    c2 = (codewords >> 2) & 1
    c3 = (codewords >> 3) & 1
    c4 = (codewords >> 4) & 1
    c5 = (codewords >> 5) & 1
    c6 = (codewords >> 6) & 1

    s0 = c0 ^ c1 ^ c3 ^ c4
    s1 = c0 ^ c2 ^ c3 ^ c5
    s2 = c1 ^ c2 ^ c3 ^ c6

    syndrome = (s0 | (s1 << 1) | (s2 << 2)).to(tl.int32)

    error_detected = (syndrome != 0).to(tl.uint8)

    error_pos = tl.where(
        syndrome == 0,
        lut0,
        tl.where(
            syndrome == 1,
            lut1,
            tl.where(
                syndrome == 2,
                lut2,
                tl.where(
                    syndrome == 3,
                    lut3,
                    tl.where(
                        syndrome == 4,
                        lut4,
                        tl.where(
                            syndrome == 5, lut5, tl.where(syndrome == 6, lut6, lut7)
                        ),
                    ),
                ),
            ),
        ),
    )

    should_correct = error_pos >= 0
    correction_mask = tl.where(should_correct, 1 << error_pos, 0).to(tl.uint8)

    corrected = codewords ^ correction_mask

    decoded = corrected & 0x0F

    tl.store(decoded_ptr + offsets, decoded, mask=mask)
    tl.store(error_detected_ptr + offsets, error_detected, mask=mask)


def hamming74_encode(int4_values: torch.Tensor) -> torch.Tensor:
    assert int4_values.is_cuda, "Input must be on CUDA device"

    original_shape = int4_values.shape
    flat = int4_values.flatten().to(torch.uint8)
    N = flat.numel()

    codewords = torch.empty_like(flat)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    hamming74_encode_kernel[grid](
        flat,
        codewords,
        N,
        BLOCK_SIZE=HAMMING74_BLOCK_SIZE,
    )

    return codewords.view(original_shape)


def hamming74_decode(
    codewords: torch.Tensor,
    return_error_detected: bool = False,
) -> Tuple[torch.Tensor, ...]:
    assert codewords.is_cuda, "Input must be on CUDA device"

    original_shape = codewords.shape
    flat = codewords.flatten().to(torch.uint8)
    N = flat.numel()

    decoded = torch.empty_like(flat)
    error_detected = torch.empty_like(flat)

    lut = SYNDROME_LUT_HAMMING74

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    hamming74_decode_kernel[grid](
        flat,
        decoded,
        error_detected,
        int(lut[0]),
        int(lut[1]),
        int(lut[2]),
        int(lut[3]),
        int(lut[4]),
        int(lut[5]),
        int(lut[6]),
        int(lut[7]),
        N,
        BLOCK_SIZE=HAMMING74_BLOCK_SIZE,
    )

    errors_corrected_count = int(error_detected.sum())

    decoded = decoded.view(original_shape)
    error_detected = error_detected.view(original_shape)

    if return_error_detected:
        return decoded, error_detected, (errors_corrected_count,)
    else:
        return decoded, (errors_corrected_count,)


def verify_triton_vs_cpu():
    from ..hamming74_sec import Hamming74

    device = "cuda"
    cpu_codec = Hamming74(device="cuda")

    print("Verifying Triton Hamming(7,4) vs CPU reference...")

    all_int4 = torch.arange(16, dtype=torch.uint8)

    cpu_codewords = cpu_codec.encode(all_int4)

    triton_codewords = hamming74_encode(all_int4.to(device))

    assert torch.equal(
        cpu_codewords, triton_codewords.cpu()
    ), "Triton encode does not match CPU!"
    print("  [PASS] Encode matches CPU reference")

    triton_decoded, stats = hamming74_decode(triton_codewords)
    assert torch.equal(all_int4.to(device), triton_decoded), "Clean decode failed!"
    assert stats[0] == 0, "Should have no errors"
    print("  [PASS] Clean decode works")

    test_val = torch.tensor([7], dtype=torch.uint8, device=device)
    codeword = hamming74_encode(test_val)

    for bit_pos in range(7):
        corrupted = codeword ^ (1 << bit_pos)
        decoded, error_detected, stats = hamming74_decode(
            corrupted, return_error_detected=True
        )
        assert decoded.item() == 7, f"Single-bit error at bit {bit_pos} not corrected"
        assert error_detected.item() == 1, f"Error at bit {bit_pos} not detected"

    print("  [PASS] Single-bit error correction works (all 7 positions)")

    large_input = torch.randint(0, 16, (100000,), dtype=torch.uint8, device=device)
    encoded = hamming74_encode(large_input)
    decoded, stats = hamming74_decode(encoded)
    assert torch.equal(large_input, decoded), "Large tensor roundtrip failed!"
    assert stats[0] == 0, "Clean data should have no corrections"
    print("  [PASS] Large tensor roundtrip works (100K values)")

    encoded_with_errors = encoded.clone()
    error_mask = torch.zeros(100000, dtype=torch.bool, device=device)
    error_mask[::2] = True
    bit_positions = torch.randint(0, 7, (50000,), device=device)
    error_pattern = (1 << bit_positions).to(torch.uint8)
    encoded_with_errors[::2] ^= error_pattern

    decoded, stats = hamming74_decode(encoded_with_errors)
    expected_errors = 50000
    assert (
        stats[0] == expected_errors
    ), f"Expected {expected_errors} corrections, got {stats[0]}"
    assert torch.equal(decoded, large_input), "Corrected data doesn't match original!"
    print(f"  [PASS] Error correction statistics correct ({stats[0]} errors corrected)")

    print("All Triton Hamming(7,4) verifications passed!")
    return True


if __name__ == "__main__":
    verify_triton_vs_cpu()
