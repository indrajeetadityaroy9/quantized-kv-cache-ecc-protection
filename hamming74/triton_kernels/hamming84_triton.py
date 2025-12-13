import torch
import triton
import triton.language as tl
from typing import Tuple

from .config import (
    HAMMING84_BLOCK_SIZE,
    SYNDROME_LUT_HAMMING84,
    ErrorType,
)


@triton.jit
def hamming84_encode_kernel(
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

    hamming7 = (
        (d0 << 0)
        | (d1 << 1)
        | (d2 << 2)
        | (d3 << 3)
        | (p0 << 4)
        | (p1 << 5)
        | (p2 << 6)
    )

    parity = hamming7 ^ (hamming7 >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    overall_parity = parity & 1

    codeword = (hamming7 | (overall_parity << 7)).to(tl.uint8)

    tl.store(codeword_ptr + offsets, codeword, mask=mask)


@triton.jit
def hamming84_decode_kernel(
    codeword_ptr,
    decoded_ptr,
    error_type_ptr,
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

    hamming7 = codewords & 0x7F
    stored_parity = (codewords >> 7) & 1

    c0 = (hamming7 >> 0) & 1
    c1 = (hamming7 >> 1) & 1
    c2 = (hamming7 >> 2) & 1
    c3 = (hamming7 >> 3) & 1
    c4 = (hamming7 >> 4) & 1
    c5 = (hamming7 >> 5) & 1
    c6 = (hamming7 >> 6) & 1

    s0 = c0 ^ c1 ^ c3 ^ c4
    s1 = c0 ^ c2 ^ c3 ^ c5
    s2 = c1 ^ c2 ^ c3 ^ c6

    syndrome = (s0 | (s1 << 1) | (s2 << 2)).to(tl.int32)

    actual_parity = hamming7 ^ (hamming7 >> 4)
    actual_parity = actual_parity ^ (actual_parity >> 2)
    actual_parity = actual_parity ^ (actual_parity >> 1)
    actual_parity = actual_parity & 1

    parity_error = stored_parity != actual_parity

    syndrome_zero = syndrome == 0

    error_type = tl.where(
        syndrome_zero, tl.where(parity_error, 3, 0), tl.where(parity_error, 1, 2)
    ).to(tl.uint8)

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

    should_correct = (error_type == 1) & (error_pos >= 0)
    correction_mask = tl.where(should_correct, 1 << error_pos, 0).to(tl.uint8)

    corrected = hamming7 ^ correction_mask

    corrected = tl.where(error_type == 2, 0, corrected).to(tl.uint8)

    decoded = corrected & 0x0F

    tl.store(decoded_ptr + offsets, decoded, mask=mask)
    tl.store(error_type_ptr + offsets, error_type, mask=mask)


def hamming84_encode(int4_values: torch.Tensor) -> torch.Tensor:
    assert int4_values.is_cuda, "Input must be on CUDA device"

    original_shape = int4_values.shape
    flat = int4_values.flatten().to(torch.uint8)
    N = flat.numel()

    codewords = torch.empty_like(flat)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    hamming84_encode_kernel[grid](
        flat,
        codewords,
        N,
        BLOCK_SIZE=HAMMING84_BLOCK_SIZE,
    )

    return codewords.view(original_shape)


def hamming84_decode(
    codewords: torch.Tensor,
    return_error_types: bool = False,
) -> Tuple[torch.Tensor, ...]:
    assert codewords.is_cuda, "Input must be on CUDA device"

    original_shape = codewords.shape
    flat = codewords.flatten().to(torch.uint8)
    N = flat.numel()

    decoded = torch.empty_like(flat)
    error_types = torch.empty_like(flat)

    lut = SYNDROME_LUT_HAMMING84

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    hamming84_decode_kernel[grid](
        flat,
        decoded,
        error_types,
        int(lut[0]),
        int(lut[1]),
        int(lut[2]),
        int(lut[3]),
        int(lut[4]),
        int(lut[5]),
        int(lut[6]),
        int(lut[7]),
        N,
        BLOCK_SIZE=HAMMING84_BLOCK_SIZE,
    )

    error_types_flat = error_types
    corrected_count = int((error_types_flat == ErrorType.SINGLE_CORRECTED).sum())
    detected_count = int((error_types_flat == ErrorType.DOUBLE_DETECTED).sum())

    decoded = decoded.view(original_shape)
    error_types = error_types.view(original_shape)

    if return_error_types:
        return decoded, error_types, (corrected_count, detected_count)
    else:
        return decoded, (corrected_count, detected_count)


def verify_triton_vs_cpu():
    from ..hamming84_secded import Hamming84

    device = "cuda"
    cpu_codec = Hamming84(device="cuda", on_double_error="zero")

    print("Verifying Triton Hamming(8,4) vs CPU reference...")

    all_int4 = torch.arange(16, dtype=torch.uint8)

    cpu_codewords = cpu_codec.encode(all_int4)

    triton_codewords = hamming84_encode(all_int4.to(device))

    assert torch.equal(
        cpu_codewords, triton_codewords.cpu()
    ), "Triton encode does not match CPU!"
    print("  [PASS] Encode matches CPU reference")

    triton_decoded, stats = hamming84_decode(triton_codewords)
    assert torch.equal(all_int4.to(device), triton_decoded), "Clean decode failed!"
    assert stats[0] == 0 and stats[1] == 0, "Should have no errors"
    print("  [PASS] Clean decode works")

    test_val = torch.tensor([7], dtype=torch.uint8, device=device)
    codeword = hamming84_encode(test_val)

    for bit_pos in range(8):
        corrupted = codeword ^ (1 << bit_pos)
        decoded, error_types, stats = hamming84_decode(
            corrupted, return_error_types=True
        )
        assert decoded.item() == 7, f"Single-bit error at bit {bit_pos} not corrected"

    print("  [PASS] Single-bit error correction works")

    for bit1 in range(7):
        for bit2 in range(bit1 + 1, 7):
            corrupted = codeword ^ (1 << bit1) ^ (1 << bit2)
            decoded, error_types, stats = hamming84_decode(
                corrupted, return_error_types=True
            )

            assert (
                decoded.item() == 0 or error_types.item() == ErrorType.DOUBLE_DETECTED
            ), f"Double-bit error at bits {bit1},{bit2} not handled correctly"

    print("  [PASS] Double-bit error detection works")

    large_input = torch.randint(0, 16, (100000,), dtype=torch.uint8, device=device)
    encoded = hamming84_encode(large_input)
    decoded, stats = hamming84_decode(encoded)
    assert torch.equal(large_input, decoded), "Large tensor roundtrip failed!"
    print("  [PASS] Large tensor roundtrip works")

    print("All Triton Hamming(8,4) verifications passed!")
    return True


if __name__ == "__main__":
    verify_triton_vs_cpu()
