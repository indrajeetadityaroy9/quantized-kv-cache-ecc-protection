import torch
import triton
import triton.language as tl

from .config import GOLAY_BLOCK_SIZE, GOLAY_PARITY_MASKS
from ..golay import Golay2412


B_COL_0 = 0b101000111011
B_COL_1 = 0b110100011101
B_COL_2 = 0b111010001110
B_COL_3 = 0b101101000111
B_COL_4 = 0b110110100011
B_COL_5 = 0b111011010001
B_COL_6 = 0b111101101000
B_COL_7 = 0b101110110100
B_COL_8 = 0b100111011010
B_COL_9 = 0b100011101101
B_COL_10 = 0b110001110110
B_COL_11 = 0b011111111111


@triton.jit
def _popcount_mod2_12bit(x):
    x = x ^ (x >> 8)
    x = x ^ (x >> 4)
    x = x ^ (x >> 2)
    x = x ^ (x >> 1)
    return x & 1


@triton.jit
def golay_encode_kernel(
    triplets_ptr,
    codeword_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    C4: tl.constexpr,
    C5: tl.constexpr,
    C6: tl.constexpr,
    C7: tl.constexpr,
    C8: tl.constexpr,
    C9: tl.constexpr,
    C10: tl.constexpr,
    C11: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    base_offsets = offsets * 3
    n0 = tl.load(triplets_ptr + base_offsets + 0, mask=mask, other=0).to(tl.int32)
    n1 = tl.load(triplets_ptr + base_offsets + 1, mask=mask, other=0).to(tl.int32)
    n2 = tl.load(triplets_ptr + base_offsets + 2, mask=mask, other=0).to(tl.int32)

    data_12bit = (n0 & 0xF) | ((n1 & 0xF) << 4) | ((n2 & 0xF) << 8)

    p0 = _popcount_mod2_12bit(data_12bit & C0)
    p1 = _popcount_mod2_12bit(data_12bit & C1)
    p2 = _popcount_mod2_12bit(data_12bit & C2)
    p3 = _popcount_mod2_12bit(data_12bit & C3)
    p4 = _popcount_mod2_12bit(data_12bit & C4)
    p5 = _popcount_mod2_12bit(data_12bit & C5)
    p6 = _popcount_mod2_12bit(data_12bit & C6)
    p7 = _popcount_mod2_12bit(data_12bit & C7)
    p8 = _popcount_mod2_12bit(data_12bit & C8)
    p9 = _popcount_mod2_12bit(data_12bit & C9)
    p10 = _popcount_mod2_12bit(data_12bit & C10)
    p11 = _popcount_mod2_12bit(data_12bit & C11)

    parity_12bit = (
        (p0 << 0)
        | (p1 << 1)
        | (p2 << 2)
        | (p3 << 3)
        | (p4 << 4)
        | (p5 << 5)
        | (p6 << 6)
        | (p7 << 7)
        | (p8 << 8)
        | (p9 << 9)
        | (p10 << 10)
        | (p11 << 11)
    )

    codeword = data_12bit | (parity_12bit << 12)

    tl.store(codeword_ptr + offsets, codeword.to(tl.int32), mask=mask)


@triton.jit
def _compute_syndrome_golay(codeword, H0, H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11):
    s0 = _popcount_mod2_24bit(codeword & H0)
    s1 = _popcount_mod2_24bit(codeword & H1)
    s2 = _popcount_mod2_24bit(codeword & H2)
    s3 = _popcount_mod2_24bit(codeword & H3)
    s4 = _popcount_mod2_24bit(codeword & H4)
    s5 = _popcount_mod2_24bit(codeword & H5)
    s6 = _popcount_mod2_24bit(codeword & H6)
    s7 = _popcount_mod2_24bit(codeword & H7)
    s8 = _popcount_mod2_24bit(codeword & H8)
    s9 = _popcount_mod2_24bit(codeword & H9)
    s10 = _popcount_mod2_24bit(codeword & H10)
    s11 = _popcount_mod2_24bit(codeword & H11)

    syndrome = (
        (s0 << 0)
        | (s1 << 1)
        | (s2 << 2)
        | (s3 << 3)
        | (s4 << 4)
        | (s5 << 5)
        | (s6 << 6)
        | (s7 << 7)
        | (s8 << 8)
        | (s9 << 9)
        | (s10 << 10)
        | (s11 << 11)
    )
    return syndrome


@triton.jit
def _popcount_mod2_24bit(x):
    x = x ^ (x >> 16)
    x = x ^ (x >> 8)
    x = x ^ (x >> 4)
    x = x ^ (x >> 2)
    x = x ^ (x >> 1)
    return x & 1


@triton.jit
def golay_decode_kernel(
    codeword_ptr,
    decoded_ptr,
    error_count_ptr,
    syndrome_lut_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    H0: tl.constexpr,
    H1: tl.constexpr,
    H2: tl.constexpr,
    H3: tl.constexpr,
    H4: tl.constexpr,
    H5: tl.constexpr,
    H6: tl.constexpr,
    H7: tl.constexpr,
    H8: tl.constexpr,
    H9: tl.constexpr,
    H10: tl.constexpr,
    H11: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    codewords = tl.load(codeword_ptr + offsets, mask=mask, other=0).to(tl.int32)

    syndrome = _compute_syndrome_golay(
        codewords, H0, H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11
    )

    error_pattern = tl.load(syndrome_lut_ptr + syndrome, mask=mask, other=0)

    is_correctable = error_pattern >= 0

    corrected = tl.where(is_correctable, codewords ^ error_pattern, codewords).to(
        tl.int32
    )

    data_12bit = corrected & 0xFFF

    n0 = (data_12bit >> 0) & 0xF
    n1 = (data_12bit >> 4) & 0xF
    n2 = (data_12bit >> 8) & 0xF

    n0 = tl.where(is_correctable, n0, 0).to(tl.uint8)
    n1 = tl.where(is_correctable, n1, 0).to(tl.uint8)
    n2 = tl.where(is_correctable, n2, 0).to(tl.uint8)

    error_bits = tl.where(is_correctable, error_pattern, 0)

    error_bits = (error_bits & 0x555555) + ((error_bits >> 1) & 0x555555)
    error_bits = (error_bits & 0x333333) + ((error_bits >> 2) & 0x333333)
    error_bits = (error_bits & 0x0F0F0F) + ((error_bits >> 4) & 0x0F0F0F)
    error_bits = (error_bits & 0x00FF00FF) + ((error_bits >> 8) & 0x00FF00FF)
    error_bits = (error_bits & 0x0000FFFF) + ((error_bits >> 16) & 0x0000FFFF)
    error_count = tl.where(is_correctable, error_bits, 4).to(tl.uint8)

    base_offsets = offsets * 3
    tl.store(decoded_ptr + base_offsets + 0, n0, mask=mask)
    tl.store(decoded_ptr + base_offsets + 1, n1, mask=mask)
    tl.store(decoded_ptr + base_offsets + 2, n2, mask=mask)

    tl.store(error_count_ptr + offsets, error_count, mask=mask)


_syndrome_table_cache = {}


UNCORRECTABLE_MARKER = -1


def _build_syndrome_table(device="cuda"):
    if device in _syndrome_table_cache:
        return _syndrome_table_cache[device]

    cpu_codec = Golay2412(device="cuda")

    table = cpu_codec.syndrome_table.clone()

    uncorrectable_mask = table == cpu_codec.UNCORRECTABLE
    table[uncorrectable_mask] = UNCORRECTABLE_MARKER

    table = table.to(torch.int32).to(device)
    _syndrome_table_cache[device] = table
    return table


def _build_h_row_masks():
    B = [
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
        [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    ]

    h_masks = []
    for i in range(12):
        mask = 0

        for j in range(12):
            if B[j][i] == 1:
                mask |= 1 << j

        mask |= 1 << (12 + i)
        h_masks.append(mask)

    return tuple(h_masks)


_H_ROW_MASKS = _build_h_row_masks()


def golay_encode(triplets):
    assert triplets.is_cuda, "Input must be on CUDA device"

    if triplets.dim() == 1:
        triplets = triplets.unsqueeze(0)

    N = triplets.shape[0]
    flat_triplets = triplets.contiguous().flatten().to(torch.uint8)

    codewords = torch.empty(N, dtype=torch.int32, device=triplets.device)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    golay_encode_kernel[grid](
        flat_triplets,
        codewords,
        N,
        BLOCK_SIZE=GOLAY_BLOCK_SIZE,
        C0=B_COL_0,
        C1=B_COL_1,
        C2=B_COL_2,
        C3=B_COL_3,
        C4=B_COL_4,
        C5=B_COL_5,
        C6=B_COL_6,
        C7=B_COL_7,
        C8=B_COL_8,
        C9=B_COL_9,
        C10=B_COL_10,
        C11=B_COL_11,
    )

    return codewords


def golay_decode(codewords, return_error_counts=False):
    assert codewords.is_cuda, "Input must be on CUDA device"

    N = codewords.numel()
    device = codewords.device

    flat_codewords = codewords.flatten().to(torch.int32)

    decoded_flat = torch.empty(N * 3, dtype=torch.uint8, device=device)
    error_counts = torch.empty(N, dtype=torch.uint8, device=device)

    syndrome_table = _build_syndrome_table(str(device))

    H = _H_ROW_MASKS

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    golay_decode_kernel[grid](
        flat_codewords,
        decoded_flat,
        error_counts,
        syndrome_table,
        N,
        BLOCK_SIZE=GOLAY_BLOCK_SIZE,
        H0=H[0],
        H1=H[1],
        H2=H[2],
        H3=H[3],
        H4=H[4],
        H5=H[5],
        H6=H[6],
        H7=H[7],
        H8=H[8],
        H9=H[9],
        H10=H[10],
        H11=H[11],
    )

    decoded = decoded_flat.view(N, 3)

    correctable_mask = error_counts < 4
    total_errors_corrected = int(error_counts[correctable_mask].sum())
    uncorrectable_count = int((~correctable_mask).sum())

    if return_error_counts:
        return decoded, error_counts, (total_errors_corrected, uncorrectable_count)
    else:
        return decoded, (total_errors_corrected, uncorrectable_count)


def verify_triton_vs_cpu():
    device = "cuda"
    cpu_codec = Golay2412(device="cuda")

    print("Verifying Triton Golay(24,12) vs CPU reference...")

    test_triplets = torch.tensor(
        [
            [5, 10, 3],
            [0, 0, 0],
            [15, 15, 15],
            [7, 8, 9],
        ],
        dtype=torch.uint8,
    )

    cpu_codewords = cpu_codec.encode(test_triplets)
    triton_codewords = golay_encode(test_triplets.to(device))

    assert torch.equal(
        cpu_codewords.to(torch.int32), triton_codewords.cpu()
    ), "Triton encode does not match CPU!"
    print("  [PASS] Encode matches CPU reference")

    decoded, stats = golay_decode(triton_codewords)
    assert torch.equal(test_triplets.to(device), decoded), "Clean decode failed!"
    assert stats[0] == 0 and stats[1] == 0, "Should have no errors"
    print("  [PASS] Clean decode works")

    test_triplet = torch.tensor([[5, 10, 3]], dtype=torch.uint8, device=device)
    codeword = golay_encode(test_triplet)

    for bit_pos in range(24):
        corrupted = codeword ^ (1 << bit_pos)
        decoded, stats = golay_decode(corrupted)
        assert torch.equal(
            decoded, test_triplet
        ), f"Single-bit error at bit {bit_pos} not corrected"

    print("  [PASS] Single-bit error correction works (all 24 positions)")

    for bit1 in range(0, 24, 4):
        for bit2 in range(bit1 + 1, 24, 4):
            corrupted = codeword ^ (1 << bit1) ^ (1 << bit2)
            decoded, stats = golay_decode(corrupted)
            assert torch.equal(
                decoded, test_triplet
            ), f"Double-bit error at bits {bit1},{bit2} not corrected"

    print("  [PASS] Double-bit error correction works")

    corrupted = codeword ^ (1 << 0) ^ (1 << 7) ^ (1 << 15)
    decoded, stats = golay_decode(corrupted)
    assert torch.equal(decoded, test_triplet), "Triple-bit error not corrected"
    print("  [PASS] Triple-bit error correction works")

    large_triplets = torch.randint(0, 16, (10000, 3), dtype=torch.uint8, device=device)
    encoded = golay_encode(large_triplets)
    decoded, stats = golay_decode(encoded)
    assert torch.equal(large_triplets, decoded), "Large tensor roundtrip failed!"
    print("  [PASS] Large tensor roundtrip works (10K triplets)")

    print("All Triton Golay(24,12) verifications passed!")
    return True


if __name__ == "__main__":
    verify_triton_vs_cpu()
