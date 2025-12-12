"""
Triton GPU Kernels for Golay(24,12) Code.

GPU-native implementation of Extended Binary Golay(24,12) encoder/decoder.
Bundles 3 INT4 values (12 bits) into a 24-bit codeword, corrects up to 3 errors.

Key optimizations:
- Encode: Hardcoded XOR parity computation (no LUT needed)
- Decode: 4096-entry syndrome table in shared memory (~16KB)
- Storage: torch.int32 (24 bits + 8 wasted for alignment)
"""

import torch
import triton
import triton.language as tl
from typing import Tuple

from .config import GOLAY_BLOCK_SIZE, GOLAY_PARITY_MASKS


# =============================================================================
# Golay B Matrix Columns (for encoding parity bits)
# parity[j] = popcount(data & B_COL[j]) % 2
# where B_COL[j] is the j-th column of B as a 12-bit mask
# =============================================================================

# B matrix from golay.py (each row is a data bit's contribution to parity):
# B[i][j] = 1 means data bit i contributes to parity bit j
# For parity computation: parity[j] = XOR of data bits where B[:,j] = 1
# So we need column masks, not row masks.
#
# Original B matrix rows:
# [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],  # d0's contribution
# [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],  # d1's contribution
# [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],  # d2's contribution
# [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],  # d3's contribution
# [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # d4's contribution
# [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],  # d5's contribution
# [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],  # d6's contribution
# [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],  # d7's contribution
# [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],  # d8's contribution
# [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],  # d9's contribution
# [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],  # d10's contribution
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # d11's contribution
#
# Column j tells which data bits contribute to parity bit j
# B_COL[j] bit i = 1 if B[i][j] = 1

B_COL_0  = 0b101000111011  # 2619 = d0, d1, d3, d4, d5, d9, d11
B_COL_1  = 0b110100011101  # 3357 = d0, d2, d3, d4, d8, d10, d11
B_COL_2  = 0b111010001110  # 3726 = d1, d2, d3, d7, d9, d10, d11
B_COL_3  = 0b101101000111  # 2887 = d0, d1, d2, d6, d8, d9, d11
B_COL_4  = 0b110110100011  # 3491 = d0, d1, d5, d7, d8, d10, d11
B_COL_5  = 0b111011010001  # 3793 = d0, d4, d6, d7, d9, d10, d11
B_COL_6  = 0b111101101000  # 3944 = d3, d5, d6, d8, d9, d10, d11
B_COL_7  = 0b101110110100  # 2996 = d2, d4, d5, d7, d8, d9, d11
B_COL_8  = 0b100111011010  # 2522 = d1, d3, d4, d6, d7, d8, d11
B_COL_9  = 0b100011101101  # 2285 = d0, d2, d3, d5, d6, d7, d11
B_COL_10 = 0b110001110110  # 3190 = d1, d2, d4, d5, d6, d10, d11
B_COL_11 = 0b011111111111  # 2047 = d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10


# =============================================================================
# Triton Encode Kernel
# =============================================================================

@triton.jit
def _popcount_mod2_12bit(x):
    """Compute popcount(x) mod 2 for 12-bit value via bit-folding."""
    # Fold 12 bits down to 1 bit
    x = x ^ (x >> 8)   # Fold upper 4 bits
    x = x ^ (x >> 4)   # Fold to 4 bits
    x = x ^ (x >> 2)   # Fold to 2 bits
    x = x ^ (x >> 1)   # Fold to 1 bit
    return x & 1


@triton.jit
def golay_encode_kernel(
    # Pointers
    triplets_ptr,     # Input: INT4 triplets [N, 3], dtype=uint8
    codeword_ptr,     # Output: 24-bit codewords [N], dtype=int32
    # Dimensions
    N,                # Total number of triplets
    # Block size
    BLOCK_SIZE: tl.constexpr,
    # B matrix columns for parity computation
    # parity[j] = popcount(data & B_COL[j]) mod 2
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    C4: tl.constexpr, C5: tl.constexpr, C6: tl.constexpr, C7: tl.constexpr,
    C8: tl.constexpr, C9: tl.constexpr, C10: tl.constexpr, C11: tl.constexpr,
):
    """
    Encode INT4 triplets to Golay(24,12) codewords on GPU.

    Systematic form: codeword = [data_12bits | parity_12bits]
    Parity[j] = popcount(data & B_COL[j]) mod 2
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load triplet values (3 consecutive uint8 per triplet)
    # Memory layout: [n0, n1, n2, n0, n1, n2, ...]
    base_offsets = offsets * 3
    n0 = tl.load(triplets_ptr + base_offsets + 0, mask=mask, other=0).to(tl.int32)
    n1 = tl.load(triplets_ptr + base_offsets + 1, mask=mask, other=0).to(tl.int32)
    n2 = tl.load(triplets_ptr + base_offsets + 2, mask=mask, other=0).to(tl.int32)

    # Pack into 12-bit data word: val = n0 | (n1 << 4) | (n2 << 8)
    data_12bit = (n0 & 0xF) | ((n1 & 0xF) << 4) | ((n2 & 0xF) << 8)

    # Compute 12 parity bits using B matrix columns
    # Each parity bit j is popcount(data & B_COL[j]) mod 2
    p0  = _popcount_mod2_12bit(data_12bit & C0)
    p1  = _popcount_mod2_12bit(data_12bit & C1)
    p2  = _popcount_mod2_12bit(data_12bit & C2)
    p3  = _popcount_mod2_12bit(data_12bit & C3)
    p4  = _popcount_mod2_12bit(data_12bit & C4)
    p5  = _popcount_mod2_12bit(data_12bit & C5)
    p6  = _popcount_mod2_12bit(data_12bit & C6)
    p7  = _popcount_mod2_12bit(data_12bit & C7)
    p8  = _popcount_mod2_12bit(data_12bit & C8)
    p9  = _popcount_mod2_12bit(data_12bit & C9)
    p10 = _popcount_mod2_12bit(data_12bit & C10)
    p11 = _popcount_mod2_12bit(data_12bit & C11)

    # Pack parity into bits 12-23
    parity_12bit = (
        (p0 << 0) | (p1 << 1) | (p2 << 2) | (p3 << 3) |
        (p4 << 4) | (p5 << 5) | (p6 << 6) | (p7 << 7) |
        (p8 << 8) | (p9 << 9) | (p10 << 10) | (p11 << 11)
    )

    # Final 24-bit codeword: [data_12bit | parity_12bit << 12]
    codeword = data_12bit | (parity_12bit << 12)

    # Store as int32
    tl.store(codeword_ptr + offsets, codeword.to(tl.int32), mask=mask)


# =============================================================================
# Triton Decode Kernel with Shared Memory Syndrome LUT
# =============================================================================

@triton.jit
def _compute_syndrome_golay(codeword, H0, H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11):
    """
    Compute 12-bit syndrome for a 24-bit Golay codeword.

    syndrome[i] = popcount(codeword & H_row[i]) mod 2
    where H = [B^T | I_12] (parity check matrix)
    """
    # Each H_row mask is 24 bits representing which codeword bits to XOR
    s0  = _popcount_mod2_24bit(codeword & H0)
    s1  = _popcount_mod2_24bit(codeword & H1)
    s2  = _popcount_mod2_24bit(codeword & H2)
    s3  = _popcount_mod2_24bit(codeword & H3)
    s4  = _popcount_mod2_24bit(codeword & H4)
    s5  = _popcount_mod2_24bit(codeword & H5)
    s6  = _popcount_mod2_24bit(codeword & H6)
    s7  = _popcount_mod2_24bit(codeword & H7)
    s8  = _popcount_mod2_24bit(codeword & H8)
    s9  = _popcount_mod2_24bit(codeword & H9)
    s10 = _popcount_mod2_24bit(codeword & H10)
    s11 = _popcount_mod2_24bit(codeword & H11)

    syndrome = (
        (s0 << 0) | (s1 << 1) | (s2 << 2) | (s3 << 3) |
        (s4 << 4) | (s5 << 5) | (s6 << 6) | (s7 << 7) |
        (s8 << 8) | (s9 << 9) | (s10 << 10) | (s11 << 11)
    )
    return syndrome


@triton.jit
def _popcount_mod2_24bit(x):
    """Compute popcount(x) mod 2 for 24-bit value via bit-folding."""
    x = x ^ (x >> 16)  # Fold upper 8 bits
    x = x ^ (x >> 8)   # Fold to 8 bits
    x = x ^ (x >> 4)   # Fold to 4 bits
    x = x ^ (x >> 2)   # Fold to 2 bits
    x = x ^ (x >> 1)   # Fold to 1 bit
    return x & 1


@triton.jit
def golay_decode_kernel(
    # Pointers
    codeword_ptr,      # Input: 24-bit codewords [N], dtype=int32
    decoded_ptr,       # Output: corrected triplets [N, 3], dtype=uint8
    error_count_ptr,   # Output: error counts [N], dtype=uint8 (0-4, 4=uncorrectable)
    syndrome_lut_ptr,  # Shared memory: syndrome -> error pattern [4096], dtype=int32
    # Dimensions
    N,
    # Block size
    BLOCK_SIZE: tl.constexpr,
    # H matrix row masks (24-bit each)
    H0: tl.constexpr, H1: tl.constexpr, H2: tl.constexpr, H3: tl.constexpr,
    H4: tl.constexpr, H5: tl.constexpr, H6: tl.constexpr, H7: tl.constexpr,
    H8: tl.constexpr, H9: tl.constexpr, H10: tl.constexpr, H11: tl.constexpr,
):
    """
    Decode Golay(24,12) codewords with error correction.

    Uses shared memory syndrome LUT for O(1) error pattern lookup.
    Corrects up to 3 bit errors per codeword.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load codewords
    codewords = tl.load(codeword_ptr + offsets, mask=mask, other=0).to(tl.int32)

    # Compute 12-bit syndrome
    syndrome = _compute_syndrome_golay(
        codewords, H0, H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11
    )

    # Lookup error pattern from syndrome table (shared memory)
    # error_pattern is 24-bit, stored as int32
    # -1 indicates uncorrectable (>3 errors)
    error_pattern = tl.load(syndrome_lut_ptr + syndrome, mask=mask, other=0)

    # Check if correctable (-1 means uncorrectable)
    is_correctable = (error_pattern >= 0)

    # Apply correction via XOR (only if correctable)
    corrected = tl.where(is_correctable, codewords ^ error_pattern, codewords).to(tl.int32)

    # Extract data bits (first 12 bits in systematic form)
    data_12bit = corrected & 0xFFF

    # Unpack to 3 INT4 values
    n0 = (data_12bit >> 0) & 0xF
    n1 = (data_12bit >> 4) & 0xF
    n2 = (data_12bit >> 8) & 0xF

    # For uncorrectable errors, zero out the data
    n0 = tl.where(is_correctable, n0, 0).to(tl.uint8)
    n1 = tl.where(is_correctable, n1, 0).to(tl.uint8)
    n2 = tl.where(is_correctable, n2, 0).to(tl.uint8)

    # Count errors (popcount of error_pattern, or 4 if uncorrectable)
    # Simplified: store 4 for uncorrectable, otherwise compute popcount
    error_bits = tl.where(is_correctable, error_pattern, 0)
    # Popcount via parallel bit count
    error_bits = (error_bits & 0x555555) + ((error_bits >> 1) & 0x555555)
    error_bits = (error_bits & 0x333333) + ((error_bits >> 2) & 0x333333)
    error_bits = (error_bits & 0x0F0F0F) + ((error_bits >> 4) & 0x0F0F0F)
    error_bits = (error_bits & 0x00FF00FF) + ((error_bits >> 8) & 0x00FF00FF)
    error_bits = (error_bits & 0x0000FFFF) + ((error_bits >> 16) & 0x0000FFFF)
    error_count = tl.where(is_correctable, error_bits, 4).to(tl.uint8)

    # Store decoded triplets (interleaved)
    base_offsets = offsets * 3
    tl.store(decoded_ptr + base_offsets + 0, n0, mask=mask)
    tl.store(decoded_ptr + base_offsets + 1, n1, mask=mask)
    tl.store(decoded_ptr + base_offsets + 2, n2, mask=mask)

    # Store error counts
    tl.store(error_count_ptr + offsets, error_count, mask=mask)


# =============================================================================
# Python Wrapper Functions
# =============================================================================

# Cache for syndrome table (lazy initialization)
_syndrome_table_cache = {}


# Marker for uncorrectable errors (fits in int32)
UNCORRECTABLE_MARKER = -1


def _build_syndrome_table(device: str = "cuda") -> torch.Tensor:
    """Build the 4096-entry syndrome lookup table on the specified device."""
    if device in _syndrome_table_cache:
        return _syndrome_table_cache[device]

    # Import CPU Golay codec to build table
    from ..golay import Golay2412
    cpu_codec = Golay2412(device="cpu")

    # Copy syndrome table to GPU as int32
    # Convert int64 to int32 (24-bit patterns fit)
    table = cpu_codec.syndrome_table.clone()

    # Mark uncorrectable with -1 (fits in int32, all errors are < 2^24)
    uncorrectable_mask = table == cpu_codec.UNCORRECTABLE
    table[uncorrectable_mask] = UNCORRECTABLE_MARKER

    table = table.to(torch.int32).to(device)
    _syndrome_table_cache[device] = table
    return table


def _build_h_row_masks() -> Tuple[int, ...]:
    """
    Build H matrix row masks for syndrome computation.

    H = [B^T | I_12] where B is the Golay parity matrix.
    Each row mask is 24 bits indicating which codeword bits to XOR.
    """
    # B^T rows (transposed B matrix from golay.py)
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

    # H = [B^T | I_12]
    # For row i: first 12 bits from B column i, bits 12-23 from I_12 row i
    h_masks = []
    for i in range(12):
        mask = 0
        # B^T portion (bits 0-11): column i of B
        for j in range(12):
            if B[j][i] == 1:
                mask |= (1 << j)
        # I_12 portion (bits 12-23): only bit (12+i) is 1
        mask |= (1 << (12 + i))
        h_masks.append(mask)

    return tuple(h_masks)


# Pre-compute H row masks at module load
_H_ROW_MASKS = _build_h_row_masks()


def golay_encode(triplets: torch.Tensor) -> torch.Tensor:
    """
    Encode INT4 triplets to Golay(24,12) codewords using Triton kernel.

    Args:
        triplets: Tensor of shape (N, 3) with INT4 values (0-15), on CUDA

    Returns:
        codewords: Tensor of shape (N,) with 24-bit codewords (int32)
    """
    assert triplets.is_cuda, "Input must be on CUDA device"

    if triplets.dim() == 1:
        triplets = triplets.unsqueeze(0)

    N = triplets.shape[0]
    flat_triplets = triplets.contiguous().flatten().to(torch.uint8)

    # Allocate output
    codewords = torch.empty(N, dtype=torch.int32, device=triplets.device)

    # Launch kernel
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    golay_encode_kernel[grid](
        flat_triplets, codewords, N,
        BLOCK_SIZE=GOLAY_BLOCK_SIZE,
        C0=B_COL_0, C1=B_COL_1, C2=B_COL_2, C3=B_COL_3,
        C4=B_COL_4, C5=B_COL_5, C6=B_COL_6, C7=B_COL_7,
        C8=B_COL_8, C9=B_COL_9, C10=B_COL_10, C11=B_COL_11,
    )

    return codewords


def golay_decode(
    codewords: torch.Tensor,
    return_error_counts: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """
    Decode Golay(24,12) codewords with error correction using Triton kernel.

    Args:
        codewords: Tensor of shape (N,) with 24-bit codewords (int32), on CUDA
        return_error_counts: If True, return per-element error counts

    Returns:
        decoded: Tensor of shape (N, 3) with recovered INT4 triplets
        error_counts: (optional) Per-element error counts (0-3, or 4 for uncorrectable)
        stats: Tuple of (total_errors_corrected, uncorrectable_count)
    """
    assert codewords.is_cuda, "Input must be on CUDA device"

    N = codewords.numel()
    device = codewords.device

    flat_codewords = codewords.flatten().to(torch.int32)

    # Allocate outputs
    decoded_flat = torch.empty(N * 3, dtype=torch.uint8, device=device)
    error_counts = torch.empty(N, dtype=torch.uint8, device=device)

    # Get syndrome table (cached)
    syndrome_table = _build_syndrome_table(str(device))

    # Get H row masks
    H = _H_ROW_MASKS

    # Launch kernel
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    golay_decode_kernel[grid](
        flat_codewords, decoded_flat, error_counts, syndrome_table,
        N,
        BLOCK_SIZE=GOLAY_BLOCK_SIZE,
        H0=H[0], H1=H[1], H2=H[2], H3=H[3],
        H4=H[4], H5=H[5], H6=H[6], H7=H[7],
        H8=H[8], H9=H[9], H10=H[10], H11=H[11],
    )

    # Reshape to (N, 3)
    decoded = decoded_flat.view(N, 3)

    # Compute statistics
    correctable_mask = error_counts < 4
    total_errors_corrected = int(error_counts[correctable_mask].sum())
    uncorrectable_count = int((~correctable_mask).sum())

    if return_error_counts:
        return decoded, error_counts, (total_errors_corrected, uncorrectable_count)
    else:
        return decoded, (total_errors_corrected, uncorrectable_count)


# =============================================================================
# Verification
# =============================================================================

def verify_triton_vs_cpu():
    """
    Verify Triton Golay kernels match CPU reference implementation.
    """
    from ..golay import Golay2412

    device = "cuda"
    cpu_codec = Golay2412(device="cpu")

    print("Verifying Triton Golay(24,12) vs CPU reference...")

    # Test 1: Encode sample triplets
    test_triplets = torch.tensor([
        [5, 10, 3],
        [0, 0, 0],
        [15, 15, 15],
        [7, 8, 9],
    ], dtype=torch.uint8)

    cpu_codewords = cpu_codec.encode(test_triplets)
    triton_codewords = golay_encode(test_triplets.to(device))

    # CPU returns int64, Triton returns int32 (both hold 24-bit values)
    assert torch.equal(cpu_codewords.to(torch.int32), triton_codewords.cpu()), \
        "Triton encode does not match CPU!"
    print("  [PASS] Encode matches CPU reference")

    # Test 2: Decode clean codewords
    decoded, stats = golay_decode(triton_codewords)
    assert torch.equal(test_triplets.to(device), decoded), \
        "Clean decode failed!"
    assert stats[0] == 0 and stats[1] == 0, "Should have no errors"
    print("  [PASS] Clean decode works")

    # Test 3: Single-bit error correction (all 24 positions)
    test_triplet = torch.tensor([[5, 10, 3]], dtype=torch.uint8, device=device)
    codeword = golay_encode(test_triplet)

    for bit_pos in range(24):
        corrupted = codeword ^ (1 << bit_pos)
        decoded, stats = golay_decode(corrupted)
        assert torch.equal(decoded, test_triplet), \
            f"Single-bit error at bit {bit_pos} not corrected"

    print("  [PASS] Single-bit error correction works (all 24 positions)")

    # Test 4: Double-bit error correction
    for bit1 in range(0, 24, 4):
        for bit2 in range(bit1 + 1, 24, 4):
            corrupted = codeword ^ (1 << bit1) ^ (1 << bit2)
            decoded, stats = golay_decode(corrupted)
            assert torch.equal(decoded, test_triplet), \
                f"Double-bit error at bits {bit1},{bit2} not corrected"

    print("  [PASS] Double-bit error correction works")

    # Test 5: Triple-bit error correction (sample)
    corrupted = codeword ^ (1 << 0) ^ (1 << 7) ^ (1 << 15)
    decoded, stats = golay_decode(corrupted)
    assert torch.equal(decoded, test_triplet), "Triple-bit error not corrected"
    print("  [PASS] Triple-bit error correction works")

    # Test 6: Large tensor roundtrip
    large_triplets = torch.randint(0, 16, (10000, 3), dtype=torch.uint8, device=device)
    encoded = golay_encode(large_triplets)
    decoded, stats = golay_decode(encoded)
    assert torch.equal(large_triplets, decoded), "Large tensor roundtrip failed!"
    print("  [PASS] Large tensor roundtrip works (10K triplets)")

    print("All Triton Golay(24,12) verifications passed!")
    return True


if __name__ == "__main__":
    if torch.cuda.is_available():
        verify_triton_vs_cpu()
    else:
        print("CUDA not available, skipping Triton verification")
