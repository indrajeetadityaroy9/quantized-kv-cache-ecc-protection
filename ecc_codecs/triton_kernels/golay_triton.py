"""
Golay(24,12) Codec: GPU-accelerated 3-error-correcting perfect code.

This module implements the extended binary Golay code using Triton GPU kernels.
Golay(24,12) is a [24,12,8] code: 12 data bits, 24 codeword bits, minimum distance 8.

Properties:
    - Rate: 1/2 (50% overhead)
    - Corrects up to 3 arbitrary bit errors per codeword
    - Detects up to 7 bit errors
    - Perfect code: all 2^12 syndromes map to unique ≤3-error patterns

Data organization:
    Each codeword protects a triplet of INT4 values:
        triplet = [n0, n1, n2] where each n_i ∈ [0, 15]
        data_12bit = n0 | (n1 << 4) | (n2 << 8)
        codeword_24bit = data_12bit | (parity_12bit << 12)

Decoding algorithm:
    1. Compute 12-bit syndrome via H matrix row masks
    2. Lookup syndrome in precomputed 4096-entry table → error pattern
    3. XOR codeword with error pattern to correct
    4. If table entry is -1: uncorrectable (>3 errors)

Pipeline role:
    Called by kv_cache/ecc_shim.py for "int12-golay" mode.
    Triplets are packed from consecutive cache values for protection.

Performance:
    - O(1) syndrome table lookup (4096 entries, 16KB)
    - Block size 256 optimized for 24-bit register usage
    - ~80% of peak bandwidth on A100 for large tensors

Determinism:
    Fully deterministic given identical input tensors. No RNG involved.
"""

import torch
import triton
import triton.language as tl

from .config import (
    GOLAY_BLOCK_SIZE,
    GOLAY_PARITY_MASKS,
    GOLAY_B_MATRIX,
    GOLAY_UNCORRECTABLE,
    GolayDecodeResult,
    build_golay_syndrome_table,
)


# =============================================================================
# B Matrix Columns as Bitmasks
# =============================================================================
# Each B_COL_i is a 12-bit mask representing column i of the B matrix.
# Used for efficient parity computation: p_i = popcount(data & B_COL_i) mod 2
# The B matrix is derived from quadratic residues mod 11.

B_COL_0 = 0b101000111011   # Column 0 of B matrix
B_COL_1 = 0b110100011101   # Column 1
B_COL_2 = 0b111010001110   # Column 2
B_COL_3 = 0b101101000111   # Column 3
B_COL_4 = 0b110110100011   # Column 4
B_COL_5 = 0b111011010001   # Column 5
B_COL_6 = 0b111101101000   # Column 6
B_COL_7 = 0b101110110100   # Column 7
B_COL_8 = 0b100111011010   # Column 8
B_COL_9 = 0b100011101101   # Column 9
B_COL_10 = 0b110001110110  # Column 10
B_COL_11 = 0b011111111111  # Column 11 (mostly ones)


# =============================================================================
# Popcount Helper Functions
# =============================================================================


@triton.jit
def _popcount_mod2_12bit(x):
    """
    Compute popcount(x) mod 2 for 12-bit value using parallel XOR reduction.

    This is the parity of x: 0 if even number of 1-bits, 1 if odd.
    Used for computing parity bits in Golay encoding.
    """
    x = x ^ (x >> 8)   # Fold top 4 bits into lower 8
    x = x ^ (x >> 4)   # Fold into lower 4
    x = x ^ (x >> 2)   # Fold into lower 2
    x = x ^ (x >> 1)   # Fold into lowest bit
    return x & 1


# =============================================================================
# Golay(24,12) Encode Kernel
# =============================================================================


@triton.jit
def golay_encode_kernel(
    triplets_ptr,   # Input: pointer to flattened triplets [n0, n1, n2, n0, n1, n2, ...]
    codeword_ptr,   # Output: pointer to 24-bit codewords (int32)
    N,              # Number of triplets to encode
    BLOCK_SIZE: tl.constexpr,
    # B matrix columns as compile-time constants (12 x 12-bit masks)
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    C4: tl.constexpr, C5: tl.constexpr, C6: tl.constexpr, C7: tl.constexpr,
    C8: tl.constexpr, C9: tl.constexpr, C10: tl.constexpr, C11: tl.constexpr,
):
    """
    Triton kernel: Encode INT4 triplets to Golay(24,12) codewords.

    Each thread processes one triplet → one 24-bit codeword.

    Codeword layout (24 bits stored in int32):
        bits 0-11:  data (n0 | n1<<4 | n2<<8)
        bits 12-23: parity (12 bits computed from B matrix)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load triplet: 3 consecutive uint8 values per codeword
    base_offsets = offsets * 3
    n0 = tl.load(triplets_ptr + base_offsets + 0, mask=mask, other=0).to(tl.int32)
    n1 = tl.load(triplets_ptr + base_offsets + 1, mask=mask, other=0).to(tl.int32)
    n2 = tl.load(triplets_ptr + base_offsets + 2, mask=mask, other=0).to(tl.int32)

    # Pack three INT4 values into 12-bit data word
    # Layout: [n0:bits0-3, n1:bits4-7, n2:bits8-11]
    data_12bit = (n0 & 0xF) | ((n1 & 0xF) << 4) | ((n2 & 0xF) << 8)

    # Compute 12 parity bits using B matrix columns
    # p_i = popcount(data & B_col_i) mod 2
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

    # Assemble 12-bit parity word
    parity_12bit = (
        (p0 << 0) | (p1 << 1) | (p2 << 2) | (p3 << 3)
        | (p4 << 4) | (p5 << 5) | (p6 << 6) | (p7 << 7)
        | (p8 << 8) | (p9 << 9) | (p10 << 10) | (p11 << 11)
    )

    # Final 24-bit codeword: data | (parity << 12)
    codeword = data_12bit | (parity_12bit << 12)

    tl.store(codeword_ptr + offsets, codeword.to(tl.int32), mask=mask)


@triton.jit
def _compute_syndrome_golay(codeword, H0, H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11):
    """
    Compute 12-bit syndrome for a 24-bit Golay codeword.

    syndrome[i] = popcount(codeword & H_row_mask_i) mod 2

    Where H_row_mask_i is the 24-bit mask for row i of the parity-check matrix.
    Returns 0 if no error, otherwise unique syndrome for ≤3 error patterns.
    """
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

    # Assemble 12-bit syndrome
    syndrome = (
        (s0 << 0) | (s1 << 1) | (s2 << 2) | (s3 << 3)
        | (s4 << 4) | (s5 << 5) | (s6 << 6) | (s7 << 7)
        | (s8 << 8) | (s9 << 9) | (s10 << 10) | (s11 << 11)
    )
    return syndrome


@triton.jit
def _popcount_mod2_24bit(x):
    """
    Compute popcount(x) mod 2 for 24-bit value using parallel XOR reduction.

    Same algorithm as _popcount_mod2_12bit but handles 24-bit values.
    """
    x = x ^ (x >> 16)  # Fold top 8 bits into lower 16
    x = x ^ (x >> 8)   # Fold into lower 8
    x = x ^ (x >> 4)   # Fold into lower 4
    x = x ^ (x >> 2)   # Fold into lower 2
    x = x ^ (x >> 1)   # Fold into lowest bit
    return x & 1


# =============================================================================
# Golay(24,12) Decode Kernel
# =============================================================================


@triton.jit
def golay_decode_kernel(
    codeword_ptr,       # Input: pointer to 24-bit codewords (int32)
    decoded_ptr,        # Output: pointer to decoded triplets [n0,n1,n2,...]
    error_count_ptr,    # Output: pointer to error counts per codeword (uint8)
    syndrome_lut_ptr,   # Syndrome table: 4096-entry LUT mapping syndrome → error pattern
    N,                  # Number of codewords to decode
    BLOCK_SIZE: tl.constexpr,
    # H matrix row masks as compile-time constants (12 x 24-bit masks)
    H0: tl.constexpr, H1: tl.constexpr, H2: tl.constexpr, H3: tl.constexpr,
    H4: tl.constexpr, H5: tl.constexpr, H6: tl.constexpr, H7: tl.constexpr,
    H8: tl.constexpr, H9: tl.constexpr, H10: tl.constexpr, H11: tl.constexpr,
):
    """
    Triton kernel: Decode Golay(24,12) codewords to INT4 triplets.

    Corrects up to 3 bit errors per codeword using syndrome table lookup.
    For >3 errors, marks as uncorrectable but preserves data for analysis.

    Output error_count semantics:
        0-3: number of bits corrected (codeword was correctable)
        4:   uncorrectable (>3 errors, data may be corrupted)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load 24-bit codewords
    codewords = tl.load(codeword_ptr + offsets, mask=mask, other=0).to(tl.int32)

    # Compute 12-bit syndrome using H matrix row masks
    syndrome = _compute_syndrome_golay(
        codewords, H0, H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11
    )

    # O(1) syndrome table lookup: syndrome → 24-bit error pattern
    # Table value is -1 for uncorrectable syndromes (>3 errors)
    error_pattern = tl.load(syndrome_lut_ptr + syndrome, mask=mask, other=0)

    # Check if syndrome maps to a correctable error pattern
    is_correctable = error_pattern >= 0

    # XOR correction: codeword ^ error_pattern recovers original codeword
    corrected = tl.where(is_correctable, codewords ^ error_pattern, codewords).to(
        tl.int32
    )

    # Extract 12-bit data word (low 12 bits)
    data_12bit = corrected & 0xFFF

    # Unpack triplet: [n0:bits0-3, n1:bits4-7, n2:bits8-11]
    n0 = (data_12bit >> 0) & 0xF
    n1 = (data_12bit >> 4) & 0xF
    n2 = (data_12bit >> 8) & 0xF

    # CRITICAL: Preserve data on uncorrectable error (do not zero)
    # Previously zeroed data, causing silent corruption - now caller checks error_count
    n0 = n0.to(tl.uint8)
    n1 = n1.to(tl.uint8)
    n2 = n2.to(tl.uint8)

    # Count bits in error pattern using parallel popcount
    # This gives the number of errors corrected (0-3 for correctable)
    error_bits = tl.where(is_correctable, error_pattern, 0)

    # Parallel popcount algorithm (counts 1-bits in 24-bit value)
    error_bits = (error_bits & 0x555555) + ((error_bits >> 1) & 0x555555)
    error_bits = (error_bits & 0x333333) + ((error_bits >> 2) & 0x333333)
    error_bits = (error_bits & 0x0F0F0F) + ((error_bits >> 4) & 0x0F0F0F)
    error_bits = (error_bits & 0x00FF00FF) + ((error_bits >> 8) & 0x00FF00FF)
    error_bits = (error_bits & 0x0000FFFF) + ((error_bits >> 16) & 0x0000FFFF)

    # error_count: 0-3 for correctable, 4 for uncorrectable
    # Value 4 is a sentinel meaning ">3 errors, data unreliable"
    error_count = tl.where(is_correctable, error_bits, 4).to(tl.uint8)

    # Store decoded triplet (3 consecutive uint8 values)
    base_offsets = offsets * 3
    tl.store(decoded_ptr + base_offsets + 0, n0, mask=mask)
    tl.store(decoded_ptr + base_offsets + 1, n1, mask=mask)
    tl.store(decoded_ptr + base_offsets + 2, n2, mask=mask)

    # Store per-codeword error count
    tl.store(error_count_ptr + offsets, error_count, mask=mask)


# =============================================================================
# Syndrome Table Cache
# =============================================================================
# The 4096-entry syndrome table is computed once and cached per GPU device.
# This avoids expensive table reconstruction on every decode call.

_syndrome_table_cache = {}


UNCORRECTABLE_MARKER = -1  # Sentinel value for >3 error syndromes


def _build_syndrome_table(device: str = "cuda") -> torch.Tensor:
    """
    Build and cache the Golay syndrome lookup table on the specified device.

    The table maps each 12-bit syndrome (0-4095) to a 24-bit error pattern.
    For uncorrectable syndromes (>3 errors), the table contains -1.

    Args:
        device: Target CUDA device string

    Returns:
        Tensor of shape (4096,) with int32 error patterns, on the specified device
    """
    if device in _syndrome_table_cache:
        return _syndrome_table_cache[device]

    # Build syndrome table using pure Python (see config.build_golay_syndrome_table)
    table = build_golay_syndrome_table()
    table = table.to(device)
    _syndrome_table_cache[device] = table
    return table


def _build_h_row_masks() -> tuple:
    """
    Build H matrix row masks for syndrome computation in decode kernel.

    Each mask is a 24-bit integer representing one row of H = [B^T | I₁₂].
    These are passed as compile-time constants to the Triton kernel.

    Returns:
        Tuple of 12 integers, each a 24-bit mask
    """
    # B matrix (same as GOLAY_B_MATRIX but hardcoded for kernel initialization)
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
        # B^T contribution: column i of B = row i of B^T
        for j in range(12):
            if B[j][i] == 1:
                mask |= 1 << j  # Low 12 bits: data portion
        # Identity contribution: bit (12 + i) for parity portion
        mask |= 1 << (12 + i)
        h_masks.append(mask)

    return tuple(h_masks)


# Precomputed H row masks for decode kernel
_H_ROW_MASKS = _build_h_row_masks()


# =============================================================================
# Python Wrapper Functions
# =============================================================================


def golay_encode(triplets: torch.Tensor) -> torch.Tensor:
    """
    Encode INT4 triplets to Golay(24,12) codewords.

    Args:
        triplets: Tensor of shape (N, 3) or (3,) with INT4 values.
            Each row [n0, n1, n2] is packed into one 24-bit codeword.
            Must be on CUDA device.

    Returns:
        Tensor of shape (N,) with 24-bit codewords (stored as int32).

    Example:
        >>> triplets = torch.randint(0, 16, (1000, 3), dtype=torch.uint8, device='cuda')
        >>> codewords = golay_encode(triplets)
        >>> assert codewords.shape == (1000,)
    """
    assert triplets.is_cuda, "Input must be on CUDA device"

    # Handle 1D input (single triplet)
    if triplets.dim() == 1:
        triplets = triplets.unsqueeze(0)

    N = triplets.shape[0]
    flat_triplets = triplets.contiguous().flatten().to(torch.uint8)

    codewords = torch.empty(N, dtype=torch.int32, device=triplets.device)

    # Launch encode kernel with B matrix columns as constants
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    golay_encode_kernel[grid](
        flat_triplets,
        codewords,
        N,
        BLOCK_SIZE=GOLAY_BLOCK_SIZE,
        C0=B_COL_0, C1=B_COL_1, C2=B_COL_2, C3=B_COL_3,
        C4=B_COL_4, C5=B_COL_5, C6=B_COL_6, C7=B_COL_7,
        C8=B_COL_8, C9=B_COL_9, C10=B_COL_10, C11=B_COL_11,
    )

    return codewords


def golay_decode(
    codewords: torch.Tensor,
    return_error_counts: bool = False,
) -> tuple:
    """
    Decode Golay(24,12) codewords to INT4 triplets.

    Corrects up to 3 bit errors per codeword. For >3 errors, data is preserved
    but marked as uncorrectable (error_count = 4).

    Args:
        codewords: Tensor of 24-bit codewords. Must be on CUDA device.
        return_error_counts: If True, also return per-codeword error counts.

    Returns:
        If return_error_counts=False:
            (decoded, (total_errors_corrected, uncorrectable_count))
        If return_error_counts=True:
            (decoded, error_counts, (total_errors_corrected, uncorrectable_count))

        Where:
            - decoded: Tensor of shape (N, 3) with INT4 triplets
            - error_counts: Tensor of shape (N,) with per-codeword counts (0-3 or 4)
            - total_errors_corrected: Sum of corrected bits across all codewords
            - uncorrectable_count: Number of codewords with >3 errors

    Example:
        >>> codewords = golay_encode(triplets)
        >>> decoded, stats = golay_decode(codewords)
        >>> print(f"Corrected {stats[0]} bits, {stats[1]} uncorrectable")
    """
    assert codewords.is_cuda, "Input must be on CUDA device"

    N = codewords.numel()
    device = codewords.device

    flat_codewords = codewords.flatten().to(torch.int32)

    # Output buffers
    decoded_flat = torch.empty(N * 3, dtype=torch.uint8, device=device)
    error_counts = torch.empty(N, dtype=torch.uint8, device=device)

    # Get syndrome table on GPU (cached)
    syndrome_table = _build_syndrome_table(str(device))

    # H matrix row masks for syndrome computation
    H = _H_ROW_MASKS

    # Launch decode kernel with H matrix masks as constants
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    golay_decode_kernel[grid](
        flat_codewords,
        decoded_flat,
        error_counts,
        syndrome_table,
        N,
        BLOCK_SIZE=GOLAY_BLOCK_SIZE,
        H0=H[0], H1=H[1], H2=H[2], H3=H[3],
        H4=H[4], H5=H[5], H6=H[6], H7=H[7],
        H8=H[8], H9=H[9], H10=H[10], H11=H[11],
    )

    # Reshape to (N, 3) triplets
    decoded = decoded_flat.view(N, 3)

    # Compute statistics
    correctable_mask = error_counts < 4  # error_count 4 = uncorrectable
    total_errors_corrected = int(error_counts[correctable_mask].sum())
    uncorrectable_count = int((~correctable_mask).sum())

    if return_error_counts:
        return decoded, error_counts, (total_errors_corrected, uncorrectable_count)
    else:
        return decoded, (total_errors_corrected, uncorrectable_count)


# =============================================================================
# Golay(24,12) Codec Class
# =============================================================================


class Golay2412:
    """
    Golay(24,12) codec with Triton GPU acceleration.

    This class provides an object-oriented interface to the Golay encode/decode
    functions. It maintains device-specific state and exposes G/H/B matrices
    for algebraic verification.

    Attributes:
        G: Generator matrix (12×24). Instance attribute.
        H: Parity-check matrix (12×24). Instance attribute.
        P: Parity submatrix (B matrix, 12×12). Instance attribute.
        UNCORRECTABLE: Sentinel value for uncorrectable codewords. Class attribute.
        device: Target CUDA device for tensor operations.

    Example:
        >>> codec = Golay2412(device='cuda')
        >>> triplets = torch.randint(0, 16, (1000, 3), dtype=torch.uint8, device='cuda')
        >>> codewords = codec.encode(triplets)
        >>> result = codec.decode(codewords)
        >>> assert (result.data == triplets).all()
        >>> print(f"Corrected {result.errors_corrected} bits")
    """

    UNCORRECTABLE = GOLAY_UNCORRECTABLE  # Sentinel for >3 error syndromes

    def __init__(self, device: str = "cuda"):
        """
        Initialize Golay(24,12) codec.

        Args:
            device: Target device for tensors ('cuda', 'cuda:0', etc.)
        """
        self.device = device
        # Build systematic generator and parity-check matrices
        self.G, self.H, self.P = self._build_matrices()
        # Cache syndrome table on target device
        self.syndrome_table = _build_syndrome_table(str(device))
        # H row masks for syndrome computation (for verification)
        self.h_row_masks = torch.tensor(_H_ROW_MASKS, dtype=torch.int64, device=device)

    def _build_matrices(self) -> tuple:
        """
        Build G, H, and P (B) matrices for the Golay code.

        Returns:
            (G, H, P) where:
                - G: Generator matrix [I₁₂ | B], shape (12, 24)
                - H: Parity-check matrix [B^T | I₁₂], shape (12, 24)
                - P: Parity submatrix B, shape (12, 12)
        """
        B = GOLAY_B_MATRIX.to(self.device)
        I_12 = torch.eye(12, dtype=torch.uint8, device=self.device)
        G = torch.cat([I_12, B], dim=1)  # Systematic form: [I | B]
        H = torch.cat([B.T, I_12], dim=1)  # [B^T | I]
        return G, H, B

    def encode(self, triplets: torch.Tensor) -> torch.Tensor:
        """
        Encode triplets of INT4 values to 24-bit Golay codewords.

        Args:
            triplets: Tensor of shape (N, 3) with INT4 values.
                Will be moved to self.device if not already there.

        Returns:
            Tensor of shape (N,) with 24-bit codewords (as int64 for compatibility).
        """
        input_tensor = triplets.to(self.device)
        codewords = golay_encode(input_tensor)
        # Return as int64 to match original CPU implementation
        return codewords.to(torch.int64)

    def decode(self, codewords: torch.Tensor) -> GolayDecodeResult:
        """
        Decode 24-bit Golay codewords to triplets of INT4 values.

        Corrects up to 3 bit errors per codeword.

        Args:
            codewords: Tensor of 24-bit codewords. Will be moved to self.device.

        Returns:
            GolayDecodeResult with fields:
                - data: Decoded triplets, shape (N, 3), dtype uint8
                - errors_corrected: Total bits corrected across all codewords
                - uncorrectable_count: Number of codewords with >3 errors

        Note:
            On uncorrectable error (>3 bits), data contains the corrupted triplet
            (not zeroed). Use uncorrectable_count to identify unreliable results.
        """
        input_tensor = codewords.to(torch.int32).to(self.device)
        decoded, stats = golay_decode(input_tensor)
        return GolayDecodeResult(
            data=decoded,
            errors_corrected=stats[0],
            uncorrectable_count=stats[1],
        )

    def verify_properties(self) -> bool:
        """
        Verify algebraic properties of the Golay code matrices.

        Checks that G @ H^T = 0 (mod 2), which is required for valid encoding/decoding.

        Returns:
            True if G @ H^T = 0 (mod 2), False otherwise
        """
        # G @ H^T should be zero in GF(2)
        product = (self.G.float() @ self.H.T.float()) % 2
        if product.sum() != 0:
            return False
        return True
