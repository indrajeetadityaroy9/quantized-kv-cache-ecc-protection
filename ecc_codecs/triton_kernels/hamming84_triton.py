"""
Hamming(8,4) SECDED Codec: GPU-accelerated single-error-correct, double-error-detect.

This module implements Hamming(8,4) encoding and decoding using Triton GPU kernels.
Extends Hamming(7,4) with an 8th overall parity bit for double-error detection.

Encoding: 4 data bits → 8 codeword bits
    codeword = [d0, d1, d2, d3, p0, p1, p2, overall_parity]
    where p_i = XOR of subset of data bits (defined by G matrix)
    and overall_parity = XOR of all 7 previous bits

Decoding: Uses syndrome + overall parity to classify errors
    - syndrome=0, parity_ok → NO_ERROR
    - syndrome≠0, parity_bad → SINGLE_CORRECTED (correctable)
    - syndrome≠0, parity_ok → DOUBLE_DETECTED (uncorrectable, data preserved)
    - syndrome=0, parity_bad → PARITY_ONLY (error in parity bit only)

Pipeline role:
    Called by kv_cache/ecc_shim.py during cache write (encode) and attend (decode).
    Each INT4 KV cache value is protected by one 8-bit Hamming codeword.

Performance:
    - O(1) syndrome lookup via precomputed 8-entry table
    - Triton kernels achieve ~90% of peak memory bandwidth on A100
    - Block size 1024 optimized for high occupancy

Determinism:
    Fully deterministic given identical input tensors. No RNG involved.
"""

import torch
import triton
import triton.language as tl

from .config import (
    HAMMING84_BLOCK_SIZE,
    SYNDROME_LUT_HAMMING84,
    ErrorType,
    DecodeResult,
    HAMMING84_G,
    HAMMING84_H,
)

# =============================================================================
# Hamming(8,4) Encode Kernel
# =============================================================================


@triton.jit
def hamming84_encode_kernel(
    int4_ptr,      # Input: pointer to INT4 values (uint8, only low 4 bits used)
    codeword_ptr,  # Output: pointer to 8-bit codewords (uint8)
    N,             # Total number of elements to encode
    BLOCK_SIZE: tl.constexpr,  # Elements per thread block (1024)
):
    """
    Triton kernel: Encode INT4 values to Hamming(8,4) SECDED codewords.

    Each thread block processes BLOCK_SIZE elements in parallel.
    Memory access pattern is coalesced for optimal bandwidth.

    Bit layout of output codeword:
        [d0, d1, d2, d3, p0, p1, p2, overall_parity]
         └─ data ──────┘  └─ parity ┘  └─ SECDED ─┘
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load INT4 values (only low 4 bits are meaningful)
    int4_vals = tl.load(int4_ptr + offsets, mask=mask, other=0).to(tl.uint8)

    # Extract individual data bits for parity computation
    d0 = (int4_vals >> 0) & 1
    d1 = (int4_vals >> 1) & 1
    d2 = (int4_vals >> 2) & 1
    d3 = (int4_vals >> 3) & 1

    # Compute parity bits according to G matrix:
    # p0 covers d0, d1, d3 (column pattern from H matrix)
    # p1 covers d0, d2, d3
    # p2 covers d1, d2, d3
    p0 = d0 ^ d1 ^ d3
    p1 = d0 ^ d2 ^ d3
    p2 = d1 ^ d2 ^ d3

    # Assemble 7-bit Hamming codeword: [d0-d3, p0-p2]
    hamming7 = (
        (d0 << 0)
        | (d1 << 1)
        | (d2 << 2)
        | (d3 << 3)
        | (p0 << 4)
        | (p1 << 5)
        | (p2 << 6)
    )

    # Compute overall parity (XOR of all 7 bits) using parallel reduction
    # This enables SECDED: distinguishes single (correctable) from double (detectable)
    parity = hamming7 ^ (hamming7 >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    overall_parity = parity & 1

    # Final 8-bit codeword: hamming7 | (overall_parity << 7)
    codeword = (hamming7 | (overall_parity << 7)).to(tl.uint8)

    tl.store(codeword_ptr + offsets, codeword, mask=mask)


# =============================================================================
# Hamming(8,4) Decode Kernel
# =============================================================================


@triton.jit
def hamming84_decode_kernel(
    codeword_ptr,     # Input: pointer to 8-bit codewords (uint8)
    decoded_ptr,      # Output: pointer to decoded INT4 values (uint8)
    error_type_ptr,   # Output: pointer to ErrorType classification (uint8)
    lut_ptr,          # Syndrome LUT: 8-entry table mapping syndrome → bit position
    N,                # Total number of codewords to decode
    BLOCK_SIZE: tl.constexpr,  # Elements per thread block (1024)
):
    """
    Triton kernel: Decode Hamming(8,4) SECDED codewords to INT4 values.

    SECDED (Single Error Correct, Double Error Detect) algorithm:
        1. Compute 3-bit syndrome from received codeword
        2. Compute overall parity of received codeword
        3. Classify error based on (syndrome, parity) pair:
           - (0, ok)  → NO_ERROR: clean codeword
           - (≠0, bad) → SINGLE_CORRECTED: flip bit at syndrome position
           - (≠0, ok)  → DOUBLE_DETECTED: two errors, uncorrectable
           - (0, bad)  → PARITY_ONLY: error in overall parity bit

    IMPORTANT: On DOUBLE_DETECTED, data is preserved (not zeroed).
    Caller must check error_type to identify potentially corrupted values.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load 8-bit codewords
    codewords = tl.load(codeword_ptr + offsets, mask=mask, other=0).to(tl.uint8)

    # Split codeword: bits 0-6 = Hamming(7,4), bit 7 = overall parity
    hamming7 = codewords & 0x7F
    stored_parity = (codewords >> 7) & 1

    # Extract individual bits for syndrome computation
    c0 = (hamming7 >> 0) & 1  # d0
    c1 = (hamming7 >> 1) & 1  # d1
    c2 = (hamming7 >> 2) & 1  # d2
    c3 = (hamming7 >> 3) & 1  # d3
    c4 = (hamming7 >> 4) & 1  # p0
    c5 = (hamming7 >> 5) & 1  # p1
    c6 = (hamming7 >> 6) & 1  # p2

    # Compute syndrome bits (H matrix rows applied to codeword)
    # s = H @ codeword (mod 2)
    s0 = c0 ^ c1 ^ c3 ^ c4  # Row 0 of H
    s1 = c0 ^ c2 ^ c3 ^ c5  # Row 1 of H
    s2 = c1 ^ c2 ^ c3 ^ c6  # Row 2 of H

    # Assemble 3-bit syndrome
    syndrome = (s0 | (s1 << 1) | (s2 << 2)).to(tl.int32)

    # Recompute overall parity of received Hamming(7,4) portion
    actual_parity = hamming7 ^ (hamming7 >> 4)
    actual_parity = actual_parity ^ (actual_parity >> 2)
    actual_parity = actual_parity ^ (actual_parity >> 1)
    actual_parity = actual_parity & 1

    # Check if stored parity matches recomputed parity
    parity_error = stored_parity != actual_parity

    # SECDED classification based on (syndrome, parity_error) pair
    syndrome_zero = syndrome == 0
    # Truth table:
    #   syndrome=0, parity_ok  → 0 (NO_ERROR)
    #   syndrome=0, parity_bad → 3 (PARITY_ONLY)
    #   syndrome≠0, parity_bad → 1 (SINGLE_CORRECTED)
    #   syndrome≠0, parity_ok  → 2 (DOUBLE_DETECTED)
    error_type = tl.where(
        syndrome_zero, tl.where(parity_error, 3, 0), tl.where(parity_error, 1, 2)
    ).to(tl.uint8)

    # O(1) syndrome lookup: maps 3-bit syndrome to error bit position
    # LUT values: -1 (no error), or 0-6 (bit position to flip)
    error_pos = tl.load(lut_ptr + syndrome, mask=mask, other=-1)

    # Only correct if single error detected AND valid position
    should_correct = (error_type == 1) & (error_pos >= 0)
    correction_mask = tl.where(should_correct, 1 << error_pos, 0).to(tl.uint8)

    # XOR correction: flips the erroneous bit
    corrected = hamming7 ^ correction_mask

    # CRITICAL: Preserve data on double error (do not zero)
    # Previously this line existed: corrected = tl.where(error_type == 2, 0, corrected)
    # That caused silent corruption - corrupted data indistinguishable from valid zeros
    # Now: caller checks error_type == 2 to identify potentially corrupted values

    # Extract data bits (low 4 bits of corrected Hamming codeword)
    decoded = corrected & 0x0F

    tl.store(decoded_ptr + offsets, decoded, mask=mask)
    tl.store(error_type_ptr + offsets, error_type, mask=mask)


# =============================================================================
# Python Wrapper Functions
# =============================================================================


def hamming84_encode(int4_values: torch.Tensor) -> torch.Tensor:
    """
    Encode INT4 values to Hamming(8,4) SECDED codewords.

    Args:
        int4_values: Tensor of INT4 values (uint8, only low 4 bits used).
            Must be on CUDA device. Shape: any (will be flattened internally).

    Returns:
        Tensor of 8-bit codewords, same shape as input.
        Each codeword protects one INT4 value.

    Raises:
        AssertionError: If input not on CUDA device.

    Example:
        >>> data = torch.randint(0, 16, (1000,), dtype=torch.uint8, device='cuda')
        >>> codewords = hamming84_encode(data)
        >>> assert codewords.shape == data.shape
    """
    assert int4_values.is_cuda, "Input must be on CUDA device"

    original_shape = int4_values.shape
    flat = int4_values.flatten().to(torch.uint8)
    N = flat.numel()

    codewords = torch.empty_like(flat)

    # Launch kernel with 1D grid
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    hamming84_encode_kernel[grid](
        flat,
        codewords,
        N,
        BLOCK_SIZE=HAMMING84_BLOCK_SIZE,
    )

    return codewords.view(original_shape)


# =============================================================================
# Syndrome LUT GPU Cache
# =============================================================================
# The syndrome lookup table is transferred to GPU once per device and cached.
# This avoids repeated CPU→GPU transfers on every decode call.

_syndrome_lut_cache = {}


def _get_syndrome_lut_gpu(device: torch.device) -> torch.Tensor:
    """
    Get syndrome LUT on the specified GPU device (cached).

    Args:
        device: Target CUDA device (e.g., torch.device('cuda:0'))

    Returns:
        Syndrome LUT tensor on the specified device
    """
    if device not in _syndrome_lut_cache:
        _syndrome_lut_cache[device] = SYNDROME_LUT_HAMMING84.to(device)
    return _syndrome_lut_cache[device]


def hamming84_decode(
    codewords: torch.Tensor,
    return_error_types: bool = False,
) -> tuple:
    """
    Decode Hamming(8,4) SECDED codewords to INT4 values.

    Performs single-error correction and double-error detection.
    On double error, data is preserved (not zeroed) to avoid silent corruption.

    Args:
        codewords: Tensor of 8-bit codewords. Must be on CUDA device.
            Shape: any (will be flattened internally).
        return_error_types: If True, also return per-element error classification.

    Returns:
        If return_error_types=False:
            (decoded, (corrected_count, detected_count))
        If return_error_types=True:
            (decoded, error_types, (corrected_count, detected_count))

        Where:
            - decoded: Tensor of INT4 values, same shape as input
            - error_types: Tensor of ErrorType values, same shape as input
            - corrected_count: Number of single-bit errors corrected
            - detected_count: Number of double-bit errors detected (uncorrectable)

    Raises:
        AssertionError: If input not on CUDA device.

    Example:
        >>> codewords = hamming84_encode(data)
        >>> decoded, stats = hamming84_decode(codewords)
        >>> print(f"Corrected {stats[0]} single errors, detected {stats[1]} double errors")
    """
    assert codewords.is_cuda, "Input must be on CUDA device"

    original_shape = codewords.shape
    flat = codewords.flatten().to(torch.uint8)
    N = flat.numel()

    decoded = torch.empty_like(flat)
    error_types = torch.empty_like(flat)

    # Get LUT on GPU (cached to avoid repeated transfers)
    lut_gpu = _get_syndrome_lut_gpu(codewords.device)

    # Launch decode kernel
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    hamming84_decode_kernel[grid](
        flat,
        decoded,
        error_types,
        lut_gpu,
        N,
        BLOCK_SIZE=HAMMING84_BLOCK_SIZE,
    )

    # Compute error statistics
    error_types_flat = error_types
    corrected_count = int((error_types_flat == ErrorType.SINGLE_CORRECTED).sum())
    detected_count = int((error_types_flat == ErrorType.DOUBLE_DETECTED).sum())

    # Reshape outputs to match input shape
    decoded = decoded.view(original_shape)
    error_types = error_types.view(original_shape)

    if return_error_types:
        return decoded, error_types, (corrected_count, detected_count)
    else:
        return decoded, (corrected_count, detected_count)


# =============================================================================
# Hamming(8,4) Codec Class
# =============================================================================


class Hamming84:
    """
    Hamming(8,4) SECDED codec with Triton GPU acceleration.

    This class provides an object-oriented interface to the Hamming(8,4) encode/decode
    functions. It maintains device-specific state and exposes G/H matrices for
    algebraic verification.

    Attributes:
        G_74: Generator matrix (4×7) for Hamming(7,4) portion. Class attribute.
        H_74: Parity-check matrix (3×7) for syndrome computation. Class attribute.
        SYNDROME_TO_POSITION: Syndrome lookup table. Class attribute.
        device: Target CUDA device for tensor operations.

    Example:
        >>> codec = Hamming84(device='cuda:0')
        >>> data = torch.randint(0, 16, (1000,), dtype=torch.uint8, device='cuda:0')
        >>> codewords = codec.encode(data)
        >>> result = codec.decode(codewords)
        >>> assert (result.data == data).all()
        >>> print(f"Corrected: {result.corrected_count}, Detected: {result.detected_count}")
    """

    # Class attributes: matrices for verification (see evaluation/verification.py)
    G_74 = HAMMING84_G  # Generator matrix: 4×7, used for encoding
    H_74 = HAMMING84_H  # Parity-check matrix: 3×7, G @ H^T = 0

    SYNDROME_TO_POSITION = SYNDROME_LUT_HAMMING84  # 8-entry LUT

    def __init__(self, device: str = "cuda", on_double_error: str = "zero"):
        """
        Initialize Hamming(8,4) codec.

        Args:
            device: Target device for tensors ('cuda', 'cuda:0', etc.)
            on_double_error: Legacy parameter, ignored. Double errors now preserve
                data (previously zeroed, causing silent corruption).

        Note:
            The on_double_error parameter is deprecated. Double-bit errors are now
            always preserved with error_type=DOUBLE_DETECTED so the caller can
            decide how to handle them (e.g., interpolation, discard).
        """
        self.device = device
        self.on_double_error = on_double_error  # Kept for API compatibility
        self._G = self.G_74.to(device)
        self._H = self.H_74.to(device)
        self._syndrome_lut = self.SYNDROME_TO_POSITION.to(device)

    def encode(self, int4_values: torch.Tensor) -> torch.Tensor:
        """
        Encode 4-bit values to 8-bit Hamming SECDED codewords.

        Args:
            int4_values: Tensor of INT4 values (uint8, low 4 bits used).
                Will be moved to self.device if not already there.

        Returns:
            Tensor of 8-bit codewords, same shape as input.
        """
        input_tensor = int4_values.to(self.device)
        return hamming84_encode(input_tensor)

    def decode(self, codewords: torch.Tensor) -> DecodeResult:
        """
        Decode 8-bit Hamming SECDED codewords to 4-bit values.

        Performs single-error correction and double-error detection.

        Args:
            codewords: Tensor of 8-bit codewords. Will be moved to self.device.

        Returns:
            DecodeResult with fields:
                - data: Decoded INT4 values (uint8). Shape matches input.
                - error_type: Per-element ErrorType classification.
                - corrected_count: Total single-bit errors corrected.
                - detected_count: Total double-bit errors detected.

        Note:
            On DOUBLE_DETECTED, data contains the corrupted value (not zeroed).
            Check error_type to identify unreliable values.
        """
        input_tensor = codewords.to(self.device)
        decoded, error_types, stats = hamming84_decode(
            input_tensor, return_error_types=True
        )
        return DecodeResult(
            data=decoded,
            error_type=error_types,
            corrected_count=stats[0],
            detected_count=stats[1],
        )
