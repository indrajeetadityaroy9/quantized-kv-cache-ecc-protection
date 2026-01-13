"""
ECC Codec Configuration: Syndrome tables and generator/parity-check matrices.

This module defines the algebraic structures (G, H matrices) and precomputed lookup
tables (syndrome LUTs) for Hamming and Golay error-correcting codes. These enable
O(1) syndrome-based decoding in GPU kernels.

Codes implemented:
    - Hamming(7,4) SEC: Single-error-correcting, 4 data → 7 codeword bits
    - Hamming(8,4) SECDED: Single-error-correct, double-error-detect (extended)
    - Golay(24,12): Perfect 3-error-correcting code, 12 data → 24 codeword bits

Pipeline role:
    Imported by triton_kernels/*.py for syndrome lookup during decode.
    Generator matrices used by evaluation/verification.py for algebraic validation.

Assumptions:
    - Systematic encoding: data bits occupy low-order positions in codeword
    - Binary field GF(2): all arithmetic is XOR-based
    - Syndrome table maps 12-bit syndrome → 24-bit error pattern (Golay)
"""

import torch
from typing import NamedTuple


# =============================================================================
# Triton Kernel Block Sizes
# =============================================================================
# Block sizes tuned for GPU occupancy on A100/H100. Each kernel processes
# BLOCK_SIZE elements per thread block. Smaller for Golay due to higher
# register pressure from 24-bit operations.

HAMMING74_BLOCK_SIZE = 1024  # 1K elements/block: high occupancy for 7-bit ops
HAMMING84_BLOCK_SIZE = 1024  # 1K elements/block: matches Hamming(7,4)
GOLAY_BLOCK_SIZE = 256       # 256 elements/block: reduced due to 24-bit state
FAULT_INJECTION_BLOCK_SIZE = 1024  # 1K elements/block: simple bit-flip ops
INTERPOLATION_BLOCK_SIZE = 1024    # 1K elements/block: float arithmetic


def get_physical_dtype(codec: str) -> torch.dtype:
    """
    Return the physical storage dtype for a given ECC codec.

    The dtype determines memory layout and codeword capacity:
        - uint8: Hamming (7 or 8 bits) and INT4 (4 bits, packed)
        - int32: Golay (24 bits requires >16 bits)
        - float16: Unprotected baseline (no quantization)

    Args:
        codec: One of "hamming74", "hamming84", "golay", "int4", "none"

    Returns:
        torch.dtype for codeword storage

    Raises:
        ValueError: Unknown codec name
    """
    if codec == "hamming74":
        return torch.uint8  # 7-bit codeword fits in uint8
    elif codec == "hamming84":
        return torch.uint8  # 8-bit codeword = 1 byte exactly
    elif codec == "golay":
        return torch.int32  # 24-bit codeword requires 32-bit storage
    elif codec == "int4":
        return torch.uint8  # 4-bit value stored in uint8 (one nibble)
    elif codec == "none":
        return torch.float16  # FP16 oracle baseline
    else:
        raise ValueError(f"Unknown codec: {codec}")


def get_codeword_bits(codec: str) -> int:
    """
    Return the number of bits per codeword for a given ECC codec.

    Used for bandwidth overhead calculations: overhead = codeword_bits / data_bits.

    Args:
        codec: One of "hamming74", "hamming84", "golay"

    Returns:
        Total bits in encoded codeword
    """
    if codec == "hamming74":
        return 7   # 4 data + 3 parity
    elif codec == "hamming84":
        return 8   # 4 data + 3 parity + 1 overall parity (SECDED)
    elif codec == "golay":
        return 24  # 12 data + 12 parity (perfect code)
    else:
        raise ValueError(f"Unknown codec: {codec}")


def get_data_bits(codec: str) -> int:
    """
    Return the number of data bits per codeword for a given ECC codec.

    Used for rate calculation: code_rate = data_bits / codeword_bits.

    Args:
        codec: One of "hamming74", "hamming84", "golay"

    Returns:
        Number of information bits encoded per codeword
    """
    if codec == "hamming74":
        return 4   # Encodes one INT4 value
    elif codec == "hamming84":
        return 4   # Same data capacity as Hamming(7,4), extra bit for detection
    elif codec == "golay":
        return 12  # Encodes three INT4 values (triplet)
    else:
        raise ValueError(f"Unknown codec: {codec}")


# =============================================================================
# Hamming Syndrome Lookup Tables
# =============================================================================
# Syndrome LUTs enable O(1) error location during decode.
#
# For Hamming(7,4), syndrome s = H @ received (mod 2) where H is 3×7.
# The 3-bit syndrome indexes this table to find the bit position with an error.
#
# Table construction: syndrome s corresponds to the s-th column of H.
# Entry -1 means syndrome=0 (no error detected).
#
# Bit layout in codeword: [d0, d1, d2, d3, p0, p1, p2]
#                          ↑ positions 0-3: data     ↑ positions 4-6: parity

SYNDROME_LUT_HAMMING74 = torch.tensor(
    [
        -1,  # syndrome=0: no error
        4,   # syndrome=1: error in bit 4 (p0)
        5,   # syndrome=2: error in bit 5 (p1)
        0,   # syndrome=3: error in bit 0 (d0)
        6,   # syndrome=4: error in bit 6 (p2)
        1,   # syndrome=5: error in bit 1 (d1)
        2,   # syndrome=6: error in bit 2 (d2)
        3,   # syndrome=7: error in bit 3 (d3)
    ],
    dtype=torch.int8,
)


# Hamming(8,4) uses same syndrome table for the 7-bit portion.
# The 8th bit (overall parity) is handled separately in the kernel
# to distinguish single errors (correctable) from double errors (detectable only).
SYNDROME_LUT_HAMMING84 = torch.tensor(
    [
        -1,  # syndrome=0: check overall parity to distinguish no-error vs double-error
        4,   # syndrome=1: single error in p0 (if overall parity also wrong)
        5,   # syndrome=2: single error in p1
        0,   # syndrome=3: single error in d0
        6,   # syndrome=4: single error in p2
        1,   # syndrome=5: single error in d1
        2,   # syndrome=6: single error in d2
        3,   # syndrome=7: single error in d3
    ],
    dtype=torch.int8,
)


# =============================================================================
# Golay(24,12) Parity Check Masks
# =============================================================================
# Each mask represents a row of the parity portion of H (the B^T matrix).
# Used to compute the 12-bit syndrome during decode.
#
# Golay(24,12) structure: codeword = [data₁₂ | parity₁₂]
# H = [B^T | I₁₂] where B is the 12×12 circulant matrix below.

GOLAY_PARITY_MASKS = torch.tensor(
    [
        0b110111000111,  # Row 0 of B^T: columns of B that contribute to p0
        0b101110001110,  # Row 1: contributes to p1
        0b011100011101,  # Row 2: contributes to p2
        0b111000111010,  # Row 3: contributes to p3
        0b110001110101,  # Row 4: contributes to p4
        0b100011101011,  # Row 5: contributes to p5
        0b000111010111,  # Row 6: contributes to p6
        0b001110101110,  # Row 7: contributes to p7
        0b011101011100,  # Row 8: contributes to p8
        0b111010111000,  # Row 9: contributes to p9
        0b110101110001,  # Row 10: contributes to p10
        0b101011100011,  # Row 11: contributes to p11
    ],
    dtype=torch.int32,
)


# =============================================================================
# Error Classification Constants
# =============================================================================


class ErrorType:
    """
    SECDED error classification for Hamming(8,4) decode.

    Decoding uses syndrome (3-bit) and overall parity to classify:
        - syndrome=0, parity_ok → NO_ERROR
        - syndrome≠0, parity_fail → SINGLE_CORRECTED (correctable)
        - syndrome≠0, parity_ok → DOUBLE_DETECTED (uncorrectable, data preserved)
        - syndrome=0, parity_fail → PARITY_ONLY (error in overall parity bit)

    Note: DOUBLE_DETECTED means data may be corrupted. Caller should check
    error_type and potentially apply interpolation or discard the value.
    """

    NO_ERROR = 0           # Clean codeword, no correction needed
    SINGLE_CORRECTED = 1   # Single bit error found and corrected
    DOUBLE_DETECTED = 2    # Double bit error detected (uncorrectable)
    PARITY_ONLY = 3        # Error only in overall parity bit (data correct)


# =============================================================================
# Decode Result Types
# =============================================================================


class DecodeResult(NamedTuple):
    """
    Result from Hamming(8,4) SECDED decode operation.

    Attributes:
        data: Decoded INT4 values. Shape matches input codeword shape.
            On DOUBLE_DETECTED, contains corrupted data (not zeroed).
        error_type: Per-element ErrorType classification. Same shape as data.
        corrected_count: Total SINGLE_CORRECTED across all elements.
        detected_count: Total DOUBLE_DETECTED across all elements.

    Example:
        result = codec.decode(codewords)
        reliable_mask = result.error_type != ErrorType.DOUBLE_DETECTED
        reliable_data = result.data[reliable_mask]
    """

    data: torch.Tensor        # Shape: same as input, dtype: uint8
    error_type: torch.Tensor  # Shape: same as input, dtype: uint8 (ErrorType values)
    corrected_count: int      # Scalar: number of single-bit corrections performed
    detected_count: int       # Scalar: number of double-bit errors detected


class GolayDecodeResult(NamedTuple):
    """
    Result from Golay(24,12) decode operation.

    Golay corrects up to 3 errors per 24-bit codeword. Errors beyond 3 bits
    are uncorrectable and marked with error_count >= 4.

    Attributes:
        data: Decoded INT4 triplets. Shape: (N, 3) for N codewords.
            On uncorrectable error, contains corrupted data (not zeroed).
        errors_corrected: Total bit errors corrected across all codewords.
        uncorrectable_count: Number of codewords with >3 bit errors.

    Example:
        result = codec.decode(codewords)
        if result.uncorrectable_count > 0:
            # Some triplets may be corrupted
            ...
    """

    data: torch.Tensor      # Shape: (N, 3), dtype: uint8 (three INT4 values)
    errors_corrected: int   # Scalar: total bits corrected (0-3 per codeword)
    uncorrectable_count: int  # Scalar: codewords with >3 errors


# =============================================================================
# Hamming Generator and Parity-Check Matrices
# =============================================================================
# These matrices define the algebraic structure of Hamming codes.
#
# Generator matrix G (4×7): Encodes 4 data bits → 7 codeword bits
#   codeword = data @ G (mod 2)
#   Systematic form: G = [I₄ | P] where P is the parity submatrix
#
# Parity-check matrix H (3×7): Computes syndrome for error detection
#   syndrome = H @ received^T (mod 2)
#   If syndrome ≠ 0, syndrome indicates error position
#
# Key property: G @ H^T = 0 (null space relationship)

HAMMING74_G = torch.tensor(
    [
        # d0 d1 d2 d3 p0 p1 p2  (bit positions in codeword)
        [1, 0, 0, 0, 1, 1, 0],  # Row 0: d0 contributes to p0, p1
        [0, 1, 0, 0, 1, 0, 1],  # Row 1: d1 contributes to p0, p2
        [0, 0, 1, 0, 0, 1, 1],  # Row 2: d2 contributes to p1, p2
        [0, 0, 0, 1, 1, 1, 1],  # Row 3: d3 contributes to p0, p1, p2
    ],
    dtype=torch.uint8,
)

HAMMING74_H = torch.tensor(
    [
        # c0 c1 c2 c3 c4 c5 c6  (bit positions: d0-d3, p0-p2)
        [1, 1, 0, 1, 1, 0, 0],  # Row 0: s0 = d0⊕d1⊕d3⊕p0
        [1, 0, 1, 1, 0, 1, 0],  # Row 1: s1 = d0⊕d2⊕d3⊕p1
        [0, 1, 1, 1, 0, 0, 1],  # Row 2: s2 = d1⊕d2⊕d3⊕p2
    ],
    dtype=torch.uint8,
)

# Hamming(8,4) SECDED uses same G, H for the 7-bit portion.
# The 8th bit (overall parity) is computed separately:
#   overall_parity = XOR of all 7 bits in Hamming(7,4) codeword
HAMMING84_G = HAMMING74_G
HAMMING84_H = HAMMING74_H


# =============================================================================
# Golay(24,12) Matrix and Syndrome Table
# =============================================================================
# The extended binary Golay code is a perfect [24,12,8] code.
#
# Properties:
#   - Encodes 12 data bits → 24 codeword bits (rate 1/2)
#   - Minimum distance d=8: detects up to 7 errors, corrects up to 3
#   - Perfect code: all 2^12 syndromes map to unique correctable patterns
#
# Generator matrix G (12×24) = [I₁₂ | B]
# Parity-check matrix H (12×24) = [B^T | I₁₂]
#
# B is a 12×12 circulant matrix derived from the quadratic residues mod 11.
# B is symmetric and self-dual: B^T = B, and B @ B = I (mod 2).

GOLAY_B_MATRIX = torch.tensor(
    [
        # Each row defines parity bits for one data bit position
        # Row i: data bit i contributes to parity bits where column j = 1
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],  # Row 0
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],  # Row 1 (cyclic shift)
        [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],  # Row 2
        [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],  # Row 3
        [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # Row 4
        [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],  # Row 5
        [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],  # Row 6
        [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],  # Row 7
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],  # Row 8
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],  # Row 9
        [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],  # Row 10
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # Row 11 (all-ones except last)
    ],
    dtype=torch.uint8,
)

# Sentinel value for uncorrectable errors in syndrome table
# Any syndrome that doesn't map to a ≤3-error pattern returns this value
GOLAY_UNCORRECTABLE = 0xFFFFFFFF


def _compute_golay_h_row_masks() -> tuple:
    """
    Compute H matrix row masks for O(1) syndrome calculation.

    Each mask is a 24-bit integer representing one row of H = [B^T | I₁₂].
    Syndrome bit i = popcount(codeword & mask_i) mod 2.

    Returns:
        Tuple of 12 integers, each a 24-bit mask
    """
    B = GOLAY_B_MATRIX.tolist()
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


# Precomputed H row masks for syndrome calculation in Triton kernels
GOLAY_H_ROW_MASKS = _compute_golay_h_row_masks()


def _compute_syndrome_for_pattern(error_pattern: int, h_masks: tuple) -> int:
    """
    Compute 12-bit syndrome for a given 24-bit error pattern.

    syndrome[i] = popcount(error_pattern AND h_masks[i]) mod 2

    Args:
        error_pattern: 24-bit error pattern (1 = flipped bit)
        h_masks: Precomputed H row masks from _compute_golay_h_row_masks()

    Returns:
        12-bit syndrome value (0-4095)
    """
    syndrome = 0
    for i in range(12):
        masked = error_pattern & h_masks[i]
        parity = bin(masked).count("1") & 1  # popcount mod 2
        syndrome |= parity << i
    return syndrome


def build_golay_syndrome_table() -> torch.Tensor:
    """
    Build the Golay(24,12) syndrome lookup table for O(1) decoding.

    The table maps each 12-bit syndrome to its unique correctable error pattern.
    Golay is a perfect code: every syndrome corresponds to exactly one
    error pattern with weight ≤3.

    Table construction:
        1. Enumerate all 1-bit, 2-bit, and 3-bit error patterns (2324 total)
        2. Compute syndrome for each pattern
        3. Store pattern at table[syndrome]
        4. Uncorrectable syndromes (>3 errors) remain -1

    Returns:
        Tensor of shape (4096,) mapping syndrome → error_pattern.
        Value -1 indicates uncorrectable (>3 bit errors).

    Note:
        This is called once at module load time. Table is cached per GPU device
        in golay_triton.py for kernel access.
    """
    table = [-1] * 4096  # 2^12 possible syndromes

    # Syndrome 0 → no error (identity mapping)
    table[0] = 0

    h_masks = GOLAY_H_ROW_MASKS

    # Single-bit errors: 24 patterns, weight 1
    for i in range(24):
        error = 1 << i
        syndrome = _compute_syndrome_for_pattern(error, h_masks)
        table[syndrome] = error

    # Double-bit errors: C(24,2) = 276 patterns, weight 2
    for i in range(24):
        for j in range(i + 1, 24):
            error = (1 << i) | (1 << j)
            syndrome = _compute_syndrome_for_pattern(error, h_masks)
            if table[syndrome] == -1:  # First pattern for this syndrome
                table[syndrome] = error

    # Triple-bit errors: C(24,3) = 2024 patterns, weight 3
    for i in range(24):
        for j in range(i + 1, 24):
            for k in range(j + 1, 24):
                error = (1 << i) | (1 << j) | (1 << k)
                syndrome = _compute_syndrome_for_pattern(error, h_masks)
                if table[syndrome] == -1:  # First pattern for this syndrome
                    table[syndrome] = error

    # Total: 1 + 24 + 276 + 2024 = 2325 correctable patterns
    # Remaining syndromes (4096 - 2325 = 1771) are uncorrectable
    return torch.tensor(table, dtype=torch.int32)
