import torch
from typing import NamedTuple


# =============================================================================
# Block Sizes
# =============================================================================

HAMMING74_BLOCK_SIZE = 1024
HAMMING84_BLOCK_SIZE = 1024
GOLAY_BLOCK_SIZE = 256
FAULT_INJECTION_BLOCK_SIZE = 1024
INTERPOLATION_BLOCK_SIZE = 1024


def get_physical_dtype(codec):
    if codec == "hamming74":
        return torch.uint8
    elif codec == "hamming84":
        return torch.uint8
    elif codec == "golay":
        return torch.int32
    elif codec == "int4":
        return torch.uint8
    elif codec == "none":
        return torch.float16
    else:
        raise ValueError(f"Unknown codec: {codec}")


def get_codeword_bits(codec):
    if codec == "hamming74":
        return 7
    elif codec == "hamming84":
        return 8
    elif codec == "golay":
        return 24
    else:
        raise ValueError(f"Unknown codec: {codec}")


def get_data_bits(codec):
    if codec == "hamming74":
        return 4
    elif codec == "hamming84":
        return 4
    elif codec == "golay":
        return 12
    else:
        raise ValueError(f"Unknown codec: {codec}")


SYNDROME_LUT_HAMMING74 = torch.tensor(
    [
        -1,
        4,
        5,
        0,
        6,
        1,
        2,
        3,
    ],
    dtype=torch.int8,
)


SYNDROME_LUT_HAMMING84 = torch.tensor(
    [
        -1,
        4,
        5,
        0,
        6,
        1,
        2,
        3,
    ],
    dtype=torch.int8,
)


GOLAY_PARITY_MASKS = torch.tensor(
    [
        0b110111000111,
        0b101110001110,
        0b011100011101,
        0b111000111010,
        0b110001110101,
        0b100011101011,
        0b000111010111,
        0b001110101110,
        0b011101011100,
        0b111010111000,
        0b110101110001,
        0b101011100011,
    ],
    dtype=torch.int32,
)


class ErrorType:
    NO_ERROR = 0
    SINGLE_CORRECTED = 1
    DOUBLE_DETECTED = 2
    PARITY_ONLY = 3


# =============================================================================
# Decode Result Types
# =============================================================================


class DecodeResult(NamedTuple):
    """Result from Hamming(8,4) decode operation."""

    data: torch.Tensor
    error_type: torch.Tensor
    corrected_count: int
    detected_count: int


class GolayDecodeResult(NamedTuple):
    """Result from Golay(24,12) decode operation."""

    data: torch.Tensor
    errors_corrected: int
    uncorrectable_count: int


# =============================================================================
# Hamming Matrices (for verification)
# =============================================================================

HAMMING74_G = torch.tensor(
    [
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=torch.uint8,
)

HAMMING74_H = torch.tensor(
    [
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1],
    ],
    dtype=torch.uint8,
)

# Hamming(8,4) uses the same G and H matrices as Hamming(7,4)
HAMMING84_G = HAMMING74_G
HAMMING84_H = HAMMING74_H


# =============================================================================
# Golay Matrix and Syndrome Table
# =============================================================================

GOLAY_B_MATRIX = torch.tensor(
    [
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
    ],
    dtype=torch.uint8,
)

GOLAY_UNCORRECTABLE = 0xFFFFFFFF


def _compute_golay_h_row_masks():
    """Compute H matrix row masks for syndrome calculation."""
    B = GOLAY_B_MATRIX.tolist()
    h_masks = []
    for i in range(12):
        mask = 0
        # B^T columns become mask bits
        for j in range(12):
            if B[j][i] == 1:
                mask |= 1 << j
        # Identity part
        mask |= 1 << (12 + i)
        h_masks.append(mask)
    return tuple(h_masks)


GOLAY_H_ROW_MASKS = _compute_golay_h_row_masks()


def _compute_syndrome_for_pattern(error_pattern, h_masks):
    """Compute syndrome for a given error pattern."""
    syndrome = 0
    for i in range(12):
        masked = error_pattern & h_masks[i]
        parity = bin(masked).count("1") & 1
        syndrome |= parity << i
    return syndrome


def build_golay_syndrome_table():
    """
    Build the Golay(24,12) syndrome lookup table.

    Returns a tensor of shape (4096,) mapping syndrome -> error pattern.
    Uncorrectable syndromes are marked with -1.
    """
    table = [-1] * 4096

    # Syndrome 0 -> no error
    table[0] = 0

    h_masks = GOLAY_H_ROW_MASKS

    # Single-bit errors (24 patterns)
    for i in range(24):
        error = 1 << i
        syndrome = _compute_syndrome_for_pattern(error, h_masks)
        table[syndrome] = error

    # Double-bit errors (C(24,2) = 276 patterns)
    for i in range(24):
        for j in range(i + 1, 24):
            error = (1 << i) | (1 << j)
            syndrome = _compute_syndrome_for_pattern(error, h_masks)
            if table[syndrome] == -1:
                table[syndrome] = error

    # Triple-bit errors (C(24,3) = 2024 patterns)
    for i in range(24):
        for j in range(i + 1, 24):
            for k in range(j + 1, 24):
                error = (1 << i) | (1 << j) | (1 << k)
                syndrome = _compute_syndrome_for_pattern(error, h_masks)
                if table[syndrome] == -1:
                    table[syndrome] = error

    return torch.tensor(table, dtype=torch.int32)
