"""
Configuration for Triton ECC Kernels.

Defines block sizes, data types, and lookup tables for GPU-native codec operations.
"""

import torch
from typing import Literal

# =============================================================================
# Triton Block Sizes (threads per block)
# =============================================================================

# Hamming(7,4): Simple operations, can use large blocks
HAMMING74_BLOCK_SIZE = 1024

# Hamming(8,4): Simple operations, can use large blocks
HAMMING84_BLOCK_SIZE = 1024

# Golay(24,12): More register pressure due to LUT access
GOLAY_BLOCK_SIZE = 256

# Fault injection: Simple Bernoulli sampling
FAULT_INJECTION_BLOCK_SIZE = 1024

# Linear interpolation: Simple neighbor averaging
INTERPOLATION_BLOCK_SIZE = 1024


# =============================================================================
# Physical Storage Types
# =============================================================================

def get_physical_dtype(codec: Literal["hamming74", "hamming84", "golay", "int4", "none"]) -> torch.dtype:
    """
    Get the physical storage dtype for a codec.

    Args:
        codec: Codec name

    Returns:
        torch.dtype for storage

    Alignment considerations:
    - Hamming(7,4): 7-bit codeword → uint8 (1 bit wasted)
    - Hamming(8,4): 8-bit codeword → uint8 (perfect alignment)
    - Golay(24,12): 24-bit codeword → int32 (8 bits wasted for alignment)
    - INT4: 4-bit value → uint8 (2 values per byte, but stored as 1:1 for simplicity)
    """
    if codec == "hamming74":
        return torch.uint8  # 7 bits used, 1 bit wasted
    elif codec == "hamming84":
        return torch.uint8
    elif codec == "golay":
        return torch.int32  # 24 bits + 8 wasted for 32-bit alignment
    elif codec == "int4":
        return torch.uint8
    elif codec == "none":
        return torch.float16
    else:
        raise ValueError(f"Unknown codec: {codec}")


def get_codeword_bits(codec: Literal["hamming74", "hamming84", "golay"]) -> int:
    """Get number of bits in a codeword for the given codec."""
    if codec == "hamming74":
        return 7
    elif codec == "hamming84":
        return 8
    elif codec == "golay":
        return 24
    else:
        raise ValueError(f"Unknown codec: {codec}")


def get_data_bits(codec: Literal["hamming74", "hamming84", "golay"]) -> int:
    """Get number of data bits encoded by the codec."""
    if codec == "hamming74":
        return 4  # INT4
    elif codec == "hamming84":
        return 4  # INT4
    elif codec == "golay":
        return 12  # 3 x INT4
    else:
        raise ValueError(f"Unknown codec: {codec}")


# =============================================================================
# Syndrome Lookup Tables
# =============================================================================

# Hamming(7,4) syndrome-to-position LUT (8 entries)
# Maps 3-bit syndrome to error position in 7-bit codeword
# -1 indicates no error (syndrome=0)
# Note: Hamming(7,4) cannot detect double-bit errors (will miscorrect them)
SYNDROME_LUT_HAMMING74 = torch.tensor([
    -1,  # syndrome=0: no error
     4,  # syndrome=1: error in bit 4 (p₀)
     5,  # syndrome=2: error in bit 5 (p₁)
     0,  # syndrome=3: error in bit 0 (d₀)
     6,  # syndrome=4: error in bit 6 (p₂)
     1,  # syndrome=5: error in bit 1 (d₁)
     2,  # syndrome=6: error in bit 2 (d₂)
     3,  # syndrome=7: error in bit 3 (d₃)
], dtype=torch.int8)

# Hamming(8,4) syndrome-to-position LUT (8 entries)
# Maps 3-bit syndrome to error position in 7-bit Hamming portion
# -1 indicates no error in bits 0-6 (may be in parity bit 7)
SYNDROME_LUT_HAMMING84 = torch.tensor([
    -1,  # syndrome=0: no error in bits 0-6
     4,  # syndrome=1: error in bit 4 (p₀)
     5,  # syndrome=2: error in bit 5 (p₁)
     0,  # syndrome=3: error in bit 0 (d₀)
     6,  # syndrome=4: error in bit 6 (p₂)
     1,  # syndrome=5: error in bit 1 (d₁)
     2,  # syndrome=6: error in bit 2 (d₂)
     3,  # syndrome=7: error in bit 3 (d₃)
], dtype=torch.int8)


# =============================================================================
# Golay(24,12) Generator Matrix Parity Portion
# =============================================================================

# G = [I₁₂ | P] where P is the 12x12 parity matrix
# For encoding: codeword = [data | data @ P]
# We only need P since the identity portion is trivial

# Golay parity matrix P (12x12) - each row defines parity for one parity bit
# Represented as 12-bit masks (one per row)
GOLAY_PARITY_MASKS = torch.tensor([
    0b110111000111,  # p0
    0b101110001110,  # p1
    0b011100011101,  # p2
    0b111000111010,  # p3
    0b110001110101,  # p4
    0b100011101011,  # p5
    0b000111010111,  # p6
    0b001110101110,  # p7
    0b011101011100,  # p8
    0b111010111000,  # p9
    0b110101110001,  # p10
    0b101011100011,  # p11
], dtype=torch.int32)


# =============================================================================
# Error Type Constants (matching CPU implementation)
# =============================================================================

class ErrorType:
    """Error classification for SECDED decoding."""
    NO_ERROR = 0           # Clean codeword
    SINGLE_CORRECTED = 1   # Single-bit error, successfully corrected
    DOUBLE_DETECTED = 2    # Double-bit error, detected but NOT corrected
    PARITY_ONLY = 3        # Error only in overall parity bit
