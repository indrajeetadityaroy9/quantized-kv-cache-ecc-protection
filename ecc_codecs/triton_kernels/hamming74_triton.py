"""
Hamming(7,4) SEC Codec: GPU-accelerated single-error-correcting code.

This module implements Hamming(7,4) encoding and decoding using Triton GPU kernels.
Unlike Hamming(8,4) SECDED, this code only corrects single errors; it cannot
distinguish single errors from double errors.

Encoding: 4 data bits → 7 codeword bits
    codeword = [d0, d1, d2, d3, p0, p1, p2]
    where p_i = XOR of subset of data bits (defined by G matrix)

Decoding: Uses 3-bit syndrome to locate and correct single-bit errors
    - syndrome=0 → no error
    - syndrome≠0 → error at position indicated by syndrome lookup

Comparison with Hamming(8,4):
    - Hamming(7,4): 7 bits/codeword, 57% overhead, SEC only
    - Hamming(8,4): 8 bits/codeword, 100% overhead, SECDED
    Use Hamming(7,4) when storage is critical and double errors are rare.

Pipeline role:
    Called by kv_cache/ecc_shim.py for "int4-hamming" mode.
    Each INT4 KV cache value is protected by one 7-bit Hamming codeword
    (stored in uint8 with bit 7 unused).

Determinism:
    Fully deterministic given identical input tensors. No RNG involved.
"""

import torch
import triton
import triton.language as tl

from .config import (
    HAMMING74_BLOCK_SIZE,
    SYNDROME_LUT_HAMMING74,
    HAMMING74_G,
    HAMMING74_H,
)


# =============================================================================
# Hamming(7,4) Encode Kernel
# =============================================================================


@triton.jit
def hamming74_encode_kernel(
    int4_ptr,      # Input: pointer to INT4 values (uint8, only low 4 bits used)
    codeword_ptr,  # Output: pointer to 7-bit codewords (stored in uint8, bit 7 unused)
    N,             # Total number of elements to encode
    BLOCK_SIZE: tl.constexpr,  # Elements per thread block (1024)
):
    """
    Triton kernel: Encode INT4 values to Hamming(7,4) codewords.

    Bit layout of output codeword (stored in uint8):
        [d0, d1, d2, d3, p0, p1, p2, 0]
         └─ data ──────┘  └─ parity ┘ └─ unused
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load INT4 values (only low 4 bits are meaningful)
    int4_vals = tl.load(int4_ptr + offsets, mask=mask, other=0).to(tl.uint8)

    # Extract individual data bits
    d0 = (int4_vals >> 0) & 1
    d1 = (int4_vals >> 1) & 1
    d2 = (int4_vals >> 2) & 1
    d3 = (int4_vals >> 3) & 1

    # Compute parity bits according to G matrix
    # Same parity equations as Hamming(8,4), without overall parity
    p0 = d0 ^ d1 ^ d3
    p1 = d0 ^ d2 ^ d3
    p2 = d1 ^ d2 ^ d3

    # Assemble 7-bit codeword (bit 7 remains 0)
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


# =============================================================================
# Hamming(7,4) Decode Kernel
# =============================================================================


@triton.jit
def hamming74_decode_kernel(
    codeword_ptr,        # Input: pointer to 7-bit codewords (uint8)
    decoded_ptr,         # Output: pointer to decoded INT4 values (uint8)
    error_detected_ptr,  # Output: pointer to error detection flags (uint8: 0 or 1)
    lut_ptr,             # Syndrome LUT: 8-entry table mapping syndrome → bit position
    N,                   # Total number of codewords to decode
    BLOCK_SIZE: tl.constexpr,  # Elements per thread block (1024)
):
    """
    Triton kernel: Decode Hamming(7,4) codewords to INT4 values.

    SEC (Single Error Correct) algorithm:
        1. Compute 3-bit syndrome from received codeword
        2. If syndrome ≠ 0, flip bit at syndrome-indicated position
        3. Extract data bits from corrected codeword

    LIMITATION: Cannot distinguish single from double errors.
    A double-bit error will produce a non-zero syndrome and "correct"
    the wrong bit, potentially introducing a third error.
    Use Hamming(8,4) SECDED if double-error detection is needed.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load 7-bit codewords (stored in uint8)
    codewords = tl.load(codeword_ptr + offsets, mask=mask, other=0).to(tl.uint8)

    # Extract individual bits for syndrome computation
    c0 = (codewords >> 0) & 1  # d0
    c1 = (codewords >> 1) & 1  # d1
    c2 = (codewords >> 2) & 1  # d2
    c3 = (codewords >> 3) & 1  # d3
    c4 = (codewords >> 4) & 1  # p0
    c5 = (codewords >> 5) & 1  # p1
    c6 = (codewords >> 6) & 1  # p2

    # Compute syndrome bits (H matrix rows applied to codeword)
    s0 = c0 ^ c1 ^ c3 ^ c4  # Row 0 of H
    s1 = c0 ^ c2 ^ c3 ^ c5  # Row 1 of H
    s2 = c1 ^ c2 ^ c3 ^ c6  # Row 2 of H

    # Assemble 3-bit syndrome
    syndrome = (s0 | (s1 << 1) | (s2 << 2)).to(tl.int32)

    # Flag indicating whether any error was detected
    error_detected = (syndrome != 0).to(tl.uint8)

    # O(1) syndrome lookup: maps 3-bit syndrome to error bit position
    error_pos = tl.load(lut_ptr + syndrome, mask=mask, other=-1)

    # Correct if syndrome indicates valid error position
    should_correct = error_pos >= 0
    correction_mask = tl.where(should_correct, 1 << error_pos, 0).to(tl.uint8)

    # XOR correction: flips the erroneous bit
    corrected = codewords ^ correction_mask

    # Extract data bits (low 4 bits)
    decoded = corrected & 0x0F

    tl.store(decoded_ptr + offsets, decoded, mask=mask)
    tl.store(error_detected_ptr + offsets, error_detected, mask=mask)


# =============================================================================
# Python Wrapper Functions
# =============================================================================


def hamming74_encode(int4_values: torch.Tensor) -> torch.Tensor:
    """
    Encode INT4 values to Hamming(7,4) codewords.

    Args:
        int4_values: Tensor of INT4 values (uint8, only low 4 bits used).
            Must be on CUDA device. Shape: any (will be flattened internally).

    Returns:
        Tensor of 7-bit codewords (stored in uint8), same shape as input.
        Bit 7 of each byte is unused (always 0).

    Raises:
        AssertionError: If input not on CUDA device.
    """
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


# =============================================================================
# Syndrome LUT GPU Cache
# =============================================================================

_syndrome_lut_cache_74 = {}


def _get_syndrome_lut_gpu_74(device: torch.device) -> torch.Tensor:
    """Get syndrome LUT on the specified GPU device (cached)."""
    if device not in _syndrome_lut_cache_74:
        _syndrome_lut_cache_74[device] = SYNDROME_LUT_HAMMING74.to(device)
    return _syndrome_lut_cache_74[device]


def hamming74_decode(
    codewords: torch.Tensor,
    return_error_detected: bool = False,
) -> tuple:
    """
    Decode Hamming(7,4) codewords to INT4 values.

    Performs single-error correction. Cannot detect double errors.

    Args:
        codewords: Tensor of 7-bit codewords. Must be on CUDA device.
        return_error_detected: If True, also return per-element error flags.

    Returns:
        If return_error_detected=False:
            (decoded, (errors_corrected_count,))
        If return_error_detected=True:
            (decoded, error_detected, (errors_corrected_count,))

        Where:
            - decoded: Tensor of INT4 values, same shape as input
            - error_detected: Tensor of uint8 (0 or 1), same shape as input
            - errors_corrected_count: Total number of corrections performed

    Warning:
        Double-bit errors are silently miscorrected. Use Hamming(8,4) SECDED
        if double-error detection is required.
    """
    assert codewords.is_cuda, "Input must be on CUDA device"

    original_shape = codewords.shape
    flat = codewords.flatten().to(torch.uint8)
    N = flat.numel()

    decoded = torch.empty_like(flat)
    error_detected = torch.empty_like(flat)

    # Get LUT on GPU (cached)
    lut_gpu = _get_syndrome_lut_gpu_74(codewords.device)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    hamming74_decode_kernel[grid](
        flat,
        decoded,
        error_detected,
        lut_gpu,
        N,
        BLOCK_SIZE=HAMMING74_BLOCK_SIZE,
    )

    # Count total corrections (syndrome ≠ 0)
    errors_corrected_count = int(error_detected.sum())

    decoded = decoded.view(original_shape)
    error_detected = error_detected.view(original_shape)

    if return_error_detected:
        return decoded, error_detected, (errors_corrected_count,)
    else:
        return decoded, (errors_corrected_count,)


# =============================================================================
# Hamming(7,4) Codec Class
# =============================================================================


class Hamming74:
    """
    Hamming(7,4) SEC codec with Triton GPU acceleration.

    This class provides an object-oriented interface to the Hamming(7,4) encode/decode
    functions. Unlike Hamming(8,4) SECDED, this code can only correct single errors;
    double errors are silently miscorrected.

    Attributes:
        G: Generator matrix (4×7). Class attribute.
        H: Parity-check matrix (3×7). Class attribute.
        SYNDROME_TO_POSITION: Syndrome lookup table. Class attribute.
        device: Target CUDA device for tensor operations.

    Example:
        >>> codec = Hamming74(device='cuda')
        >>> data = torch.randint(0, 16, (1000,), dtype=torch.uint8, device='cuda')
        >>> codewords = codec.encode(data)
        >>> decoded, errors = codec.decode(codewords)
        >>> assert (decoded == data).all()
    """

    # Class attributes: matrices for verification
    G = HAMMING74_G  # Generator matrix: 4×7
    H = HAMMING74_H  # Parity-check matrix: 3×7, G @ H^T = 0

    SYNDROME_TO_POSITION = SYNDROME_LUT_HAMMING74

    def __init__(self, device: str = "cuda"):
        """
        Initialize Hamming(7,4) codec.

        Args:
            device: Target device for tensors ('cuda', 'cuda:0', etc.)
        """
        self.device = device
        self._G = self.G.to(device)
        self._H = self.H.to(device)
        self._syndrome_lut = self.SYNDROME_TO_POSITION.to(device)

    def encode(self, int4_values: torch.Tensor) -> torch.Tensor:
        """
        Encode 4-bit values to 7-bit Hamming codewords.

        Args:
            int4_values: Tensor of INT4 values (uint8, low 4 bits used).

        Returns:
            Tensor of 7-bit codewords (stored in uint8), same shape.
        """
        input_tensor = int4_values.to(self.device)
        return hamming74_encode(input_tensor)

    def decode(self, codewords: torch.Tensor) -> tuple:
        """
        Decode 7-bit Hamming codewords to 4-bit values.

        Args:
            codewords: Tensor of 7-bit codewords (uint8).

        Returns:
            (decoded_values, errors_detected_bool_tensor)
            - decoded_values: INT4 tensor, same shape as input
            - errors_detected: Boolean tensor, True where corrections applied
        """
        input_tensor = codewords.to(self.device)
        decoded, error_detected, _ = hamming74_decode(
            input_tensor, return_error_detected=True
        )
        # Convert to bool tensor to match original CPU interface
        return decoded, error_detected.bool()

    def encode_batch(self, int4_tensor: torch.Tensor) -> torch.Tensor:
        """Encode batch of 4-bit values. Alias for encode()."""
        return self.encode(int4_tensor)

    def decode_batch(self, codeword_tensor: torch.Tensor) -> tuple:
        """Decode batch of codewords. Alias for decode()."""
        return self.decode(codeword_tensor)
