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
    lut_ptr,  # Pointer to 8-element syndrome LUT
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

    # Direct LUT lookup - replaces 8-level nested tl.where()
    error_pos = tl.load(lut_ptr + syndrome, mask=mask, other=-1)

    should_correct = (error_type == 1) & (error_pos >= 0)
    correction_mask = tl.where(should_correct, 1 << error_pos, 0).to(tl.uint8)

    corrected = hamming7 ^ correction_mask

    corrected = tl.where(error_type == 2, 0, corrected).to(tl.uint8)

    decoded = corrected & 0x0F

    tl.store(decoded_ptr + offsets, decoded, mask=mask)
    tl.store(error_type_ptr + offsets, error_type, mask=mask)


def hamming84_encode(int4_values):
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


# Cache for GPU LUT to avoid repeated transfers
_syndrome_lut_cache = {}


def _get_syndrome_lut_gpu(device):
    """Get syndrome LUT on the specified GPU device (cached)."""
    if device not in _syndrome_lut_cache:
        _syndrome_lut_cache[device] = SYNDROME_LUT_HAMMING84.to(device)
    return _syndrome_lut_cache[device]


def hamming84_decode(codewords, return_error_types=False):
    assert codewords.is_cuda, "Input must be on CUDA device"

    original_shape = codewords.shape
    flat = codewords.flatten().to(torch.uint8)
    N = flat.numel()

    decoded = torch.empty_like(flat)
    error_types = torch.empty_like(flat)

    # Get LUT on GPU (cached)
    lut_gpu = _get_syndrome_lut_gpu(codewords.device)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    hamming84_decode_kernel[grid](
        flat,
        decoded,
        error_types,
        lut_gpu,
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


class Hamming84:
    """
    Hamming(8,4) SECDED codec wrapper using Triton GPU kernels.

    Provides the same interface as the original CPU implementation.
    Single Error Correct, Double Error Detect (SECDED).
    """

    # Class attributes for verification
    G_74 = HAMMING84_G
    H_74 = HAMMING84_H

    SYNDROME_TO_POSITION = SYNDROME_LUT_HAMMING84

    def __init__(self, device="cuda", on_double_error="zero"):
        """
        Initialize Hamming(8,4) codec.

        Args:
            device: Target device for tensors
            on_double_error: How to handle double errors ("zero" or "raise")
        """
        self.device = device
        self.on_double_error = on_double_error
        self._G = self.G_74.to(device)
        self._H = self.H_74.to(device)
        self._syndrome_lut = self.SYNDROME_TO_POSITION.to(device)

    def encode(self, int4_values: torch.Tensor) -> torch.Tensor:
        """Encode 4-bit values to 8-bit Hamming SECDED codewords."""
        input_tensor = int4_values.to(self.device)
        return hamming84_encode(input_tensor)

    def decode(self, codewords: torch.Tensor) -> DecodeResult:
        """
        Decode 8-bit Hamming SECDED codewords to 4-bit values.

        Returns:
            DecodeResult: NamedTuple with (data, error_type, corrected_count, detected_count)
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
