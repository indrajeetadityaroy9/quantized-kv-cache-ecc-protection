import torch
import triton
import triton.language as tl

from .config import (
    HAMMING74_BLOCK_SIZE,
    SYNDROME_LUT_HAMMING74,
    HAMMING74_G,
    HAMMING74_H,
)


@triton.jit
def hamming74_encode_kernel(
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


@triton.jit
def hamming74_decode_kernel(
    codeword_ptr,
    decoded_ptr,
    error_detected_ptr,
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

    c0 = (codewords >> 0) & 1
    c1 = (codewords >> 1) & 1
    c2 = (codewords >> 2) & 1
    c3 = (codewords >> 3) & 1
    c4 = (codewords >> 4) & 1
    c5 = (codewords >> 5) & 1
    c6 = (codewords >> 6) & 1

    s0 = c0 ^ c1 ^ c3 ^ c4
    s1 = c0 ^ c2 ^ c3 ^ c5
    s2 = c1 ^ c2 ^ c3 ^ c6

    syndrome = (s0 | (s1 << 1) | (s2 << 2)).to(tl.int32)

    error_detected = (syndrome != 0).to(tl.uint8)

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

    should_correct = error_pos >= 0
    correction_mask = tl.where(should_correct, 1 << error_pos, 0).to(tl.uint8)

    corrected = codewords ^ correction_mask

    decoded = corrected & 0x0F

    tl.store(decoded_ptr + offsets, decoded, mask=mask)
    tl.store(error_detected_ptr + offsets, error_detected, mask=mask)


def hamming74_encode(int4_values):
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


def hamming74_decode(codewords, return_error_detected=False):
    assert codewords.is_cuda, "Input must be on CUDA device"

    original_shape = codewords.shape
    flat = codewords.flatten().to(torch.uint8)
    N = flat.numel()

    decoded = torch.empty_like(flat)
    error_detected = torch.empty_like(flat)

    lut = SYNDROME_LUT_HAMMING74

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    hamming74_decode_kernel[grid](
        flat,
        decoded,
        error_detected,
        int(lut[0]),
        int(lut[1]),
        int(lut[2]),
        int(lut[3]),
        int(lut[4]),
        int(lut[5]),
        int(lut[6]),
        int(lut[7]),
        N,
        BLOCK_SIZE=HAMMING74_BLOCK_SIZE,
    )

    errors_corrected_count = int(error_detected.sum())

    decoded = decoded.view(original_shape)
    error_detected = error_detected.view(original_shape)

    if return_error_detected:
        return decoded, error_detected, (errors_corrected_count,)
    else:
        return decoded, (errors_corrected_count,)


class Hamming74:
    """
    Hamming(7,4) codec wrapper using Triton GPU kernels.

    Provides the same interface as the original CPU implementation.
    """

    # Class attributes for verification (accessible as Hamming74.G, Hamming74.H)
    G = HAMMING74_G
    H = HAMMING74_H

    SYNDROME_TO_POSITION = SYNDROME_LUT_HAMMING74

    def __init__(self, device="cuda"):
        self.device = device
        self._G = self.G.to(device)
        self._H = self.H.to(device)
        self._syndrome_lut = self.SYNDROME_TO_POSITION.to(device)

    def encode(self, int4_values: torch.Tensor) -> torch.Tensor:
        """Encode 4-bit values to 7-bit Hamming codewords."""
        input_tensor = int4_values.to(self.device)
        return hamming74_encode(input_tensor)

    def decode(self, codewords: torch.Tensor):
        """
        Decode 7-bit Hamming codewords to 4-bit values.

        Returns:
            tuple: (decoded_values, errors_detected_bool_tensor)
        """
        input_tensor = codewords.to(self.device)
        decoded, error_detected, _ = hamming74_decode(
            input_tensor, return_error_detected=True
        )
        # Convert to bool tensor to match original CPU interface
        return decoded, error_detected.bool()

    def encode_batch(self, int4_tensor: torch.Tensor) -> torch.Tensor:
        """Encode batch of 4-bit values."""
        return self.encode(int4_tensor)

    def decode_batch(self, codeword_tensor: torch.Tensor):
        """Decode batch of codewords."""
        return self.decode(codeword_tensor)
