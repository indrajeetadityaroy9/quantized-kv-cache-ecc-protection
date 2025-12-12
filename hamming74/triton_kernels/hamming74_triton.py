"""
Triton GPU Kernels for Hamming(7,4) SEC.

GPU-native implementation of Hamming(7,4) encoder/decoder.
Single-error correction only (no double-error detection like SECDED).

Key differences from Hamming(8,4):
- 7-bit codeword (no overall parity bit)
- Returns boolean error_detected, not ErrorType enum
- May miscorrect double-bit errors (cannot detect them)

Storage: torch.uint8 (7 bits used per INT4 value, MSB always 0)
"""

import torch
import triton
import triton.language as tl
from typing import Tuple

from .config import (
    HAMMING74_BLOCK_SIZE,
    SYNDROME_LUT_HAMMING74,
)


# =============================================================================
# Triton Encode Kernel
# =============================================================================

@triton.jit
def hamming74_encode_kernel(
    # Pointers
    int4_ptr,        # Input: INT4 values [N], dtype=uint8
    codeword_ptr,    # Output: 7-bit codewords [N], dtype=uint8
    # Dimensions
    N,               # Total number of values
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Encode INT4 values to Hamming(7,4) codewords on GPU.

    Generator matrix encoding (systematic form G = [I₄ | P]):
    - data_bits = [d₀, d₁, d₂, d₃] (4 bits from INT4)
    - parity_bits:
        p₀ = d₀ ⊕ d₁ ⊕ d₃
        p₁ = d₀ ⊕ d₂ ⊕ d₃
        p₂ = d₁ ⊕ d₂ ⊕ d₃
    - codeword = [d₀, d₁, d₂, d₃, p₀, p₁, p₂] (7 bits, MSB unused)
    """
    # Program ID and offsets
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load INT4 values (0-15)
    int4_vals = tl.load(int4_ptr + offsets, mask=mask, other=0).to(tl.uint8)

    # Extract data bits using bitwise AND and shift
    d0 = (int4_vals >> 0) & 1
    d1 = (int4_vals >> 1) & 1
    d2 = (int4_vals >> 2) & 1
    d3 = (int4_vals >> 3) & 1

    # Compute parity bits (XOR operations native in Triton)
    # Based on generator matrix G = [I₄ | P] where:
    # P = [[1,1,0], [1,0,1], [0,1,1], [1,1,1]]
    p0 = d0 ^ d1 ^ d3       # Row 0,1,3 of P column 0
    p1 = d0 ^ d2 ^ d3       # Row 0,2,3 of P column 1
    p2 = d1 ^ d2 ^ d3       # Row 1,2,3 of P column 2

    # Pack into 7-bit codeword: [d0,d1,d2,d3,p0,p1,p2]
    codeword = (
        (d0 << 0) |
        (d1 << 1) |
        (d2 << 2) |
        (d3 << 3) |
        (p0 << 4) |
        (p1 << 5) |
        (p2 << 6)
    ).to(tl.uint8)

    # Store result
    tl.store(codeword_ptr + offsets, codeword, mask=mask)


# =============================================================================
# Triton Decode Kernel
# =============================================================================

@triton.jit
def hamming74_decode_kernel(
    # Pointers
    codeword_ptr,         # Input: 7-bit codewords [N], dtype=uint8
    decoded_ptr,          # Output: corrected INT4 values [N], dtype=uint8
    error_detected_ptr,   # Output: error detected flags [N], dtype=uint8 (0 or 1)
    # Syndrome LUT (8 entries, passed as individual values for register storage)
    lut0: tl.constexpr, lut1: tl.constexpr, lut2: tl.constexpr, lut3: tl.constexpr,
    lut4: tl.constexpr, lut5: tl.constexpr, lut6: tl.constexpr, lut7: tl.constexpr,
    # Dimensions
    N,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Decode Hamming(7,4) codewords with SEC on GPU.

    SEC Logic (no SECDED - cannot detect double errors):
    - syndrome=0: No error (or undetectable multi-bit error)
    - syndrome≠0: Single-bit error (or miscorrected multi-bit error)

    Syndrome computation uses parity check matrix H:
    H = [[1,1,0,1,1,0,0],   -> s₀
         [1,0,1,1,0,1,0],   -> s₁
         [0,1,1,1,0,0,1]]   -> s₂

    WARNING: Double-bit errors will be miscorrected (appears as different single-bit error)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load codewords (only 7 bits used)
    codewords = tl.load(codeword_ptr + offsets, mask=mask, other=0).to(tl.uint8)

    # Extract individual bits for syndrome computation
    c0 = (codewords >> 0) & 1
    c1 = (codewords >> 1) & 1
    c2 = (codewords >> 2) & 1
    c3 = (codewords >> 3) & 1
    c4 = (codewords >> 4) & 1
    c5 = (codewords >> 5) & 1
    c6 = (codewords >> 6) & 1

    # Compute syndrome: z = H @ r^T (mod 2)
    # H row 0: [1,1,0,1,1,0,0] -> s0 = c0 ^ c1 ^ c3 ^ c4
    # H row 1: [1,0,1,1,0,1,0] -> s1 = c0 ^ c2 ^ c3 ^ c5
    # H row 2: [0,1,1,1,0,0,1] -> s2 = c1 ^ c2 ^ c3 ^ c6
    s0 = c0 ^ c1 ^ c3 ^ c4
    s1 = c0 ^ c2 ^ c3 ^ c5
    s2 = c1 ^ c2 ^ c3 ^ c6

    # Pack syndrome into 3-bit index
    syndrome = (s0 | (s1 << 1) | (s2 << 2)).to(tl.int32)

    # Error detected flag (syndrome != 0)
    error_detected = (syndrome != 0).to(tl.uint8)

    # Lookup error position from syndrome (inline LUT to keep in registers)
    # LUT: syndrome -> bit position to flip (-1 means no flip needed)
    error_pos = tl.where(syndrome == 0, lut0,
                tl.where(syndrome == 1, lut1,
                tl.where(syndrome == 2, lut2,
                tl.where(syndrome == 3, lut3,
                tl.where(syndrome == 4, lut4,
                tl.where(syndrome == 5, lut5,
                tl.where(syndrome == 6, lut6,
                         lut7)))))))

    # Compute correction mask (only if error_pos >= 0)
    should_correct = (error_pos >= 0)
    correction_mask = tl.where(should_correct, 1 << error_pos, 0).to(tl.uint8)

    # Apply correction via XOR
    corrected = codewords ^ correction_mask

    # Extract data bits (first 4 bits in systematic form)
    decoded = corrected & 0x0F

    # Store results
    tl.store(decoded_ptr + offsets, decoded, mask=mask)
    tl.store(error_detected_ptr + offsets, error_detected, mask=mask)


# =============================================================================
# Python Wrapper Functions
# =============================================================================

def hamming74_encode(int4_values: torch.Tensor) -> torch.Tensor:
    """
    Encode INT4 values to Hamming(7,4) codewords using Triton kernel.

    Args:
        int4_values: Tensor of INT4 values (0-15), any shape, on CUDA

    Returns:
        codewords: Tensor of 7-bit codewords (uint8), same shape
                   (Only lower 7 bits are used, MSB is always 0)
    """
    assert int4_values.is_cuda, "Input must be on CUDA device"

    original_shape = int4_values.shape
    flat = int4_values.flatten().to(torch.uint8)
    N = flat.numel()

    # Allocate output
    codewords = torch.empty_like(flat)

    # Launch kernel
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    hamming74_encode_kernel[grid](
        flat, codewords, N,
        BLOCK_SIZE=HAMMING74_BLOCK_SIZE,
    )

    return codewords.view(original_shape)


def hamming74_decode(
    codewords: torch.Tensor,
    return_error_detected: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """
    Decode Hamming(7,4) codewords with SEC using Triton kernel.

    Args:
        codewords: Tensor of 7-bit codewords (uint8), any shape, on CUDA
        return_error_detected: If True, return error_detected tensor

    Returns:
        decoded: Corrected INT4 values
        error_detected: (optional) Boolean tensor (as uint8) for each value
        stats: Tuple of (errors_corrected_count,)

    Note:
        Unlike Hamming(8,4) SECDED, Hamming(7,4) cannot detect double-bit errors.
        Double-bit errors will be miscorrected (silently producing wrong values).
    """
    assert codewords.is_cuda, "Input must be on CUDA device"

    original_shape = codewords.shape
    flat = codewords.flatten().to(torch.uint8)
    N = flat.numel()

    # Allocate outputs
    decoded = torch.empty_like(flat)
    error_detected = torch.empty_like(flat)

    # Syndrome LUT values (passed as constexpr for register storage)
    lut = SYNDROME_LUT_HAMMING74

    # Launch kernel
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    hamming74_decode_kernel[grid](
        flat, decoded, error_detected,
        int(lut[0]), int(lut[1]), int(lut[2]), int(lut[3]),
        int(lut[4]), int(lut[5]), int(lut[6]), int(lut[7]),
        N,
        BLOCK_SIZE=HAMMING74_BLOCK_SIZE,
    )

    # Compute statistics
    errors_corrected_count = int(error_detected.sum())

    decoded = decoded.view(original_shape)
    error_detected = error_detected.view(original_shape)

    if return_error_detected:
        return decoded, error_detected, (errors_corrected_count,)
    else:
        return decoded, (errors_corrected_count,)


# =============================================================================
# Verification
# =============================================================================

def verify_triton_vs_cpu():
    """
    Verify Triton kernels match CPU reference implementation.

    Tests:
    1. Encode produces identical codewords
    2. Decode corrects single-bit errors
    3. Large tensor roundtrip
    """
    from ..hamming74_sec import Hamming74

    device = "cuda"
    cpu_codec = Hamming74(device="cpu")

    print("Verifying Triton Hamming(7,4) vs CPU reference...")

    # Test 1: Encode all 16 possible INT4 values
    all_int4 = torch.arange(16, dtype=torch.uint8)

    # CPU encode
    cpu_codewords = cpu_codec.encode(all_int4)

    # Triton encode
    triton_codewords = hamming74_encode(all_int4.to(device))

    assert torch.equal(cpu_codewords, triton_codewords.cpu()), \
        "Triton encode does not match CPU!"
    print("  [PASS] Encode matches CPU reference")

    # Test 2: Decode clean codewords
    triton_decoded, stats = hamming74_decode(triton_codewords)
    assert torch.equal(all_int4.to(device), triton_decoded), \
        "Clean decode failed!"
    assert stats[0] == 0, "Should have no errors"
    print("  [PASS] Clean decode works")

    # Test 3: Single-bit error correction (all 7 positions)
    test_val = torch.tensor([7], dtype=torch.uint8, device=device)
    codeword = hamming74_encode(test_val)

    for bit_pos in range(7):
        corrupted = codeword ^ (1 << bit_pos)
        decoded, error_detected, stats = hamming74_decode(corrupted, return_error_detected=True)
        assert decoded.item() == 7, f"Single-bit error at bit {bit_pos} not corrected"
        assert error_detected.item() == 1, f"Error at bit {bit_pos} not detected"

    print("  [PASS] Single-bit error correction works (all 7 positions)")

    # Test 4: Large random tensor roundtrip
    large_input = torch.randint(0, 16, (100000,), dtype=torch.uint8, device=device)
    encoded = hamming74_encode(large_input)
    decoded, stats = hamming74_decode(encoded)
    assert torch.equal(large_input, decoded), "Large tensor roundtrip failed!"
    assert stats[0] == 0, "Clean data should have no corrections"
    print("  [PASS] Large tensor roundtrip works (100K values)")

    # Test 5: Verify error correction statistics
    # Inject single-bit errors into half the codewords
    encoded_with_errors = encoded.clone()
    error_mask = torch.zeros(100000, dtype=torch.bool, device=device)
    error_mask[::2] = True  # Every other codeword
    bit_positions = torch.randint(0, 7, (50000,), device=device)
    error_pattern = (1 << bit_positions).to(torch.uint8)
    encoded_with_errors[::2] ^= error_pattern

    decoded, stats = hamming74_decode(encoded_with_errors)
    expected_errors = 50000
    assert stats[0] == expected_errors, f"Expected {expected_errors} corrections, got {stats[0]}"
    assert torch.equal(decoded, large_input), "Corrected data doesn't match original!"
    print(f"  [PASS] Error correction statistics correct ({stats[0]} errors corrected)")

    print("All Triton Hamming(7,4) verifications passed!")
    return True


if __name__ == "__main__":
    if torch.cuda.is_available():
        verify_triton_vs_cpu()
    else:
        print("CUDA not available, skipping Triton verification")
