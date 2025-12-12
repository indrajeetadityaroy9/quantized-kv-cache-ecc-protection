"""
Triton GPU Kernels for Hamming(8,4) SECDED.

GPU-native implementation of Extended Hamming(8,4) encoder/decoder.
Eliminates CPU-GPU transfer overhead of the reference implementation.

Key optimizations:
- All operations in registers (no intermediate stores)
- Vectorized bit manipulation using native Triton ops
- 8-entry syndrome LUT fits in registers (no shared memory needed)

Storage: torch.uint8 (8 bits per INT4 value)
"""

import torch
import triton
import triton.language as tl
from typing import Tuple

from .config import (
    HAMMING84_BLOCK_SIZE,
    SYNDROME_LUT_HAMMING84,
    ErrorType,
)


# =============================================================================
# Triton Encode Kernel
# =============================================================================

@triton.jit
def hamming84_encode_kernel(
    # Pointers
    int4_ptr,        # Input: INT4 values [N], dtype=uint8
    codeword_ptr,    # Output: 8-bit codewords [N], dtype=uint8
    # Dimensions
    N,               # Total number of values
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Encode INT4 values to Hamming(8,4) codewords on GPU.

    Generator matrix encoding (systematic form G = [I₄ | P]):
    - data_bits = [d₀, d₁, d₂, d₃] (4 bits from INT4)
    - parity_bits:
        p₀ = d₀ ⊕ d₁ ⊕ d₃
        p₁ = d₀ ⊕ d₂ ⊕ d₃
        p₂ = d₁ ⊕ d₂ ⊕ d₃
    - overall_parity = ⊕(d₀, d₁, d₂, d₃, p₀, p₁, p₂)
    - codeword = [d₀, d₁, d₂, d₃, p₀, p₁, p₂, overall_parity]
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

    # Pack into 7-bit Hamming codeword: [d0,d1,d2,d3,p0,p1,p2]
    hamming7 = (
        (d0 << 0) |
        (d1 << 1) |
        (d2 << 2) |
        (d3 << 3) |
        (p0 << 4) |
        (p1 << 5) |
        (p2 << 6)
    )

    # Compute overall parity (XOR of all 7 bits) via bit-folding
    # This is popcount(hamming7) mod 2
    parity = hamming7 ^ (hamming7 >> 4)  # Fold 4 bits
    parity = parity ^ (parity >> 2)       # Fold 2 bits
    parity = parity ^ (parity >> 1)       # Fold 1 bit
    overall_parity = parity & 1

    # Final 8-bit codeword: [hamming7 | overall_parity << 7]
    codeword = (hamming7 | (overall_parity << 7)).to(tl.uint8)

    # Store result
    tl.store(codeword_ptr + offsets, codeword, mask=mask)


# =============================================================================
# Triton Decode Kernel
# =============================================================================

@triton.jit
def hamming84_decode_kernel(
    # Pointers
    codeword_ptr,         # Input: 8-bit codewords [N], dtype=uint8
    decoded_ptr,          # Output: corrected INT4 values [N], dtype=uint8
    error_type_ptr,       # Output: error classification [N], dtype=uint8
    # Syndrome LUT (8 entries, passed as individual values for register storage)
    lut0: tl.constexpr, lut1: tl.constexpr, lut2: tl.constexpr, lut3: tl.constexpr,
    lut4: tl.constexpr, lut5: tl.constexpr, lut6: tl.constexpr, lut7: tl.constexpr,
    # Dimensions
    N,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Decode Hamming(8,4) codewords with SECDED on GPU.

    SECDED Logic:
    - syndrome=0, parity=0: No error (NO_ERROR=0)
    - syndrome=0, parity=1: Error in overall parity bit only (PARITY_ONLY=3)
    - syndrome≠0, parity=1: Single-bit error, correctable (SINGLE_CORRECTED=1)
    - syndrome≠0, parity=0: Double-bit error, detected only (DOUBLE_DETECTED=2)

    Syndrome computation uses parity check matrix H:
    H = [[1,1,0,1,1,0,0],   -> s₀
         [1,0,1,1,0,1,0],   -> s₁
         [0,1,1,1,0,0,1]]   -> s₂
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load codewords
    codewords = tl.load(codeword_ptr + offsets, mask=mask, other=0).to(tl.uint8)

    # Extract Hamming(7,4) portion (bits 0-6) and overall parity (bit 7)
    hamming7 = codewords & 0x7F
    stored_parity = (codewords >> 7) & 1

    # Extract individual bits for syndrome computation
    c0 = (hamming7 >> 0) & 1
    c1 = (hamming7 >> 1) & 1
    c2 = (hamming7 >> 2) & 1
    c3 = (hamming7 >> 3) & 1
    c4 = (hamming7 >> 4) & 1
    c5 = (hamming7 >> 5) & 1
    c6 = (hamming7 >> 6) & 1

    # Compute syndrome: z = H @ r^T (mod 2)
    # H row 0: [1,1,0,1,1,0,0] -> s0 = c0 ^ c1 ^ c3 ^ c4
    # H row 1: [1,0,1,1,0,1,0] -> s1 = c0 ^ c2 ^ c3 ^ c5
    # H row 2: [0,1,1,1,0,0,1] -> s2 = c1 ^ c2 ^ c3 ^ c6
    s0 = c0 ^ c1 ^ c3 ^ c4
    s1 = c0 ^ c2 ^ c3 ^ c5
    s2 = c1 ^ c2 ^ c3 ^ c6

    # Pack syndrome into 3-bit index
    syndrome = (s0 | (s1 << 1) | (s2 << 2)).to(tl.int32)

    # Compute actual parity of received 7 bits (via bit-folding XOR)
    actual_parity = hamming7 ^ (hamming7 >> 4)
    actual_parity = actual_parity ^ (actual_parity >> 2)
    actual_parity = actual_parity ^ (actual_parity >> 1)
    actual_parity = actual_parity & 1

    # Parity error flag
    parity_error = (stored_parity != actual_parity)

    # SECDED classification
    syndrome_zero = (syndrome == 0)

    # ErrorType: 0=NO_ERROR, 1=SINGLE_CORRECTED, 2=DOUBLE_DETECTED, 3=PARITY_ONLY
    # syndrome=0, parity_ok -> NO_ERROR (0)
    # syndrome=0, parity_err -> PARITY_ONLY (3)
    # syndrome≠0, parity_err -> SINGLE_CORRECTED (1)
    # syndrome≠0, parity_ok -> DOUBLE_DETECTED (2)
    error_type = tl.where(
        syndrome_zero,
        tl.where(parity_error, 3, 0),  # PARITY_ONLY or NO_ERROR
        tl.where(parity_error, 1, 2)   # SINGLE_CORRECTED or DOUBLE_DETECTED
    ).to(tl.uint8)

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

    # Compute correction mask (only for single-bit errors)
    # Only correct if error_type == SINGLE_CORRECTED (1) and error_pos >= 0
    should_correct = (error_type == 1) & (error_pos >= 0)
    correction_mask = tl.where(should_correct, 1 << error_pos, 0).to(tl.uint8)

    # Apply correction via XOR
    corrected = hamming7 ^ correction_mask

    # Handle double errors: zero out (policy: on_double_error="zero")
    corrected = tl.where(error_type == 2, 0, corrected).to(tl.uint8)

    # Extract data bits (first 4 bits in systematic form)
    decoded = corrected & 0x0F

    # Store results
    tl.store(decoded_ptr + offsets, decoded, mask=mask)
    tl.store(error_type_ptr + offsets, error_type, mask=mask)


# =============================================================================
# Python Wrapper Functions
# =============================================================================

def hamming84_encode(int4_values: torch.Tensor) -> torch.Tensor:
    """
    Encode INT4 values to Hamming(8,4) codewords using Triton kernel.

    Args:
        int4_values: Tensor of INT4 values (0-15), any shape, on CUDA

    Returns:
        codewords: Tensor of 8-bit codewords (uint8), same shape
    """
    assert int4_values.is_cuda, "Input must be on CUDA device"

    original_shape = int4_values.shape
    flat = int4_values.flatten().to(torch.uint8)
    N = flat.numel()

    # Allocate output
    codewords = torch.empty_like(flat)

    # Launch kernel
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    hamming84_encode_kernel[grid](
        flat, codewords, N,
        BLOCK_SIZE=HAMMING84_BLOCK_SIZE,
    )

    return codewords.view(original_shape)


def hamming84_decode(
    codewords: torch.Tensor,
    return_error_types: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """
    Decode Hamming(8,4) codewords with SECDED using Triton kernel.

    Args:
        codewords: Tensor of 8-bit codewords (uint8), any shape, on CUDA
        return_error_types: If True, return error type tensor

    Returns:
        decoded: Corrected INT4 values
        error_types: (optional) ErrorType classification for each value
        stats: Tuple of (corrected_count, detected_count)
    """
    assert codewords.is_cuda, "Input must be on CUDA device"

    original_shape = codewords.shape
    flat = codewords.flatten().to(torch.uint8)
    N = flat.numel()

    # Allocate outputs
    decoded = torch.empty_like(flat)
    error_types = torch.empty_like(flat)

    # Syndrome LUT values (passed as constexpr for register storage)
    lut = SYNDROME_LUT_HAMMING84

    # Launch kernel
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    hamming84_decode_kernel[grid](
        flat, decoded, error_types,
        int(lut[0]), int(lut[1]), int(lut[2]), int(lut[3]),
        int(lut[4]), int(lut[5]), int(lut[6]), int(lut[7]),
        N,
        BLOCK_SIZE=HAMMING84_BLOCK_SIZE,
    )

    # Compute statistics
    error_types_flat = error_types
    corrected_count = int((error_types_flat == ErrorType.SINGLE_CORRECTED).sum())
    detected_count = int((error_types_flat == ErrorType.DOUBLE_DETECTED).sum())

    decoded = decoded.view(original_shape)
    error_types = error_types.view(original_shape)

    if return_error_types:
        return decoded, error_types, (corrected_count, detected_count)
    else:
        return decoded, (corrected_count, detected_count)


# =============================================================================
# Verification
# =============================================================================

def verify_triton_vs_cpu():
    """
    Verify Triton kernels match CPU reference implementation.

    Tests:
    1. Encode produces identical codewords
    2. Decode corrects single-bit errors
    3. Decode detects double-bit errors
    """
    from ..hamming84_secded import Hamming84

    device = "cuda"
    cpu_codec = Hamming84(device="cpu", on_double_error="zero")

    print("Verifying Triton Hamming(8,4) vs CPU reference...")

    # Test 1: Encode all 16 possible INT4 values
    all_int4 = torch.arange(16, dtype=torch.uint8)

    # CPU encode
    cpu_codewords = cpu_codec.encode(all_int4)

    # Triton encode
    triton_codewords = hamming84_encode(all_int4.to(device))

    assert torch.equal(cpu_codewords, triton_codewords.cpu()), \
        "Triton encode does not match CPU!"
    print("  [PASS] Encode matches CPU reference")

    # Test 2: Decode clean codewords
    triton_decoded, stats = hamming84_decode(triton_codewords)
    assert torch.equal(all_int4.to(device), triton_decoded), \
        "Clean decode failed!"
    assert stats[0] == 0 and stats[1] == 0, "Should have no errors"
    print("  [PASS] Clean decode works")

    # Test 3: Single-bit error correction
    test_val = torch.tensor([7], dtype=torch.uint8, device=device)
    codeword = hamming84_encode(test_val)

    for bit_pos in range(8):
        corrupted = codeword ^ (1 << bit_pos)
        decoded, error_types, stats = hamming84_decode(corrupted, return_error_types=True)
        assert decoded.item() == 7, f"Single-bit error at bit {bit_pos} not corrected"

    print("  [PASS] Single-bit error correction works")

    # Test 4: Double-bit error detection
    for bit1 in range(7):
        for bit2 in range(bit1 + 1, 7):
            corrupted = codeword ^ (1 << bit1) ^ (1 << bit2)
            decoded, error_types, stats = hamming84_decode(corrupted, return_error_types=True)
            # Should be detected (zeroed out per policy)
            assert decoded.item() == 0 or error_types.item() == ErrorType.DOUBLE_DETECTED, \
                f"Double-bit error at bits {bit1},{bit2} not handled correctly"

    print("  [PASS] Double-bit error detection works")

    # Test 5: Large random tensor
    large_input = torch.randint(0, 16, (100000,), dtype=torch.uint8, device=device)
    encoded = hamming84_encode(large_input)
    decoded, stats = hamming84_decode(encoded)
    assert torch.equal(large_input, decoded), "Large tensor roundtrip failed!"
    print("  [PASS] Large tensor roundtrip works")

    print("All Triton Hamming(8,4) verifications passed!")
    return True


if __name__ == "__main__":
    if torch.cuda.is_available():
        verify_triton_vs_cpu()
    else:
        print("CUDA not available, skipping Triton verification")
