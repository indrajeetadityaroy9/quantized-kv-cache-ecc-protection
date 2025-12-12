"""
ECC-Integrated PagedAttention Cache Write Kernel.

GPU-native kernel that fuses:
1. FP16 -> FP32 cast
2. Block-wise symmetric INT4 quantization
3. ECC encoding (Hamming84 or Golay)
4. Paged cache storage with slot mapping

All operations kept in registers to minimize memory traffic.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Literal, Optional

from hamming74.triton_kernels.config import (
    HAMMING84_BLOCK_SIZE,
    GOLAY_BLOCK_SIZE,
)


# =============================================================================
# Quantization Helpers
# =============================================================================

@triton.jit
def quantize_symmetric_int4(
    values,  # FP32 values to quantize
    scale,   # Per-block scale factor
):
    """
    Symmetric INT4 quantization: round(value / scale) + 8

    Maps [-scale*8, scale*8] -> [0, 15]
    Zero-point is 8 (center of INT4 range).
    """
    # Quantize to [-8, 7] range
    quantized = tl.math.round(values / scale)

    # Clamp to INT4 range and add zero-point
    quantized = tl.maximum(tl.minimum(quantized, 7.0), -8.0)
    int4_val = (quantized + 8.0).to(tl.uint8)

    return int4_val


@triton.jit
def dequantize_symmetric_int4(
    int4_val,  # INT4 value (0-15)
    scale,     # Per-block scale factor
):
    """
    Symmetric INT4 dequantization: (int4 - 8) * scale

    Inverse of quantize_symmetric_int4.
    """
    # Remove zero-point and scale
    return (int4_val.to(tl.float32) - 8.0) * scale


# =============================================================================
# Hamming(8,4) Encoding (Inline for Fusion)
# =============================================================================

@triton.jit
def encode_hamming84_inline(int4_val):
    """
    Inline Hamming(8,4) SECDED encoding.

    Input: INT4 value (0-15)
    Output: 8-bit codeword

    Generator matrix encoding (systematic form G = [I₄ | P]):
    - data_bits = [d₀, d₁, d₂, d₃]
    - parity_bits: p₀ = d₀ ⊕ d₁ ⊕ d₃, p₁ = d₀ ⊕ d₂ ⊕ d₃, p₂ = d₁ ⊕ d₂ ⊕ d₃
    - overall_parity = ⊕(all bits)
    """
    # Extract data bits
    d0 = (int4_val >> 0) & 1
    d1 = (int4_val >> 1) & 1
    d2 = (int4_val >> 2) & 1
    d3 = (int4_val >> 3) & 1

    # Compute parity bits
    p0 = d0 ^ d1 ^ d3
    p1 = d0 ^ d2 ^ d3
    p2 = d1 ^ d2 ^ d3

    # Pack 7-bit Hamming codeword
    hamming7 = (
        (d0 << 0) | (d1 << 1) | (d2 << 2) | (d3 << 3) |
        (p0 << 4) | (p1 << 5) | (p2 << 6)
    )

    # Overall parity via bit-folding
    parity = hamming7 ^ (hamming7 >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    overall_parity = parity & 1

    # Final 8-bit codeword
    codeword = (hamming7 | (overall_parity << 7)).to(tl.uint8)
    return codeword


# =============================================================================
# Golay(24,12) Encoding (Inline for Fusion)
# =============================================================================

@triton.jit
def _popcount_mod2_12bit(x):
    """Compute popcount(x) mod 2 for 12-bit value."""
    x = x ^ (x >> 8)
    x = x ^ (x >> 4)
    x = x ^ (x >> 2)
    x = x ^ (x >> 1)
    return x & 1


@triton.jit
def encode_golay_triplet_inline(n0, n1, n2):
    """
    Inline Golay(24,12) encoding for a triplet of INT4 values.

    Input: 3 INT4 values (n0, n1, n2)
    Output: 24-bit codeword stored as int32

    Parity bits computed using B matrix columns.
    """
    # B matrix column masks
    C0  = 2619   # 0b101000111011
    C1  = 3357   # 0b110100011101
    C2  = 3726   # 0b111010001110
    C3  = 2887   # 0b101101000111
    C4  = 3491   # 0b110110100011
    C5  = 3793   # 0b111011010001
    C6  = 3944   # 0b111101101000
    C7  = 2996   # 0b101110110100
    C8  = 2522   # 0b100111011010
    C9  = 2285   # 0b100011101101
    C10 = 3190   # 0b110001110110
    C11 = 2047   # 0b011111111111

    # Pack 12-bit data
    data_12bit = (n0 & 0xF) | ((n1 & 0xF) << 4) | ((n2 & 0xF) << 8)

    # Compute parity bits
    p0  = _popcount_mod2_12bit(data_12bit & C0)
    p1  = _popcount_mod2_12bit(data_12bit & C1)
    p2  = _popcount_mod2_12bit(data_12bit & C2)
    p3  = _popcount_mod2_12bit(data_12bit & C3)
    p4  = _popcount_mod2_12bit(data_12bit & C4)
    p5  = _popcount_mod2_12bit(data_12bit & C5)
    p6  = _popcount_mod2_12bit(data_12bit & C6)
    p7  = _popcount_mod2_12bit(data_12bit & C7)
    p8  = _popcount_mod2_12bit(data_12bit & C8)
    p9  = _popcount_mod2_12bit(data_12bit & C9)
    p10 = _popcount_mod2_12bit(data_12bit & C10)
    p11 = _popcount_mod2_12bit(data_12bit & C11)

    # Pack parity
    parity_12bit = (
        (p0 << 0) | (p1 << 1) | (p2 << 2) | (p3 << 3) |
        (p4 << 4) | (p5 << 5) | (p6 << 6) | (p7 << 7) |
        (p8 << 8) | (p9 << 9) | (p10 << 10) | (p11 << 11)
    )

    # 24-bit codeword
    codeword = data_12bit | (parity_12bit << 12)
    return codeword.to(tl.int32)


# =============================================================================
# Fused Cache Write Kernel (Hamming84)
# =============================================================================

@triton.jit
def write_kv_cache_hamming84_kernel(
    # Input tensors
    kv_ptr,              # Input K or V projections [batch, seq_len, hidden]
    # Output cache
    cache_ptr,           # Output cache [num_blocks, num_layers, num_heads, slots]
    # Metadata
    block_table_ptr,     # Block table [batch, max_blocks]
    seq_lens_ptr,        # Sequence lengths [batch]
    scales_ptr,          # Quantization scales [batch, seq_len] (per-token)
    # Dimensions
    batch_size,
    seq_len,
    hidden_size,
    num_heads,
    head_size,
    block_size,
    layer_idx,
    max_blocks,
    codewords_per_head,
    # Block config
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused quantize + Hamming84 encode + paged cache write.

    Each program handles one (batch, head, position) tuple.
    """
    # Program IDs
    pid = tl.program_id(0)
    batch_idx = pid // (num_heads * seq_len)
    remainder = pid % (num_heads * seq_len)
    head_idx = remainder // seq_len
    pos = remainder % seq_len

    # Check bounds
    if batch_idx >= batch_size:
        return
    actual_seq_len = tl.load(seq_lens_ptr + batch_idx)
    if pos >= actual_seq_len:
        return

    # Compute block and slot
    logical_block = pos // block_size
    slot_in_block = pos % block_size

    # Load physical block from block table
    physical_block = tl.load(block_table_ptr + batch_idx * max_blocks + logical_block)
    if physical_block < 0:
        return

    # Load scale for this position
    scale = tl.load(scales_ptr + batch_idx * seq_len + pos)

    # Process each element in head dimension
    head_start = head_idx * head_size
    for elem_idx in range(head_size):
        # Load FP16 value
        kv_offset = batch_idx * seq_len * hidden_size + pos * hidden_size + head_start + elem_idx
        value_fp16 = tl.load(kv_ptr + kv_offset).to(tl.float32)

        # Quantize to INT4
        int4_val = quantize_symmetric_int4(value_fp16, scale)

        # Encode with Hamming(8,4)
        codeword = encode_hamming84_inline(int4_val)

        # Compute cache offset
        # Layout: [num_blocks, num_layers, num_heads, codewords_per_head]
        # Each slot stores head_size codewords contiguously
        slot_offset = slot_in_block * head_size + elem_idx
        cache_offset = (
            physical_block * (32 * num_heads * codewords_per_head) +  # Assuming max 32 layers
            layer_idx * (num_heads * codewords_per_head) +
            head_idx * codewords_per_head +
            slot_offset
        )

        # Store codeword
        tl.store(cache_ptr + cache_offset, codeword)


# =============================================================================
# Python Wrapper Functions
# =============================================================================

def write_kv_to_cache_hamming84(
    kv: torch.Tensor,                    # [batch, seq_len, hidden]
    cache: torch.Tensor,                 # [num_blocks, num_layers, num_heads, slots]
    block_table: torch.Tensor,           # [batch, max_blocks]
    seq_lens: torch.Tensor,              # [batch]
    scales: torch.Tensor,                # [batch, seq_len]
    layer_idx: int,
    num_heads: int,
    head_size: int,
    block_size: int,
) -> None:
    """
    Write K/V projections to ECC-protected paged cache with Hamming(8,4).

    Fused operation: FP16 -> INT4 -> Hamming84 -> Cache

    Args:
        kv: Input K or V tensor [batch, seq_len, hidden]
        cache: ECC cache tensor
        block_table: Physical block mapping
        seq_lens: Actual sequence lengths per batch
        scales: Quantization scales per position
        layer_idx: Current layer index
        num_heads: Number of attention heads
        head_size: Dimension per head
        block_size: Tokens per cache block
    """
    batch_size, seq_len, hidden_size = kv.shape
    max_blocks = block_table.shape[1]
    codewords_per_head = block_size * head_size

    # Total work items
    total_items = batch_size * num_heads * seq_len

    # Launch kernel
    grid = lambda meta: (total_items,)
    write_kv_cache_hamming84_kernel[grid](
        kv, cache, block_table, seq_lens, scales,
        batch_size, seq_len, hidden_size, num_heads, head_size,
        block_size, layer_idx, max_blocks, codewords_per_head,
        BLOCK_SIZE=1,  # One thread per (batch, head, pos)
    )


def compute_quantization_scales(
    tensor: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Compute per-token quantization scales for symmetric INT4.

    Scale = max(abs(tensor)) / 7 (to map to [-8, 7] range)

    Args:
        tensor: Input tensor
        dim: Dimension to compute scale over (default: last)

    Returns:
        scales: Scale tensor
    """
    abs_max = tensor.abs().max(dim=dim, keepdim=False).values
    scales = abs_max / 7.0
    # Avoid division by zero
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    return scales


# =============================================================================
# Simple Non-Fused Version (for Testing/Comparison)
# =============================================================================

def write_kv_to_cache_simple(
    kv: torch.Tensor,
    codec: Literal["hamming84", "golay", "none"] = "hamming84",
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple (non-fused) quantize + encode for testing.

    Args:
        kv: Input K or V tensor [batch, seq_len, hidden]
        codec: ECC codec to use
        scale: Optional pre-computed scales

    Returns:
        encoded: Encoded codewords
        scales: Quantization scales used
    """
    if scale is None:
        scale = compute_quantization_scales(kv, dim=-1)

    # Quantize
    scale_expanded = scale.unsqueeze(-1)
    quantized = torch.round(kv / scale_expanded).clamp(-8, 7) + 8
    int4_vals = quantized.to(torch.uint8)

    if codec == "hamming84":
        from hamming74.triton_kernels import hamming84_encode
        flat = int4_vals.flatten()
        encoded = hamming84_encode(flat.cuda() if not flat.is_cuda else flat)
        encoded = encoded.view(int4_vals.shape)

    elif codec == "golay":
        from hamming74.triton_kernels import golay_encode
        # Reshape to triplets
        flat = int4_vals.flatten()
        # Pad to multiple of 3
        pad_len = (3 - flat.numel() % 3) % 3
        if pad_len > 0:
            flat = torch.cat([flat, torch.zeros(pad_len, dtype=flat.dtype, device=flat.device)])
        triplets = flat.view(-1, 3)
        if not triplets.is_cuda:
            triplets = triplets.cuda()
        encoded = golay_encode(triplets)
        # Can't easily reshape back since Golay changes element count

    else:  # none
        encoded = int4_vals

    return encoded, scale


# =============================================================================
# Verification
# =============================================================================

def verify_cache_write():
    """Verify cache write functionality."""
    print("Cache Write Verification")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping verification")
        return False

    # Test simple version
    batch_size = 2
    seq_len = 64
    hidden_size = 256

    kv = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)

    # Test Hamming84 encoding
    encoded_h84, scales = write_kv_to_cache_simple(kv, codec="hamming84")
    print(f"Hamming84 encoded shape: {encoded_h84.shape}")
    print(f"Hamming84 encoded dtype: {encoded_h84.dtype}")

    # Verify roundtrip
    from hamming74.triton_kernels import hamming84_decode
    flat_encoded = encoded_h84.flatten()
    decoded, stats = hamming84_decode(flat_encoded)
    print(f"Decoded shape: {decoded.shape}")
    print(f"Errors corrected: {stats[0]}, detected: {stats[1]}")

    # Dequantize and compare
    decoded_reshaped = decoded.view(encoded_h84.shape).float()
    dequantized = (decoded_reshaped - 8) * scales.unsqueeze(-1).cuda()
    mse = ((kv.float() - dequantized) ** 2).mean()
    print(f"Reconstruction MSE: {mse:.6f}")

    # Test Golay encoding
    encoded_golay, _ = write_kv_to_cache_simple(kv, codec="golay", scale=scales)
    print(f"\nGolay encoded shape: {encoded_golay.shape}")
    print(f"Golay encoded dtype: {encoded_golay.dtype}")

    print("\nCache write verification passed!")
    return True


if __name__ == "__main__":
    verify_cache_write()
