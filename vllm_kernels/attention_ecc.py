"""
ECC-Integrated PagedAttention Read Kernel.

GPU-native kernel that fuses:
1. Paged cache lookup via block tables
2. ECC decoding (Hamming84 or Golay) on-the-fly
3. INT4 -> FP16/FP32 dequantization
4. Scaled dot-product attention with Online Softmax

Parallelism Strategy:
- One Thread Block per (batch, head) pair
- Threads parallelize over HEAD_DIM dimension
- tl.sum() reduction for dot products
- Sequential loop over KV blocks
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional, Literal
import math

from hamming74.triton_kernels.config import SYNDROME_LUT_HAMMING84
from hamming74.triton_kernels.golay_triton import _H_ROW_MASKS, _build_syndrome_table


# =============================================================================
# Inline Hamming(8,4) Decoder (Register-Resident)
# =============================================================================

@triton.jit
def decode_hamming84_inline(
    codeword,
    # Syndrome LUT (8 entries passed as constexpr for register storage)
    lut0: tl.constexpr, lut1: tl.constexpr, lut2: tl.constexpr, lut3: tl.constexpr,
    lut4: tl.constexpr, lut5: tl.constexpr, lut6: tl.constexpr, lut7: tl.constexpr,
):
    """
    Inline Hamming(8,4) SECDED decoder.

    All operations in registers - no memory access.
    Returns decoded INT4 value (0-15).

    SECDED Logic:
    - syndrome=0, parity=0: No error
    - syndrome=0, parity=1: Error in overall parity bit only
    - syndrome≠0, parity=1: Single-bit error, correctable
    - syndrome≠0, parity=0: Double-bit error, detected only (zeroed)
    """
    # Extract Hamming(7,4) portion (bits 0-6) and overall parity (bit 7)
    hamming7 = codeword & 0x7F
    stored_parity = (codeword >> 7) & 1

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
    syndrome = (s0 | (s1 << 1) | (s2 << 2))

    # Compute actual parity of received 7 bits (via bit-folding XOR)
    actual_parity = hamming7 ^ (hamming7 >> 4)
    actual_parity = actual_parity ^ (actual_parity >> 2)
    actual_parity = actual_parity ^ (actual_parity >> 1)
    actual_parity = actual_parity & 1

    # Parity error flag
    parity_error = (stored_parity != actual_parity)

    # SECDED classification
    syndrome_zero = (syndrome == 0)

    # error_type: 0=NO_ERROR, 1=SINGLE_CORRECTED, 2=DOUBLE_DETECTED, 3=PARITY_ONLY
    is_single_error = (~syndrome_zero) & parity_error
    is_double_error = (~syndrome_zero) & (~parity_error)

    # Lookup error position from syndrome (inline LUT to keep in registers)
    error_pos = tl.where(syndrome == 0, lut0,
                tl.where(syndrome == 1, lut1,
                tl.where(syndrome == 2, lut2,
                tl.where(syndrome == 3, lut3,
                tl.where(syndrome == 4, lut4,
                tl.where(syndrome == 5, lut5,
                tl.where(syndrome == 6, lut6,
                         lut7)))))))

    # Compute correction mask (only for single-bit errors)
    should_correct = is_single_error & (error_pos >= 0)
    correction_mask = tl.where(should_correct, 1 << error_pos, 0)

    # Apply correction via XOR
    corrected = hamming7 ^ correction_mask

    # Handle double errors: zero out (policy: on_double_error="zero")
    corrected = tl.where(is_double_error, 0, corrected)

    # Extract data bits (first 4 bits in systematic form)
    decoded = corrected & 0x0F

    return decoded


@triton.jit
def dequantize_int4(int4_val, scale):
    """
    Dequantize INT4 value to float.

    Symmetric dequantization: (int4 - 8) * scale
    Maps [0, 15] -> [-8*scale, 7*scale]
    """
    return (int4_val.to(tl.float32) - 8.0) * scale


# =============================================================================
# Inline Golay(24,12) Decoder (Shared Memory LUT)
# =============================================================================

@triton.jit
def _popcount_mod2_24bit(x):
    """Compute popcount(x) mod 2 for 24-bit value via bit-folding."""
    x = x ^ (x >> 16)  # Fold upper 8 bits
    x = x ^ (x >> 8)   # Fold to 8 bits
    x = x ^ (x >> 4)   # Fold to 4 bits
    x = x ^ (x >> 2)   # Fold to 2 bits
    x = x ^ (x >> 1)   # Fold to 1 bit
    return x & 1


@triton.jit
def decode_golay_inline(
    codeword,
    syndrome_lut_ptr,
    # H matrix row masks (24-bit each) - passed as constexpr
    H0: tl.constexpr, H1: tl.constexpr, H2: tl.constexpr, H3: tl.constexpr,
    H4: tl.constexpr, H5: tl.constexpr, H6: tl.constexpr, H7: tl.constexpr,
    H8: tl.constexpr, H9: tl.constexpr, H10: tl.constexpr, H11: tl.constexpr,
):
    """
    Inline Golay(24,12) decoder.

    Uses shared memory syndrome LUT for O(1) error pattern lookup.
    Returns (n0, n1, n2) - three INT4 values.
    """
    # Compute 12-bit syndrome
    s0 = _popcount_mod2_24bit(codeword & H0)
    s1 = _popcount_mod2_24bit(codeword & H1)
    s2 = _popcount_mod2_24bit(codeword & H2)
    s3 = _popcount_mod2_24bit(codeword & H3)
    s4 = _popcount_mod2_24bit(codeword & H4)
    s5 = _popcount_mod2_24bit(codeword & H5)
    s6 = _popcount_mod2_24bit(codeword & H6)
    s7 = _popcount_mod2_24bit(codeword & H7)
    s8 = _popcount_mod2_24bit(codeword & H8)
    s9 = _popcount_mod2_24bit(codeword & H9)
    s10 = _popcount_mod2_24bit(codeword & H10)
    s11 = _popcount_mod2_24bit(codeword & H11)

    syndrome = (
        (s0 << 0) | (s1 << 1) | (s2 << 2) | (s3 << 3) |
        (s4 << 4) | (s5 << 5) | (s6 << 6) | (s7 << 7) |
        (s8 << 8) | (s9 << 9) | (s10 << 10) | (s11 << 11)
    )

    # Lookup error pattern from syndrome table
    # -1 indicates uncorrectable (>3 errors)
    error_pattern = tl.load(syndrome_lut_ptr + syndrome)

    # Check if correctable
    is_correctable = (error_pattern >= 0)

    # Apply correction via XOR (only if correctable)
    corrected = tl.where(is_correctable, codeword ^ error_pattern, codeword)

    # Extract data bits (first 12 bits in systematic form)
    data_12bit = corrected & 0xFFF

    # Unpack to 3 INT4 values
    n0 = (data_12bit >> 0) & 0xF
    n1 = (data_12bit >> 4) & 0xF
    n2 = (data_12bit >> 8) & 0xF

    # For uncorrectable errors, zero out the data
    n0 = tl.where(is_correctable, n0, 0)
    n1 = tl.where(is_correctable, n1, 0)
    n2 = tl.where(is_correctable, n2, 0)

    return n0, n1, n2


# =============================================================================
# PagedAttention ECC Kernel (Hamming84-Only, Iteration 1)
# =============================================================================

@triton.jit
def paged_attention_ecc_kernel(
    # Output
    output_ptr,           # [batch, num_heads, head_dim]
    # Query input
    query_ptr,            # [batch, num_heads, head_dim]
    # ECC-encoded KV cache
    k_cache_ptr,          # [num_blocks, num_layers, num_heads, codewords_per_head]
    v_cache_ptr,          # [num_blocks, num_layers, num_heads, codewords_per_head]
    # Metadata
    block_table_ptr,      # [batch, max_blocks]
    context_lens_ptr,     # [batch]
    scales_ptr,           # [num_blocks, num_layers, num_heads, block_size]
    # Dimensions
    batch_size,
    num_heads,
    head_dim,
    num_layers,
    layer_idx,
    block_size,
    max_blocks,
    max_context_len,
    # Attention scaling
    sm_scale,
    # Syndrome LUT for Hamming84
    lut0: tl.constexpr, lut1: tl.constexpr, lut2: tl.constexpr, lut3: tl.constexpr,
    lut4: tl.constexpr, lut5: tl.constexpr, lut6: tl.constexpr, lut7: tl.constexpr,
    # Block sizes
    BLOCK_HEAD_DIM: tl.constexpr,
    MAX_CONTEXT_BLOCKS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
):
    """
    PagedAttention with ECC decoding (Hamming84).

    Parallelism:
    - One program per (batch, head) pair
    - Threads parallelize over HEAD_DIM using BLOCK_HEAD_DIM
    - Sequential loop over context KV blocks

    Online Softmax accumulation:
    - m: running max
    - l: running sum of exp(x - m)
    - acc: weighted sum of values
    """
    # Program ID -> (batch, head)
    pid = tl.program_id(0)
    batch_idx = pid // num_heads
    head_idx = pid % num_heads

    # Thread indices within head dimension
    head_offsets = tl.arange(0, BLOCK_HEAD_DIM)
    head_mask = head_offsets < head_dim

    # Load context length for this sequence
    context_len = tl.load(context_lens_ptr + batch_idx)

    # Load query vector for this (batch, head)
    # Q shape: [batch, num_heads, head_dim]
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim
    q = tl.load(query_ptr + q_offset + head_offsets, mask=head_mask, other=0.0)

    # Initialize Online Softmax state
    m_i = -1e20  # Running max (use large negative instead of -inf for stability)
    l_i = 0.0    # Running sum
    acc = tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32)  # Accumulator

    # Loop over KV blocks (use constexpr max to enable unrolling)
    for block_idx in range(MAX_CONTEXT_BLOCKS):
        # Check if this block is within context
        start_pos = block_idx * KV_BLOCK_SIZE
        block_valid = start_pos < context_len

        # Load physical block index from block table
        physical_block = tl.load(
            block_table_ptr + batch_idx * max_blocks + block_idx,
            mask=block_valid,
            other=-1
        )

        # Process each token in the block
        for slot in range(KV_BLOCK_SIZE):
            token_pos = start_pos + slot
            token_valid = block_valid & (token_pos < context_len) & (physical_block >= 0)

            # ==== Load and decode K ====
            # Cache layout: [num_blocks, num_layers, num_heads, codewords_per_head]
            # codewords_per_head = block_size * head_dim for Hamming84
            k_base_offset = (
                physical_block * num_layers * num_heads * KV_BLOCK_SIZE * head_dim +
                layer_idx * num_heads * KV_BLOCK_SIZE * head_dim +
                head_idx * KV_BLOCK_SIZE * head_dim +
                slot * head_dim
            )

            # Load encoded K values (uint8 codewords) - use mask for validity
            load_mask = head_mask & token_valid
            k_encoded = tl.load(
                k_cache_ptr + k_base_offset + head_offsets,
                mask=load_mask,
                other=0
            ).to(tl.uint8)

            # Decode Hamming84 (inline)
            k_int4 = decode_hamming84_inline(
                k_encoded,
                lut0, lut1, lut2, lut3, lut4, lut5, lut6, lut7
            )

            # Load scale for dequantization
            # Scale layout: [num_blocks, num_layers, num_heads, block_size]
            scale_offset = (
                physical_block * num_layers * num_heads * KV_BLOCK_SIZE +
                layer_idx * num_heads * KV_BLOCK_SIZE +
                head_idx * KV_BLOCK_SIZE +
                slot
            )
            scale = tl.load(scales_ptr + scale_offset, mask=token_valid, other=1.0)

            # Dequantize to float
            k_float = dequantize_int4(k_int4, scale)

            # ==== Compute attention score ====
            # Dot product: q · k (parallel across head_dim, then reduce)
            qk_partial = q * k_float
            qk_score = tl.sum(qk_partial, axis=0) * sm_scale

            # Mask invalid scores to -inf (won't contribute to softmax)
            qk_score = tl.where(token_valid, qk_score, -1e20)

            # ==== Online Softmax update ====
            # m_new = max(m_i, qk_score)
            m_new = tl.maximum(m_i, qk_score)

            # alpha = exp(m_i - m_new), beta = exp(qk_score - m_new)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(qk_score - m_new)

            # l_new = alpha * l_i + beta
            l_new = alpha * l_i + beta

            # ==== Load and decode V ====
            v_base_offset = (
                physical_block * num_layers * num_heads * KV_BLOCK_SIZE * head_dim +
                layer_idx * num_heads * KV_BLOCK_SIZE * head_dim +
                head_idx * KV_BLOCK_SIZE * head_dim +
                slot * head_dim
            )

            v_encoded = tl.load(
                v_cache_ptr + v_base_offset + head_offsets,
                mask=load_mask,
                other=0
            ).to(tl.uint8)

            v_int4 = decode_hamming84_inline(
                v_encoded,
                lut0, lut1, lut2, lut3, lut4, lut5, lut6, lut7
            )

            v_float = dequantize_int4(v_int4, scale)

            # ==== Accumulate weighted V (only for valid tokens) ====
            # For invalid tokens, beta=0 (exp(-inf-m_new) ~= 0) so they don't contribute
            acc = alpha * acc + beta * v_float

            # Update running stats
            m_i = m_new
            l_i = l_new

    # ==== Finalize: output = acc / l ====
    # Avoid division by zero
    output = tl.where(l_i > 0, acc / l_i, tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32))

    # Store output
    out_offset = batch_idx * num_heads * head_dim + head_idx * head_dim
    tl.store(output_ptr + out_offset + head_offsets, output, mask=head_mask)


# =============================================================================
# Python Wrapper
# =============================================================================

def paged_attention_ecc(
    query: torch.Tensor,           # [batch, num_heads, head_dim]
    k_cache: torch.Tensor,         # [num_blocks, num_layers, num_heads, codewords_per_head]
    v_cache: torch.Tensor,         # [num_blocks, num_layers, num_heads, codewords_per_head]
    block_table: torch.Tensor,     # [batch, max_blocks]
    context_lens: torch.Tensor,    # [batch]
    scales: torch.Tensor,          # [num_blocks, num_layers, num_heads, block_size]
    layer_idx: int,
    block_size: int,
    sm_scale: Optional[float] = None,
    codec: Literal["hamming84", "golay"] = "hamming84",
    syndrome_table: Optional[torch.Tensor] = None,  # For Golay: pre-computed syndrome LUT
) -> torch.Tensor:
    """
    PagedAttention with ECC decoding.

    Args:
        query: Query tensor [batch, num_heads, head_dim]
        k_cache: ECC-encoded key cache (Hamming84: uint8, Golay: int32)
        v_cache: ECC-encoded value cache (Hamming84: uint8, Golay: int32)
        block_table: Physical block mapping [batch, max_blocks]
        context_lens: Context length per sequence [batch]
        scales: Dequantization scales [num_blocks, num_layers, num_heads, block_size]
        layer_idx: Current layer index
        block_size: Tokens per cache block
        sm_scale: Attention scaling factor (default: 1/sqrt(head_dim))
        codec: ECC codec to use ("hamming84" or "golay")
        syndrome_table: For Golay - pre-computed syndrome LUT (4096 entries)

    Returns:
        output: Attention output [batch, num_heads, head_dim]
    """
    assert query.is_cuda, "Query must be on CUDA"

    batch_size, num_heads, head_dim = query.shape
    num_blocks, num_layers, _, codewords_per_head = k_cache.shape
    max_blocks = block_table.shape[1]
    max_context_len = max_blocks * block_size

    # Default attention scale
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Allocate output
    output = torch.empty_like(query)

    if codec == "hamming84":
        assert k_cache.dtype == torch.uint8, "Hamming84 K cache must be uint8"
        assert v_cache.dtype == torch.uint8, "Hamming84 V cache must be uint8"

        # Syndrome LUT for Hamming84
        lut = SYNDROME_LUT_HAMMING84

        # Determine BLOCK_HEAD_DIM (power of 2 >= head_dim)
        BLOCK_HEAD_DIM = triton.next_power_of_2(head_dim)

        # Constexpr for loop bounds (must be compile-time constants)
        MAX_CONTEXT_BLOCKS = max_blocks
        KV_BLOCK_SIZE = block_size

        # Launch kernel
        grid = (batch_size * num_heads,)
        paged_attention_ecc_kernel[grid](
            output,
            query,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            scales,
            batch_size,
            num_heads,
            head_dim,
            num_layers,
            layer_idx,
            block_size,
            max_blocks,
            max_context_len,
            sm_scale,
            int(lut[0]), int(lut[1]), int(lut[2]), int(lut[3]),
            int(lut[4]), int(lut[5]), int(lut[6]), int(lut[7]),
            BLOCK_HEAD_DIM=BLOCK_HEAD_DIM,
            MAX_CONTEXT_BLOCKS=MAX_CONTEXT_BLOCKS,
            KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        )

    elif codec == "golay":
        # Golay has different memory layout (3:1 packing)
        # For now, use reference implementation
        # TODO: Implement fused Golay attention kernel
        assert k_cache.dtype == torch.int32, "Golay K cache must be int32"
        assert v_cache.dtype == torch.int32, "Golay V cache must be int32"

        # Fall back to reference implementation for Golay
        output = reference_attention_ecc(
            query, k_cache, v_cache, block_table, context_lens, scales,
            layer_idx, block_size, head_dim, sm_scale, codec="golay"
        )

    else:
        raise ValueError(f"Unknown codec: {codec}")

    return output


def paged_attention_ecc_adaptive(
    query: torch.Tensor,           # [batch, num_heads, head_dim]
    # Hamming84 cache (for context blocks)
    k_cache: torch.Tensor,         # [num_blocks, num_layers, num_heads, codewords_per_head] uint8
    v_cache: torch.Tensor,         # Same
    # Golay cache (for sink blocks)
    sink_k_cache: torch.Tensor,    # [num_blocks, num_layers, num_heads, codewords_per_head] int32
    sink_v_cache: torch.Tensor,    # Same
    # Metadata
    block_table: torch.Tensor,     # [batch, max_blocks]
    context_lens: torch.Tensor,    # [batch]
    scales: torch.Tensor,          # [num_blocks, num_layers, num_heads, block_size]
    sink_scales: Optional[torch.Tensor] = None,  # Optional separate scales for sink blocks
    layer_idx: int = 0,
    block_size: int = 16,
    sink_boundary: int = 4,  # Number of blocks to protect with Golay
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    PagedAttention with Adaptive Unequal Error Protection (UEP).

    Uses position-based codec routing:
    - Sink blocks (0 to sink_boundary-1): Golay(24,12) - corrects up to 3 errors
    - Context blocks (>= sink_boundary): Hamming(8,4) - corrects 1 error

    This implements the "attention sink" protection strategy where early tokens
    that receive disproportionately high attention weights get stronger ECC
    protection to minimize their error amplification through the softmax.

    Args:
        query: Query tensor [batch, num_heads, head_dim]
        k_cache: Hamming84-encoded key cache (uint8)
        v_cache: Hamming84-encoded value cache (uint8)
        sink_k_cache: Golay-encoded key cache for sink blocks (int32)
        sink_v_cache: Golay-encoded value cache for sink blocks (int32)
        block_table: Physical block mapping [batch, max_blocks]
        context_lens: Context length per sequence [batch]
        scales: Dequantization scales (for Hamming84 blocks)
        sink_scales: Dequantization scales for sink blocks (uses main scales if None)
        layer_idx: Current layer index
        block_size: Tokens per cache block
        sink_boundary: Number of blocks to protect with Golay
        sm_scale: Attention scaling factor (default: 1/sqrt(head_dim))

    Returns:
        output: Attention output [batch, num_heads, head_dim]

    Note:
        Currently uses reference implementation for adaptive routing.
        A fused Triton kernel for adaptive UEP is a future optimization.
    """
    batch_size, num_heads, head_dim = query.shape

    return reference_attention_ecc(
        query=query,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        context_lens=context_lens,
        scales=scales,
        layer_idx=layer_idx,
        block_size=block_size,
        head_dim=head_dim,
        sm_scale=sm_scale,
        codec="hamming84",  # Default for context blocks
        sink_boundary=sink_boundary,
        sink_k_cache=sink_k_cache,
        sink_v_cache=sink_v_cache,
        sink_scales=sink_scales,
    )


# =============================================================================
# Reference Implementation (Non-Fused, for Testing)
# =============================================================================

def reference_attention_ecc(
    query: torch.Tensor,           # [batch, num_heads, head_dim]
    k_cache: torch.Tensor,         # [num_blocks, num_layers, num_heads, codewords_per_head]
    v_cache: torch.Tensor,         # [num_blocks, num_layers, num_heads, codewords_per_head]
    block_table: torch.Tensor,     # [batch, max_blocks]
    context_lens: torch.Tensor,    # [batch]
    scales: torch.Tensor,          # [num_blocks, num_layers, num_heads, block_size]
    layer_idx: int,
    block_size: int,
    head_dim: int,
    sm_scale: Optional[float] = None,
    codec: Literal["hamming84", "golay", "none"] = "hamming84",
    # Adaptive UEP parameters
    sink_boundary: int = 0,  # Number of blocks to use sink_codec (0 = disabled)
    sink_k_cache: Optional[torch.Tensor] = None,  # Golay cache for sink blocks
    sink_v_cache: Optional[torch.Tensor] = None,
    sink_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Reference (non-fused) attention implementation for verification.

    Uses the standalone Triton decode kernels then computes attention in PyTorch.

    Memory layouts:
    - Hamming84: codewords_per_head = block_size * head_dim (1:1)
    - Golay: codewords_per_head = ceil(block_size * head_dim / 3) (3:1 packing)

    Adaptive UEP:
    When sink_boundary > 0 and sink caches are provided:
    - Blocks 0 to sink_boundary-1: Use Golay from sink caches (stronger protection)
    - Blocks >= sink_boundary: Use Hamming84 from main caches (lighter protection)
    This implements position-based Unequal Error Protection where attention sinks
    (early tokens that receive disproportionately high attention) get stronger ECC.
    """
    from hamming74.triton_kernels import hamming84_decode, golay_decode

    batch_size, num_heads, _ = query.shape
    device = query.device

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Determine if adaptive UEP is enabled
    adaptive_uep = (sink_boundary > 0 and
                    sink_k_cache is not None and
                    sink_v_cache is not None)

    outputs = []

    for b in range(batch_size):
        ctx_len = int(context_lens[b].item())
        if ctx_len <= 0:
            outputs.append(torch.zeros(num_heads, head_dim, device=device))
            continue

        num_ctx_blocks = (ctx_len + block_size - 1) // block_size

        # Collect K and V for this sequence
        k_list = []
        v_list = []
        scale_list = []

        for blk_idx in range(num_ctx_blocks):
            phys_block = int(block_table[b, blk_idx].item())
            if phys_block < 0:
                continue

            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, ctx_len)
            tokens_in_block = end_pos - start_pos

            # Determine codec for this block (adaptive UEP routing)
            if adaptive_uep and blk_idx < sink_boundary:
                block_codec = "golay"
                block_k_cache = sink_k_cache
                block_v_cache = sink_v_cache
                block_scales = sink_scales if sink_scales is not None else scales
            else:
                block_codec = codec
                block_k_cache = k_cache
                block_v_cache = v_cache
                block_scales = scales

            for slot in range(tokens_in_block):
                if block_codec == "hamming84":
                    # Hamming84: 1:1 layout - [num_blocks, num_layers, num_heads, block_size * head_dim]
                    k_offset_start = slot * head_dim
                    k_offset_end = k_offset_start + head_dim

                    k_encoded = block_k_cache[phys_block, layer_idx, :, k_offset_start:k_offset_end]
                    v_encoded = block_v_cache[phys_block, layer_idx, :, k_offset_start:k_offset_end]

                    k_decoded, _ = hamming84_decode(k_encoded.flatten())
                    v_decoded, _ = hamming84_decode(v_encoded.flatten())
                    k_decoded = k_decoded.view(num_heads, head_dim)
                    v_decoded = v_decoded.view(num_heads, head_dim)

                elif block_codec == "golay":
                    # Golay: 3:1 packing - [num_blocks, num_layers, num_heads, codewords_per_slot]
                    # Each slot has ceil(head_dim / 3) codewords
                    codewords_per_slot = (head_dim + 2) // 3

                    k_offset_start = slot * codewords_per_slot
                    k_offset_end = k_offset_start + codewords_per_slot

                    k_encoded = block_k_cache[phys_block, layer_idx, :, k_offset_start:k_offset_end]
                    v_encoded = block_v_cache[phys_block, layer_idx, :, k_offset_start:k_offset_end]

                    # Decode per head
                    k_decoded_list = []
                    v_decoded_list = []
                    for h in range(num_heads):
                        k_triplets, _ = golay_decode(k_encoded[h])  # [codewords, 3]
                        v_triplets, _ = golay_decode(v_encoded[h])
                        # Flatten and trim to head_dim
                        k_vals = k_triplets.flatten()[:head_dim]
                        v_vals = v_triplets.flatten()[:head_dim]
                        k_decoded_list.append(k_vals)
                        v_decoded_list.append(v_vals)

                    k_decoded = torch.stack(k_decoded_list, dim=0)  # [num_heads, head_dim]
                    v_decoded = torch.stack(v_decoded_list, dim=0)

                else:  # none
                    k_offset_start = slot * head_dim
                    k_offset_end = k_offset_start + head_dim
                    k_decoded = block_k_cache[phys_block, layer_idx, :, k_offset_start:k_offset_end]
                    v_decoded = block_v_cache[phys_block, layer_idx, :, k_offset_start:k_offset_end]

                # Get scale for this slot
                slot_scale = block_scales[phys_block, layer_idx, :, slot]  # [num_heads]

                k_list.append(k_decoded)
                v_list.append(v_decoded)
                scale_list.append(slot_scale)

        if not k_list:
            outputs.append(torch.zeros(num_heads, head_dim, device=device))
            continue

        # Stack: [ctx_len, num_heads, head_dim]
        k_stacked = torch.stack(k_list, dim=0)
        v_stacked = torch.stack(v_list, dim=0)
        scale_stacked = torch.stack(scale_list, dim=0)  # [ctx_len, num_heads]

        # Dequantize: (int4 - 8) * scale
        k_float = (k_stacked.float() - 8.0) * scale_stacked.unsqueeze(-1)
        v_float = (v_stacked.float() - 8.0) * scale_stacked.unsqueeze(-1)

        # Transpose for attention: [num_heads, ctx_len, head_dim]
        k_float = k_float.permute(1, 0, 2)
        v_float = v_float.permute(1, 0, 2)

        # Query for this batch: [num_heads, head_dim]
        q = query[b]  # [num_heads, head_dim]

        # Attention scores: [num_heads, ctx_len]
        scores = torch.einsum('hd,hcd->hc', q, k_float) * sm_scale

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum: [num_heads, head_dim]
        out = torch.einsum('hc,hcd->hd', attn_weights, v_float)

        outputs.append(out)

    return torch.stack(outputs, dim=0)


# =============================================================================
# Verification
# =============================================================================

def verify_attention_kernel():
    """Verify attention kernel produces correct results."""
    print("Attention ECC Kernel Verification")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping verification")
        return False

    from hamming74.triton_kernels import hamming84_encode

    # Test configuration
    batch_size = 2
    num_heads = 4
    head_dim = 64
    num_layers = 2
    layer_idx = 0
    block_size = 16
    context_len = 48  # 3 blocks

    num_blocks = 32
    max_blocks = 8

    print(f"Config: batch={batch_size}, heads={num_heads}, head_dim={head_dim}")
    print(f"Context: {context_len} tokens, block_size={block_size}")

    device = "cuda"

    # Create random Q, K, V (FP16)
    torch.manual_seed(42)
    query = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float32)

    # Create block table
    block_table = torch.full((batch_size, max_blocks), -1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        num_ctx_blocks = (context_len + block_size - 1) // block_size
        for i in range(num_ctx_blocks):
            block_table[b, i] = b * num_ctx_blocks + i

    context_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device=device)

    # Create KV cache (encode with Hamming84)
    codewords_per_head = block_size * head_dim
    k_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                          dtype=torch.uint8, device=device)
    v_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                          dtype=torch.uint8, device=device)
    scales = torch.zeros(num_blocks, num_layers, num_heads, block_size,
                         dtype=torch.float32, device=device)

    # Fill cache with random quantized values
    for b in range(batch_size):
        num_ctx_blocks = (context_len + block_size - 1) // block_size
        for blk_idx in range(num_ctx_blocks):
            phys_block = int(block_table[b, blk_idx].item())

            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, context_len)
            tokens_in_block = end_pos - start_pos

            for slot in range(tokens_in_block):
                # Random FP16 KV data
                k_fp = torch.randn(num_heads, head_dim, device=device)
                v_fp = torch.randn(num_heads, head_dim, device=device)

                # Compute scales
                k_scale = k_fp.abs().max(dim=-1).values / 7.0
                v_scale = v_fp.abs().max(dim=-1).values / 7.0
                scale = torch.maximum(k_scale, v_scale)
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)

                # Quantize
                k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)
                v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)

                # Encode with Hamming84
                k_encoded = hamming84_encode(k_int4.flatten()).view(num_heads, head_dim)
                v_encoded = hamming84_encode(v_int4.flatten()).view(num_heads, head_dim)

                # Store in cache
                offset_start = slot * head_dim
                offset_end = offset_start + head_dim
                k_cache[phys_block, layer_idx, :, offset_start:offset_end] = k_encoded
                v_cache[phys_block, layer_idx, :, offset_start:offset_end] = v_encoded
                scales[phys_block, layer_idx, :, slot] = scale

    print("\nRunning reference implementation...")
    ref_output = reference_attention_ecc(
        query, k_cache, v_cache, block_table, context_lens, scales,
        layer_idx, block_size, head_dim
    )
    print(f"Reference output shape: {ref_output.shape}")

    print("\nRunning Triton kernel...")
    triton_output = paged_attention_ecc(
        query, k_cache, v_cache, block_table, context_lens, scales,
        layer_idx, block_size
    )
    print(f"Triton output shape: {triton_output.shape}")

    # Compare
    diff = (ref_output - triton_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nMax absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")

    # Check within tolerance (quantization introduces some error)
    tolerance = 1e-4
    if max_diff < tolerance:
        print(f"\n✓ Verification PASSED (max_diff < {tolerance})")
        return True
    else:
        print(f"\n✗ Verification FAILED (max_diff >= {tolerance})")
        return False


if __name__ == "__main__":
    verify_attention_kernel()
