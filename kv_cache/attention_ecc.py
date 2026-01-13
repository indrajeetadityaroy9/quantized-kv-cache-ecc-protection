"""
ECC-Decoded Paged Attention: Triton kernels with inline Hamming/Golay decoding.

This module implements GPU-accelerated paged attention that decodes ECC-protected
cache entries on-the-fly during the scaled dot-product computation. By fusing ECC
decode into the attention kernel, we avoid materializing decoded K,V tensors.

Architecture:
    - decode_hamming84_inline: Inline SECDED decoding (no function call overhead)
    - decode_golay_inline: Inline Golay(24,12) decoding for 3-error correction
    - paged_attention_ecc_kernel: Token-by-token attention with online softmax
    - paged_attention_ecc_tiled_kernel: Experimental tiled variant (slower - see note)
    - paged_attention_ecc: Python wrapper that dispatches to appropriate kernel
    - reference_attention_ecc: Python reference implementation for correctness testing

Double-Error Handling:
    The inline decoder PRESERVES corrupted data on double-error detection rather
    than zeroing it. This provides a better approximation than zero for attention
    computation. For applications requiring interpolation on double errors, use
    the Python path in ecc_shim.py which calls interpolate_double_errors().

Performance Note:
    The tiled kernel was implemented to reduce softmax rescaling overhead, but
    benchmarking shows it's SLOWER than the token-by-token kernel. ECC decoding
    (bit manipulation) dominates compute time, making Flash Attention assumptions
    inapplicable. Use use_tiled=False (default) for best performance.

Memory Layout:
    k_cache, v_cache: [num_blocks, num_layers, num_heads, block_size * head_dim]
    k_scales: [num_blocks, num_layers, num_heads, block_size]
    block_table: [batch_size, max_blocks] - maps logical to physical blocks

Usage:
    output = paged_attention_ecc(
        query,           # [batch, heads, head_dim]
        k_cache,         # Hamming84-encoded K cache
        v_cache,         # Hamming84-encoded V cache
        block_table,     # Physical block mapping
        context_lens,    # Context length per sequence
        k_scales,        # Quantization scales
        layer_idx,
        block_size,
    )
"""
import torch
import triton
import triton.language as tl
import math

from ecc_codecs.triton_kernels import hamming84_decode, golay_decode, hamming84_encode
from ecc_codecs.triton_kernels.config import SYNDROME_LUT_HAMMING84
from ecc_codecs.triton_kernels.golay_triton import _H_ROW_MASKS, _build_syndrome_table


@triton.jit
def decode_hamming84_inline(
    codeword,
    lut0: tl.constexpr,
    lut1: tl.constexpr,
    lut2: tl.constexpr,
    lut3: tl.constexpr,
    lut4: tl.constexpr,
    lut5: tl.constexpr,
    lut6: tl.constexpr,
    lut7: tl.constexpr,
):
    """
    Inline Hamming(8,4) SECDED decode without function call overhead.

    This is a Triton JIT function that performs single-error-correct, double-error-
    detect decoding directly in the attention kernel. By inlining, we avoid the
    overhead of calling a separate decode function for each cache element.

    Algorithm:
        1. Extract 7-bit Hamming codeword and overall parity bit
        2. Compute 3-bit syndrome from parity checks
        3. Compute actual parity to detect double errors
        4. Classify: no_error, single_error (correctable), double_error (detected)
        5. Correct single errors using syndrome lookup table
        6. Return lower 4 data bits

    Args:
        codeword: 8-bit Hamming(8,4) codeword [d0,d1,d2,d3,p0,p1,p2,overall_parity]
        lut0-lut7: Syndrome lookup table entries (error bit position for each syndrome)
                   Passed as constexpr to avoid memory loads in inner loop

    Returns:
        Decoded 4-bit INT4 value (0-15). On double error, returns corrupted data
        rather than zero to provide better approximation for attention computation.

    Note:
        The LUT is passed as 8 separate constexpr values because Triton doesn't
        support dynamic indexing into arrays with tensor indices. The nested
        tl.where() implements a branchless switch on syndrome value.
    """
    # Extract 7-bit Hamming code (bits 0-6) and overall parity (bit 7)
    hamming7 = codeword & 0x7F
    stored_parity = (codeword >> 7) & 1

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

    syndrome = s0 | (s1 << 1) | (s2 << 2)

    actual_parity = hamming7 ^ (hamming7 >> 4)
    actual_parity = actual_parity ^ (actual_parity >> 2)
    actual_parity = actual_parity ^ (actual_parity >> 1)
    actual_parity = actual_parity & 1

    parity_error = stored_parity != actual_parity

    syndrome_zero = syndrome == 0

    is_single_error = (~syndrome_zero) & parity_error
    is_double_error = (~syndrome_zero) & (~parity_error)

    error_pos = tl.where(
        syndrome == 0,lut0,
        tl.where(syndrome == 1,lut1,
            tl.where(syndrome == 2,lut2,
                tl.where(syndrome == 3,lut3,
                    tl.where(syndrome == 4,lut4,
                        tl.where(syndrome == 5, lut5, tl.where(syndrome == 6, lut6, lut7)
                        ),
                    ),
                ),
            ),
        ),
    )

    should_correct = is_single_error & (error_pos >= 0)
    correction_mask = tl.where(should_correct, 1 << error_pos, 0)
    corrected = hamming7 ^ correction_mask
    # NOTE: On double error (detected but uncorrectable), we PRESERVE the corrupted
    # data rather than zeroing it. This provides a better approximation for attention
    # than returning 0. For interpolation-based recovery, use the Python path in
    # ecc_shim.py which calls interpolate_double_errors() after hamming84_decode().
    decoded = corrected & 0x0F
    return decoded


@triton.jit
def dequantize_int4(int4_val, scale):
    """
    Dequantize unsigned INT4 (0-15) back to float using symmetric quantization.

    Maps the stored unsigned value back to signed range then scales:
        float_val = (int4_val - 8) * scale

    The zero-point of 8 maps unsigned [0,15] to signed [-8,+7].

    Args:
        int4_val: Unsigned 4-bit value (0-15)
        scale: Per-position quantization scale factor

    Returns:
        Dequantized float32 value
    """
    return (int4_val.to(tl.float32) - 8.0) * scale


@triton.jit
def _popcount_mod2_24bit(x):
    """Compute popcount(x) mod 2 for 24-bit value using XOR reduction."""
    x = x ^ (x >> 16)
    x = x ^ (x >> 8)
    x = x ^ (x >> 4)
    x = x ^ (x >> 2)
    x = x ^ (x >> 1)
    return x & 1


@triton.jit
def decode_golay_inline(
    codeword,
    syndrome_lut_ptr,
    H0: tl.constexpr,
    H1: tl.constexpr,
    H2: tl.constexpr,
    H3: tl.constexpr,
    H4: tl.constexpr,
    H5: tl.constexpr,
    H6: tl.constexpr,
    H7: tl.constexpr,
    H8: tl.constexpr,
    H9: tl.constexpr,
    H10: tl.constexpr,
    H11: tl.constexpr,
):
    """
    Inline Golay(24,12) decode for 3-error correction.

    Decodes a 24-bit Golay codeword containing 3 packed INT4 values. Golay(24,12)
    can correct up to 3 arbitrary bit errors, making it suitable for high-BER
    environments or high-value "sink" tokens in adaptive UEP schemes.

    Algorithm:
        1. Compute 12-bit syndrome via parity-check matrix multiplication
        2. Look up error pattern in precomputed 4096-entry syndrome table
        3. XOR error pattern to correct (if pattern is valid)
        4. Extract three 4-bit nibbles from corrected 12 data bits

    Args:
        codeword: 24-bit Golay codeword (12 data + 12 parity)
        syndrome_lut_ptr: Pointer to precomputed syndrome → error pattern table
        H0-H11: Rows of parity-check matrix (constexpr for performance)

    Returns:
        n0, n1, n2: Three decoded 4-bit INT4 values (0-15 each).
                    Returns 0 for each nibble if error is uncorrectable (≥4 errors).
    """
    # Compute 12-bit syndrome via parity checks (H matrix rows as bitmasks)
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
        (s0 << 0)
        | (s1 << 1)
        | (s2 << 2)
        | (s3 << 3)
        | (s4 << 4)
        | (s5 << 5)
        | (s6 << 6)
        | (s7 << 7)
        | (s8 << 8)
        | (s9 << 9)
        | (s10 << 10)
        | (s11 << 11)
    )

    error_pattern = tl.load(syndrome_lut_ptr + syndrome)
    is_correctable = error_pattern >= 0
    corrected = tl.where(is_correctable, codeword ^ error_pattern, codeword)
    data_12bit = corrected & 0xFFF
    n0 = (data_12bit >> 0) & 0xF
    n1 = (data_12bit >> 4) & 0xF
    n2 = (data_12bit >> 8) & 0xF
    n0 = tl.where(is_correctable, n0, 0)
    n1 = tl.where(is_correctable, n1, 0)
    n2 = tl.where(is_correctable, n2, 0)

    return n0, n1, n2


@triton.jit
def paged_attention_ecc_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    context_lens_ptr,
    k_scales_ptr,
    v_scales_ptr,
    batch_size,
    num_heads,
    head_dim,
    num_layers,
    layer_idx,
    block_size,
    max_blocks,
    max_context_len,
    sm_scale,
    lut0: tl.constexpr,
    lut1: tl.constexpr,
    lut2: tl.constexpr,
    lut3: tl.constexpr,
    lut4: tl.constexpr,
    lut5: tl.constexpr,
    lut6: tl.constexpr,
    lut7: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    MAX_CONTEXT_BLOCKS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
):
    """
    Paged attention kernel with fused Hamming(8,4) ECC decoding.

    This kernel computes scaled dot-product attention over an ECC-protected KV cache
    using online softmax. Each thread block processes one (batch, head) pair, iterating
    over all context tokens to accumulate the attention-weighted output.

    Algorithm (Online Softmax - Numerically Stable):
        For each token i in context:
            1. Load and decode K[i] from Hamming(8,4) codeword
            2. Compute QK score: score[i] = Q · K[i] * sm_scale
            3. Update running max: m_new = max(m_old, score[i])
            4. Rescale accumulator: acc = acc * exp(m_old - m_new)
            5. Compute attention weight: w[i] = exp(score[i] - m_new)
            6. Load and decode V[i], accumulate: acc += w[i] * V[i]
            7. Update normalizer: l = l * exp(m_old - m_new) + w[i]
        Output = acc / l

    Grid: (batch_size * num_heads,)
        - Each program handles one (batch, head) pair
        - Parallelism is across batch and heads, not context tokens

    Memory Layout:
        k_cache: [num_blocks, num_layers, num_heads, block_size * head_dim]
                 where each element is a Hamming(8,4) codeword (uint8)
        k_scales: [num_blocks, num_layers, num_heads, block_size]
                  per-token quantization scale factors

    Performance Notes:
        - ECC decode dominates compute; tiled variants don't help
        - ~20x faster than Python loop with separate decode calls
        - Bottleneck is bit manipulation, not memory bandwidth
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_heads
    head_idx = pid % num_heads

    head_offsets = tl.arange(0, BLOCK_HEAD_DIM)
    head_mask = head_offsets < head_dim

    context_len = tl.load(context_lens_ptr + batch_idx)

    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim
    q = tl.load(query_ptr + q_offset + head_offsets, mask=head_mask, other=0.0)

    # Online softmax state: m = running max, l = normalizer sum, acc = weighted sum
    # Initialize m to -inf equivalent so first real score dominates
    m_i = -1e20  # -inf approximation; below any valid attention score
    l_i = 0.0
    acc = tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32)

    for block_idx in range(MAX_CONTEXT_BLOCKS):
        start_pos = block_idx * KV_BLOCK_SIZE
        block_valid = start_pos < context_len

        physical_block = tl.load(
            block_table_ptr + batch_idx * max_blocks + block_idx,
            mask=block_valid,
            other=-1,
        )

        for slot in range(KV_BLOCK_SIZE):
            token_pos = start_pos + slot
            token_valid = (
                block_valid & (token_pos < context_len) & (physical_block >= 0)
            )

            k_base_offset = (
                physical_block * num_layers * num_heads * KV_BLOCK_SIZE * head_dim
                + layer_idx * num_heads * KV_BLOCK_SIZE * head_dim
                + head_idx * KV_BLOCK_SIZE * head_dim
                + slot * head_dim
            )

            load_mask = head_mask & token_valid
            k_encoded = tl.load(
                k_cache_ptr + k_base_offset + head_offsets, mask=load_mask, other=0
            ).to(tl.uint8)

            k_int4 = decode_hamming84_inline(
                k_encoded, lut0, lut1, lut2, lut3, lut4, lut5, lut6, lut7
            )

            scale_offset = (
                physical_block * num_layers * num_heads * KV_BLOCK_SIZE
                + layer_idx * num_heads * KV_BLOCK_SIZE
                + head_idx * KV_BLOCK_SIZE
                + slot
            )
            k_scale = tl.load(k_scales_ptr + scale_offset, mask=token_valid, other=1.0)

            k_float = dequantize_int4(k_int4, k_scale)

            qk_partial = q * k_float
            qk_score = tl.sum(qk_partial, axis=0) * sm_scale

            qk_score = tl.where(token_valid, qk_score, -1e20)

            m_new = tl.maximum(m_i, qk_score)

            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(qk_score - m_new)

            l_new = alpha * l_i + beta

            v_base_offset = (
                physical_block * num_layers * num_heads * KV_BLOCK_SIZE * head_dim
                + layer_idx * num_heads * KV_BLOCK_SIZE * head_dim
                + head_idx * KV_BLOCK_SIZE * head_dim
                + slot * head_dim
            )

            v_encoded = tl.load(
                v_cache_ptr + v_base_offset + head_offsets, mask=load_mask, other=0
            ).to(tl.uint8)

            v_int4 = decode_hamming84_inline(
                v_encoded, lut0, lut1, lut2, lut3, lut4, lut5, lut6, lut7
            )

            v_scale = tl.load(v_scales_ptr + scale_offset, mask=token_valid, other=1.0)
            v_float = dequantize_int4(v_int4, v_scale)

            acc = alpha * acc + beta * v_float

            m_i = m_new
            l_i = l_new

    output = tl.where(l_i > 0, acc / l_i, tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32))

    out_offset = batch_idx * num_heads * head_dim + head_idx * head_dim
    tl.store(output_ptr + out_offset + head_offsets, output, mask=head_mask)


@triton.jit
def paged_attention_ecc_tiled_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    context_lens_ptr,
    k_scales_ptr,
    v_scales_ptr,
    batch_size,
    num_heads,
    head_dim,
    num_layers,
    layer_idx,
    block_size,
    max_blocks,
    max_context_len,
    sm_scale,
    lut0: tl.constexpr,
    lut1: tl.constexpr,
    lut2: tl.constexpr,
    lut3: tl.constexpr,
    lut4: tl.constexpr,
    lut5: tl.constexpr,
    lut6: tl.constexpr,
    lut7: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    MAX_CONTEXT_BLOCKS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,  # Tokens per tile (4, 8, or 16)
):
    """
    Tiled paged attention kernel with ECC decoding (EXPERIMENTAL).

    NOTE: Benchmarking shows this tiled kernel is SLOWER than the original
    token-by-token kernel for ECC-protected attention. The reason is that:
    1. ECC decoding (bit manipulation) dominates compute, not softmax rescaling
    2. Tiling requires either storing QK scores (register pressure) or
       re-computing K (2x memory loads)
    3. Standard Flash Attention assumptions don't apply to ECC workloads

    This kernel is kept for correctness testing and future optimization attempts,
    but use_tiled=False (the default) is recommended for production.

    Processes BLOCK_M tokens per inner iteration instead of 1, enabling:
    - Tile-wise online softmax updates (fewer rescale operations)
    - Potential for better memory coalescing (not realized in practice)
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_heads
    head_idx = pid % num_heads

    head_offsets = tl.arange(0, BLOCK_HEAD_DIM)
    head_mask = head_offsets < head_dim

    context_len = tl.load(context_lens_ptr + batch_idx)

    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim
    q = tl.load(query_ptr + q_offset + head_offsets, mask=head_mask, other=0.0)

    # Online softmax accumulators
    m_i = -1e20
    l_i = 0.0
    acc = tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32)

    # Process tiles of BLOCK_M tokens
    num_tiles = tl.cdiv(KV_BLOCK_SIZE, BLOCK_M)

    for block_idx in range(MAX_CONTEXT_BLOCKS):
        start_pos = block_idx * KV_BLOCK_SIZE
        block_valid = start_pos < context_len

        physical_block = tl.load(
            block_table_ptr + batch_idx * max_blocks + block_idx,
            mask=block_valid,
            other=-1,
        )

        # Process BLOCK_M tokens at a time within this KV block
        for tile_idx in range(num_tiles):
            tile_start = tile_idx * BLOCK_M

            # First pass: find max QK score in tile (for stable softmax)
            m_tile = -1e20
            for m in range(BLOCK_M):
                slot = tile_start + m
                token_pos = start_pos + slot
                token_valid = (
                    block_valid & (token_pos < context_len) & (physical_block >= 0)
                    & (slot < KV_BLOCK_SIZE)
                )

                k_base_offset = (
                    physical_block * num_layers * num_heads * KV_BLOCK_SIZE * head_dim
                    + layer_idx * num_heads * KV_BLOCK_SIZE * head_dim
                    + head_idx * KV_BLOCK_SIZE * head_dim
                    + slot * head_dim
                )

                load_mask = head_mask & token_valid
                k_encoded = tl.load(
                    k_cache_ptr + k_base_offset + head_offsets, mask=load_mask, other=0
                ).to(tl.uint8)

                k_int4 = decode_hamming84_inline(
                    k_encoded, lut0, lut1, lut2, lut3, lut4, lut5, lut6, lut7
                )

                scale_offset = (
                    physical_block * num_layers * num_heads * KV_BLOCK_SIZE
                    + layer_idx * num_heads * KV_BLOCK_SIZE
                    + head_idx * KV_BLOCK_SIZE
                    + slot
                )
                k_scale = tl.load(k_scales_ptr + scale_offset, mask=token_valid, other=1.0)

                k_float = dequantize_int4(k_int4, k_scale)

                qk_partial = q * k_float
                qk_score = tl.sum(qk_partial, axis=0) * sm_scale
                qk_score = tl.where(token_valid, qk_score, -1e20)

                m_tile = tl.maximum(m_tile, qk_score)

            # Update global max and rescale accumulator ONCE per tile
            m_new = tl.maximum(m_i, m_tile)
            alpha = tl.exp(m_i - m_new)
            acc = alpha * acc
            l_i = alpha * l_i

            # Second pass: accumulate with correct softmax weights
            for m in range(BLOCK_M):
                slot = tile_start + m
                token_pos = start_pos + slot
                token_valid = (
                    block_valid & (token_pos < context_len) & (physical_block >= 0)
                    & (slot < KV_BLOCK_SIZE)
                )

                kv_base_offset = (
                    physical_block * num_layers * num_heads * KV_BLOCK_SIZE * head_dim
                    + layer_idx * num_heads * KV_BLOCK_SIZE * head_dim
                    + head_idx * KV_BLOCK_SIZE * head_dim
                    + slot * head_dim
                )

                load_mask = head_mask & token_valid

                # Re-compute K score (trades compute for register pressure)
                k_encoded = tl.load(
                    k_cache_ptr + kv_base_offset + head_offsets, mask=load_mask, other=0
                ).to(tl.uint8)
                k_int4 = decode_hamming84_inline(
                    k_encoded, lut0, lut1, lut2, lut3, lut4, lut5, lut6, lut7
                )

                scale_offset = (
                    physical_block * num_layers * num_heads * KV_BLOCK_SIZE
                    + layer_idx * num_heads * KV_BLOCK_SIZE
                    + head_idx * KV_BLOCK_SIZE
                    + slot
                )
                k_scale = tl.load(k_scales_ptr + scale_offset, mask=token_valid, other=1.0)

                k_float = dequantize_int4(k_int4, k_scale)
                qk_score = tl.sum(q * k_float, axis=0) * sm_scale
                qk_score = tl.where(token_valid, qk_score, -1e20)

                # Load and decode V
                v_encoded = tl.load(
                    v_cache_ptr + kv_base_offset + head_offsets, mask=load_mask, other=0
                ).to(tl.uint8)
                v_int4 = decode_hamming84_inline(
                    v_encoded, lut0, lut1, lut2, lut3, lut4, lut5, lut6, lut7
                )
                v_scale = tl.load(v_scales_ptr + scale_offset, mask=token_valid, other=1.0)
                v_float = dequantize_int4(v_int4, v_scale)

                beta = tl.exp(qk_score - m_new)
                acc = acc + beta * v_float
                l_i = l_i + tl.where(token_valid, beta, 0.0)

            m_i = m_new

    output = tl.where(l_i > 0, acc / l_i, tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32))

    out_offset = batch_idx * num_heads * head_dim + head_idx * head_dim
    tl.store(output_ptr + out_offset + head_offsets, output, mask=head_mask)


def paged_attention_ecc(
    query,
    k_cache,
    v_cache,
    block_table,
    context_lens,
    k_scales,
    layer_idx,
    block_size,
    sm_scale=None,
    codec="hamming84",
    syndrome_table=None,
    use_tiled=False,
    block_m=4,
    v_scales=None,
):
    """
    Paged attention with ECC-protected KV cache.

    Args:
        query: Query tensor [batch_size, num_heads, head_dim]
        k_cache: Key cache [num_blocks, num_layers, num_heads, block_size * head_dim]
        v_cache: Value cache [num_blocks, num_layers, num_heads, block_size * head_dim]
        block_table: Physical block mapping [batch_size, max_blocks]
        context_lens: Context lengths [batch_size]
        k_scales: Key quantization scales [num_blocks, num_layers, num_heads, block_size]
        layer_idx: Current layer index
        block_size: Tokens per KV block
        sm_scale: Softmax scale (default: 1/sqrt(head_dim))
        codec: ECC codec ("hamming84" or "golay")
        syndrome_table: Optional syndrome table for Golay
        use_tiled: Use tiled kernel (default: False, NOT recommended - see note)
        block_m: Tokens per tile for tiled kernel (default: 4)
        v_scales: Value quantization scales (if None, uses k_scales for both)

    Note:
        The tiled kernel was implemented to reduce softmax rescaling overhead,
        but benchmarking shows it's actually SLOWER than the original kernel.
        ECC decoding dominates the computation, making tiling ineffective.
        Keep use_tiled=False for best performance.

    Returns:
        Output tensor [batch_size, num_heads, head_dim]
    """
    assert query.is_cuda, "Query must be on CUDA"

    # Handle backward compatibility: if v_scales not provided, use k_scales for both
    if v_scales is None:
        v_scales = k_scales

    batch_size, num_heads, head_dim = query.shape
    num_blocks, num_layers, _, codewords_per_head = k_cache.shape
    max_blocks = block_table.shape[1]
    max_context_len = max_blocks * block_size

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    output = torch.empty_like(query)

    if codec == "hamming84":
        assert k_cache.dtype == torch.uint8, "Hamming84 K cache must be uint8"
        assert v_cache.dtype == torch.uint8, "Hamming84 V cache must be uint8"

        lut = SYNDROME_LUT_HAMMING84

        BLOCK_HEAD_DIM = triton.next_power_of_2(head_dim)

        MAX_CONTEXT_BLOCKS = max_blocks
        KV_BLOCK_SIZE = block_size

        grid = (batch_size * num_heads,)

        if use_tiled and block_size >= block_m:
            # Use tiled kernel for better parallelism
            paged_attention_ecc_tiled_kernel[grid](
                output,
                query,
                k_cache,
                v_cache,
                block_table,
                context_lens,
                k_scales,
                v_scales,
                batch_size,
                num_heads,
                head_dim,
                num_layers,
                layer_idx,
                block_size,
                max_blocks,
                max_context_len,
                sm_scale,
                int(lut[0]),
                int(lut[1]),
                int(lut[2]),
                int(lut[3]),
                int(lut[4]),
                int(lut[5]),
                int(lut[6]),
                int(lut[7]),
                BLOCK_HEAD_DIM=BLOCK_HEAD_DIM,
                MAX_CONTEXT_BLOCKS=MAX_CONTEXT_BLOCKS,
                KV_BLOCK_SIZE=KV_BLOCK_SIZE,
                BLOCK_M=block_m,
            )
        else:
            # Use original token-by-token kernel
            paged_attention_ecc_kernel[grid](
                output,
                query,
                k_cache,
                v_cache,
                block_table,
                context_lens,
                k_scales,
                v_scales,
                batch_size,
                num_heads,
                head_dim,
                num_layers,
                layer_idx,
                block_size,
                max_blocks,
                max_context_len,
                sm_scale,
                int(lut[0]),
                int(lut[1]),
                int(lut[2]),
                int(lut[3]),
                int(lut[4]),
                int(lut[5]),
                int(lut[6]),
                int(lut[7]),
                BLOCK_HEAD_DIM=BLOCK_HEAD_DIM,
                MAX_CONTEXT_BLOCKS=MAX_CONTEXT_BLOCKS,
                KV_BLOCK_SIZE=KV_BLOCK_SIZE,
            )

    elif codec == "golay":
        assert k_cache.dtype == torch.int32, "Golay K cache must be int32"
        assert v_cache.dtype == torch.int32, "Golay V cache must be int32"

        output = reference_attention_ecc(
            query,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            k_scales,
            layer_idx,
            block_size,
            head_dim,
            sm_scale,
            codec="golay",
        )

    else:
        raise ValueError(f"Unknown codec: {codec}")

    return output


def reference_attention_ecc(
    query,
    k_cache,
    v_cache,
    block_table,
    context_lens,
    scales,
    layer_idx,
    block_size,
    head_dim,
    sm_scale=None,
    codec="hamming84",
):
    batch_size, num_heads, _ = query.shape
    device = query.device

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    outputs = []

    for b in range(batch_size):
        ctx_len = int(context_lens[b].item())
        if ctx_len <= 0:
            outputs.append(torch.zeros(num_heads, head_dim, device=device))
            continue

        num_ctx_blocks = (ctx_len + block_size - 1) // block_size

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

            for slot in range(tokens_in_block):
                if codec == "hamming84":
                    k_offset_start = slot * head_dim
                    k_offset_end = k_offset_start + head_dim

                    k_encoded = k_cache[
                        phys_block, layer_idx, :, k_offset_start:k_offset_end
                    ]
                    v_encoded = v_cache[
                        phys_block, layer_idx, :, k_offset_start:k_offset_end
                    ]

                    k_decoded, _ = hamming84_decode(k_encoded.flatten())
                    v_decoded, _ = hamming84_decode(v_encoded.flatten())
                    k_decoded = k_decoded.view(num_heads, head_dim)
                    v_decoded = v_decoded.view(num_heads, head_dim)

                elif codec == "golay":
                    codewords_per_slot = (head_dim + 2) // 3

                    k_offset_start = slot * codewords_per_slot
                    k_offset_end = k_offset_start + codewords_per_slot

                    k_encoded = k_cache[
                        phys_block, layer_idx, :, k_offset_start:k_offset_end
                    ]
                    v_encoded = v_cache[
                        phys_block, layer_idx, :, k_offset_start:k_offset_end
                    ]

                    k_decoded_list = []
                    v_decoded_list = []
                    for h in range(num_heads):
                        k_triplets, _ = golay_decode(k_encoded[h])
                        v_triplets, _ = golay_decode(v_encoded[h])

                        k_vals = k_triplets.flatten()[:head_dim]
                        v_vals = v_triplets.flatten()[:head_dim]
                        k_decoded_list.append(k_vals)
                        v_decoded_list.append(v_vals)

                    k_decoded = torch.stack(k_decoded_list, dim=0)
                    v_decoded = torch.stack(v_decoded_list, dim=0)

                else:
                    k_offset_start = slot * head_dim
                    k_offset_end = k_offset_start + head_dim
                    k_decoded = k_cache[
                        phys_block, layer_idx, :, k_offset_start:k_offset_end
                    ]
                    v_decoded = v_cache[
                        phys_block, layer_idx, :, k_offset_start:k_offset_end
                    ]

                slot_scale = scales[phys_block, layer_idx, :, slot]

                k_list.append(k_decoded)
                v_list.append(v_decoded)
                scale_list.append(slot_scale)

        if not k_list:
            outputs.append(torch.zeros(num_heads, head_dim, device=device))
            continue

        k_stacked = torch.stack(k_list, dim=0)
        v_stacked = torch.stack(v_list, dim=0)
        scale_stacked = torch.stack(scale_list, dim=0)

        k_float = (k_stacked.float() - 8.0) * scale_stacked.unsqueeze(-1)
        v_float = (v_stacked.float() - 8.0) * scale_stacked.unsqueeze(-1)

        k_float = k_float.permute(1, 0, 2)
        v_float = v_float.permute(1, 0, 2)

        q = query[b]

        scores = torch.einsum("hd,hcd->hc", q, k_float) * sm_scale

        attn_weights = torch.softmax(scores, dim=-1)

        out = torch.einsum("hc,hcd->hd", attn_weights, v_float)

        outputs.append(out)

    return torch.stack(outputs, dim=0)


def verify_attention_kernel():
    print("Attention ECC Kernel Verification")
    print("=" * 60)

    batch_size = 2
    num_heads = 4
    head_dim = 64
    num_layers = 2
    layer_idx = 0
    block_size = 16
    context_len = 48

    num_blocks = 32
    max_blocks = 8

    print(f"Config: batch={batch_size}, heads={num_heads}, head_dim={head_dim}")
    print(f"Context: {context_len} tokens, block_size={block_size}")

    device = "cuda"

    torch.manual_seed(42)
    query = torch.randn(
        batch_size, num_heads, head_dim, device=device, dtype=torch.float32
    )

    block_table = torch.full(
        (batch_size, max_blocks), -1, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        num_ctx_blocks = (context_len + block_size - 1) // block_size
        for i in range(num_ctx_blocks):
            block_table[b, i] = b * num_ctx_blocks + i

    context_lens = torch.full(
        (batch_size,), context_len, dtype=torch.int32, device=device
    )

    codewords_per_head = block_size * head_dim
    k_cache = torch.zeros(
        num_blocks,
        num_layers,
        num_heads,
        codewords_per_head,
        dtype=torch.uint8,
        device=device,
    )
    v_cache = torch.zeros(
        num_blocks,
        num_layers,
        num_heads,
        codewords_per_head,
        dtype=torch.uint8,
        device=device,
    )
    scales = torch.zeros(
        num_blocks,
        num_layers,
        num_heads,
        block_size,
        dtype=torch.float32,
        device=device,
    )

    for b in range(batch_size):
        num_ctx_blocks = (context_len + block_size - 1) // block_size
        for blk_idx in range(num_ctx_blocks):
            phys_block = int(block_table[b, blk_idx].item())

            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, context_len)
            tokens_in_block = end_pos - start_pos

            for slot in range(tokens_in_block):
                k_fp = torch.randn(num_heads, head_dim, device=device)
                v_fp = torch.randn(num_heads, head_dim, device=device)

                k_scale = k_fp.abs().max(dim=-1).values / 7.0
                v_scale = v_fp.abs().max(dim=-1).values / 7.0
                scale = torch.maximum(k_scale, v_scale)
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)

                k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)
                v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)

                k_encoded = hamming84_encode(k_int4.flatten()).view(num_heads, head_dim)
                v_encoded = hamming84_encode(v_int4.flatten()).view(num_heads, head_dim)

                offset_start = slot * head_dim
                offset_end = offset_start + head_dim
                k_cache[phys_block, layer_idx, :, offset_start:offset_end] = k_encoded
                v_cache[phys_block, layer_idx, :, offset_start:offset_end] = v_encoded
                scales[phys_block, layer_idx, :, slot] = scale

    print("\nRunning reference implementation...")
    ref_output = reference_attention_ecc(
        query,
        k_cache,
        v_cache,
        block_table,
        context_lens,
        scales,
        layer_idx,
        block_size,
        head_dim,
    )
    print(f"Reference output shape: {ref_output.shape}")

    print("\nRunning Triton kernel...")
    triton_output = paged_attention_ecc(
        query,
        k_cache,
        v_cache,
        block_table,
        context_lens,
        scales,
        layer_idx,
        block_size,
    )
    print(f"Triton output shape: {triton_output.shape}")

    diff = (ref_output - triton_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nMax absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")

    tolerance = 1e-4
    if max_diff < tolerance:
        print(f"\n✓ Verification PASSED (max_diff < {tolerance})")
        return True
    else:
        print(f"\n✗ Verification FAILED (max_diff >= {tolerance})")
        return False


if __name__ == "__main__":
    verify_attention_kernel()
