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
    corrected = tl.where(is_double_error, 0, corrected)
    decoded = corrected & 0x0F
    return decoded


@triton.jit
def dequantize_int4(int4_val, scale):
    return (int4_val.to(tl.float32) - 8.0) * scale


@triton.jit
def _popcount_mod2_24bit(x):
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
    scales_ptr,
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
    pid = tl.program_id(0)
    batch_idx = pid // num_heads
    head_idx = pid % num_heads

    head_offsets = tl.arange(0, BLOCK_HEAD_DIM)
    head_mask = head_offsets < head_dim

    context_len = tl.load(context_lens_ptr + batch_idx)

    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim
    q = tl.load(query_ptr + q_offset + head_offsets, mask=head_mask, other=0.0)

    m_i = -1e20
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
            scale = tl.load(scales_ptr + scale_offset, mask=token_valid, other=1.0)

            k_float = dequantize_int4(k_int4, scale)

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

            v_float = dequantize_int4(v_int4, scale)

            acc = alpha * acc + beta * v_float

            m_i = m_new
            l_i = l_new

    output = tl.where(l_i > 0, acc / l_i, tl.zeros([BLOCK_HEAD_DIM], dtype=tl.float32))

    out_offset = batch_idx * num_heads * head_dim + head_idx * head_dim
    tl.store(output_ptr + out_offset + head_offsets, output, mask=head_mask)


def paged_attention_ecc(
    query,
    k_cache,
    v_cache,
    block_table,
    context_lens,
    scales,
    layer_idx,
    block_size,
    sm_scale=None,
    codec="hamming84",
    syndrome_table=None,
):
    assert query.is_cuda, "Query must be on CUDA"

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
            scales,
            layer_idx,
            block_size,
            head_dim,
            sm_scale,
            codec="golay",
        )

    else:
        raise ValueError(f"Unknown codec: {codec}")

    return output


def paged_attention_ecc_adaptive(
    query,
    k_cache,
    v_cache,
    sink_k_cache,
    sink_v_cache,
    block_table,
    context_lens,
    scales,
    sink_scales=None,
    layer_idx=0,
    block_size=16,
    sink_boundary=4,
    sm_scale=None,
):
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
        codec="hamming84",
        sink_boundary=sink_boundary,
        sink_k_cache=sink_k_cache,
        sink_v_cache=sink_v_cache,
        sink_scales=sink_scales,
    )


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
    sink_boundary=0,
    sink_k_cache=None,
    sink_v_cache=None,
    sink_scales=None,
):
    batch_size, num_heads, _ = query.shape
    device = query.device

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    adaptive_uep = (
        sink_boundary > 0 and sink_k_cache is not None and sink_v_cache is not None
    )

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
                    k_offset_start = slot * head_dim
                    k_offset_end = k_offset_start + head_dim

                    k_encoded = block_k_cache[
                        phys_block, layer_idx, :, k_offset_start:k_offset_end
                    ]
                    v_encoded = block_v_cache[
                        phys_block, layer_idx, :, k_offset_start:k_offset_end
                    ]

                    k_decoded, _ = hamming84_decode(k_encoded.flatten())
                    v_decoded, _ = hamming84_decode(v_encoded.flatten())
                    k_decoded = k_decoded.view(num_heads, head_dim)
                    v_decoded = v_decoded.view(num_heads, head_dim)

                elif block_codec == "golay":
                    codewords_per_slot = (head_dim + 2) // 3

                    k_offset_start = slot * codewords_per_slot
                    k_offset_end = k_offset_start + codewords_per_slot

                    k_encoded = block_k_cache[
                        phys_block, layer_idx, :, k_offset_start:k_offset_end
                    ]
                    v_encoded = block_v_cache[
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
                    k_decoded = block_k_cache[
                        phys_block, layer_idx, :, k_offset_start:k_offset_end
                    ]
                    v_decoded = block_v_cache[
                        phys_block, layer_idx, :, k_offset_start:k_offset_end
                    ]

                slot_scale = block_scales[phys_block, layer_idx, :, slot]

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
