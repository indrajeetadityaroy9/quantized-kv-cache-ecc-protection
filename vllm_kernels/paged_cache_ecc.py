import torch
import triton
import triton.language as tl

from hamming74.triton_kernels import hamming84_encode, golay_encode, hamming84_decode
from hamming74.triton_kernels.config import (
    HAMMING84_BLOCK_SIZE,
    GOLAY_BLOCK_SIZE,
)


@triton.jit
def quantize_symmetric_int4(
    values,
    scale,
):
    quantized = tl.math.round(values / scale)

    quantized = tl.maximum(tl.minimum(quantized, 7.0), -8.0)
    int4_val = (quantized + 8.0).to(tl.uint8)

    return int4_val


@triton.jit
def dequantize_symmetric_int4(
    int4_val,
    scale,
):
    return (int4_val.to(tl.float32) - 8.0) * scale


@triton.jit
def encode_hamming84_inline(int4_val):
    d0 = (int4_val >> 0) & 1
    d1 = (int4_val >> 1) & 1
    d2 = (int4_val >> 2) & 1
    d3 = (int4_val >> 3) & 1

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
    return codeword


@triton.jit
def _popcount_mod2_12bit(x):
    x = x ^ (x >> 8)
    x = x ^ (x >> 4)
    x = x ^ (x >> 2)
    x = x ^ (x >> 1)
    return x & 1


@triton.jit
def encode_golay_triplet_inline(n0, n1, n2):
    C0 = 2619
    C1 = 3357
    C2 = 3726
    C3 = 2887
    C4 = 3491
    C5 = 3793
    C6 = 3944
    C7 = 2996
    C8 = 2522
    C9 = 2285
    C10 = 3190
    C11 = 2047

    data_12bit = (n0 & 0xF) | ((n1 & 0xF) << 4) | ((n2 & 0xF) << 8)

    p0 = _popcount_mod2_12bit(data_12bit & C0)
    p1 = _popcount_mod2_12bit(data_12bit & C1)
    p2 = _popcount_mod2_12bit(data_12bit & C2)
    p3 = _popcount_mod2_12bit(data_12bit & C3)
    p4 = _popcount_mod2_12bit(data_12bit & C4)
    p5 = _popcount_mod2_12bit(data_12bit & C5)
    p6 = _popcount_mod2_12bit(data_12bit & C6)
    p7 = _popcount_mod2_12bit(data_12bit & C7)
    p8 = _popcount_mod2_12bit(data_12bit & C8)
    p9 = _popcount_mod2_12bit(data_12bit & C9)
    p10 = _popcount_mod2_12bit(data_12bit & C10)
    p11 = _popcount_mod2_12bit(data_12bit & C11)

    parity_12bit = (
        (p0 << 0)
        | (p1 << 1)
        | (p2 << 2)
        | (p3 << 3)
        | (p4 << 4)
        | (p5 << 5)
        | (p6 << 6)
        | (p7 << 7)
        | (p8 << 8)
        | (p9 << 9)
        | (p10 << 10)
        | (p11 << 11)
    )

    codeword = data_12bit | (parity_12bit << 12)
    return codeword.to(tl.int32)


@triton.jit
def write_kv_cache_hamming84_kernel(
    kv_ptr,
    cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    scales_ptr,
    batch_size,
    seq_len,
    hidden_size,
    num_heads,
    head_size,
    block_size,
    layer_idx,
    max_blocks,
    codewords_per_head,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // (num_heads * seq_len)
    remainder = pid % (num_heads * seq_len)
    head_idx = remainder // seq_len
    pos = remainder % seq_len

    if batch_idx >= batch_size:
        return
    actual_seq_len = tl.load(seq_lens_ptr + batch_idx)
    if pos >= actual_seq_len:
        return

    logical_block = pos // block_size
    slot_in_block = pos % block_size

    physical_block = tl.load(block_table_ptr + batch_idx * max_blocks + logical_block)
    if physical_block < 0:
        return

    scale = tl.load(scales_ptr + batch_idx * seq_len + pos)

    head_start = head_idx * head_size
    for elem_idx in range(head_size):
        kv_offset = (
            batch_idx * seq_len * hidden_size
            + pos * hidden_size
            + head_start
            + elem_idx
        )
        value_fp16 = tl.load(kv_ptr + kv_offset).to(tl.float32)

        int4_val = quantize_symmetric_int4(value_fp16, scale)

        codeword = encode_hamming84_inline(int4_val)

        slot_offset = slot_in_block * head_size + elem_idx
        cache_offset = (
            physical_block * (32 * num_heads * codewords_per_head)
            + layer_idx * (num_heads * codewords_per_head)
            + head_idx * codewords_per_head
            + slot_offset
        )

        tl.store(cache_ptr + cache_offset, codeword)


def write_kv_to_cache_hamming84(
    kv,
    cache,
    block_table,
    seq_lens,
    scales,
    layer_idx,
    num_heads,
    head_size,
    block_size,
):
    batch_size, seq_len, hidden_size = kv.shape
    max_blocks = block_table.shape[1]
    codewords_per_head = block_size * head_size

    total_items = batch_size * num_heads * seq_len

    grid = lambda meta: (total_items,)
    write_kv_cache_hamming84_kernel[grid](
        kv,
        cache,
        block_table,
        seq_lens,
        scales,
        batch_size,
        seq_len,
        hidden_size,
        num_heads,
        head_size,
        block_size,
        layer_idx,
        max_blocks,
        codewords_per_head,
        BLOCK_SIZE=1,
    )


def compute_quantization_scales(
    tensor,
    dim=-1,
):
    abs_max = tensor.abs().max(dim=dim, keepdim=False).values
    scales = abs_max / 7.0

    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    return scales


def write_kv_to_cache_simple(
    kv,
    codec="hamming84",
    scale=None,
):
    if scale is None:
        scale = compute_quantization_scales(kv, dim=-1)

    scale_expanded = scale.unsqueeze(-1)
    quantized = torch.round(kv / scale_expanded).clamp(-8, 7) + 8
    int4_vals = quantized.to(torch.uint8)

    if codec == "hamming84":
        flat = int4_vals.flatten()
        encoded = hamming84_encode(flat.cuda() if not flat.is_cuda else flat)
        encoded = encoded.view(int4_vals.shape)

    elif codec == "golay":
        flat = int4_vals.flatten()

        pad_len = (3 - flat.numel() % 3) % 3
        if pad_len > 0:
            flat = torch.cat(
                [flat, torch.zeros(pad_len, dtype=flat.dtype, device=flat.device)]
            )
        triplets = flat.view(-1, 3)
        if not triplets.is_cuda:
            triplets = triplets.cuda()
        encoded = golay_encode(triplets)

    else:
        encoded = int4_vals

    return encoded, scale


def verify_cache_write():
    print("Cache Write Verification")
    print("=" * 60)

    batch_size = 2
    seq_len = 64
    hidden_size = 256

    kv = torch.randn(
        batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16
    )

    encoded_h84, scales = write_kv_to_cache_simple(kv, codec="hamming84")
    print(f"Hamming84 encoded shape: {encoded_h84.shape}")
    print(f"Hamming84 encoded dtype: {encoded_h84.dtype}")

    flat_encoded = encoded_h84.flatten()
    decoded, stats = hamming84_decode(flat_encoded)
    print(f"Decoded shape: {decoded.shape}")
    print(f"Errors corrected: {stats[0]}, detected: {stats[1]}")

    decoded_reshaped = decoded.view(encoded_h84.shape).float()
    dequantized = (decoded_reshaped - 8) * scales.unsqueeze(-1).cuda()
    mse = ((kv.float() - dequantized) ** 2).mean()
    print(f"Reconstruction MSE: {mse:.6f}")

    encoded_golay, _ = write_kv_to_cache_simple(kv, codec="golay", scale=scales)
    print(f"\nGolay encoded shape: {encoded_golay.shape}")
    print(f"Golay encoded dtype: {encoded_golay.dtype}")

    print("\nCache write verification passed!")
    return True


if __name__ == "__main__":
    verify_cache_write()
