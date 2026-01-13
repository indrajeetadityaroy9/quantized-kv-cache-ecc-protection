import pytest
import torch
import math


class TestAttentionHamming84:
    def test_single_block_clean_data(self):
        from kv_cache.attention_ecc import (
            paged_attention_ecc,
            reference_attention_ecc,
        )
        from ecc_codecs.triton_kernels import hamming84_encode

        batch_size = 1
        num_heads = 2
        head_dim = 32
        num_layers = 1
        layer_idx = 0
        block_size = 16
        context_len = 12

        num_blocks = 4
        max_blocks = 4
        codewords_per_head = block_size * head_dim

        device = "cuda"
        torch.manual_seed(42)

        query = torch.randn(
            batch_size, num_heads, head_dim, device=device, dtype=torch.float32
        )

        block_table = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device
        )
        block_table[0, 0] = 0

        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

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

        for slot in range(context_len):
            k_fp = torch.randn(num_heads, head_dim, device=device)
            v_fp = torch.randn(num_heads, head_dim, device=device)

            scale = (
                torch.maximum(
                    k_fp.abs().max(dim=-1).values, v_fp.abs().max(dim=-1).values
                )
                / 7.0
            )
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)

            k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(
                torch.uint8
            )
            v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(
                torch.uint8
            )

            k_encoded = hamming84_encode(k_int4.flatten()).view(num_heads, head_dim)
            v_encoded = hamming84_encode(v_int4.flatten()).view(num_heads, head_dim)

            offset_start = slot * head_dim
            offset_end = offset_start + head_dim
            k_cache[0, layer_idx, :, offset_start:offset_end] = k_encoded
            v_cache[0, layer_idx, :, offset_start:offset_end] = v_encoded
            scales[0, layer_idx, :, slot] = scale

        ref_out = reference_attention_ecc(
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
        triton_out = paged_attention_ecc(
            query,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            scales,
            layer_idx,
            block_size,
        )

        max_diff = (ref_out - triton_out).abs().max().item()
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"

    def test_multi_block_clean_data(self):
        from kv_cache.attention_ecc import (
            paged_attention_ecc,
            reference_attention_ecc,
        )
        from ecc_codecs.triton_kernels import hamming84_encode

        batch_size = 2
        num_heads = 4
        head_dim = 64
        num_layers = 2
        layer_idx = 1
        block_size = 16
        context_len = 50

        num_blocks = 32
        max_blocks = 8
        codewords_per_head = block_size * head_dim

        device = "cuda"
        torch.manual_seed(123)

        query = torch.randn(
            batch_size, num_heads, head_dim, device=device, dtype=torch.float32
        )

        block_table = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device
        )
        num_ctx_blocks = (context_len + block_size - 1) // block_size
        for b in range(batch_size):
            for i in range(num_ctx_blocks):
                block_table[b, i] = b * num_ctx_blocks + i

        context_lens = torch.full(
            (batch_size,), context_len, dtype=torch.int32, device=device
        )

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
            for blk_idx in range(num_ctx_blocks):
                phys_block = int(block_table[b, blk_idx].item())
                start_pos = blk_idx * block_size
                end_pos = min(start_pos + block_size, context_len)

                for slot in range(end_pos - start_pos):
                    k_fp = torch.randn(num_heads, head_dim, device=device)
                    v_fp = torch.randn(num_heads, head_dim, device=device)

                    scale = (
                        torch.maximum(
                            k_fp.abs().max(dim=-1).values, v_fp.abs().max(dim=-1).values
                        )
                        / 7.0
                    )
                    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

                    k_int4 = (
                        torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8
                    ).to(torch.uint8)
                    v_int4 = (
                        torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8
                    ).to(torch.uint8)

                    k_encoded = hamming84_encode(k_int4.flatten()).view(
                        num_heads, head_dim
                    )
                    v_encoded = hamming84_encode(v_int4.flatten()).view(
                        num_heads, head_dim
                    )

                    offset_start = slot * head_dim
                    offset_end = offset_start + head_dim
                    k_cache[
                        phys_block, layer_idx, :, offset_start:offset_end
                    ] = k_encoded
                    v_cache[
                        phys_block, layer_idx, :, offset_start:offset_end
                    ] = v_encoded
                    scales[phys_block, layer_idx, :, slot] = scale

        ref_out = reference_attention_ecc(
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
        triton_out = paged_attention_ecc(
            query,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            scales,
            layer_idx,
            block_size,
        )

        max_diff = (ref_out - triton_out).abs().max().item()
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"

    def test_varying_context_lengths(self):
        from kv_cache.attention_ecc import (
            paged_attention_ecc,
            reference_attention_ecc,
        )
        from ecc_codecs.triton_kernels import hamming84_encode

        batch_size = 3
        num_heads = 2
        head_dim = 32
        num_layers = 1
        layer_idx = 0
        block_size = 8

        context_lengths = [10, 24, 5]

        num_blocks = 32
        max_blocks = 8
        codewords_per_head = block_size * head_dim

        device = "cuda"
        torch.manual_seed(456)

        query = torch.randn(
            batch_size, num_heads, head_dim, device=device, dtype=torch.float32
        )

        block_table = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device
        )
        next_block = 0
        for b in range(batch_size):
            num_ctx_blocks = (context_lengths[b] + block_size - 1) // block_size
            for i in range(num_ctx_blocks):
                block_table[b, i] = next_block
                next_block += 1

        context_lens = torch.tensor(context_lengths, dtype=torch.int32, device=device)

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
            ctx_len = context_lengths[b]
            num_ctx_blocks = (ctx_len + block_size - 1) // block_size

            for blk_idx in range(num_ctx_blocks):
                phys_block = int(block_table[b, blk_idx].item())
                start_pos = blk_idx * block_size
                end_pos = min(start_pos + block_size, ctx_len)

                for slot in range(end_pos - start_pos):
                    k_fp = torch.randn(num_heads, head_dim, device=device)
                    v_fp = torch.randn(num_heads, head_dim, device=device)

                    scale = (
                        torch.maximum(
                            k_fp.abs().max(dim=-1).values, v_fp.abs().max(dim=-1).values
                        )
                        / 7.0
                    )
                    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

                    k_int4 = (
                        torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8
                    ).to(torch.uint8)
                    v_int4 = (
                        torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8
                    ).to(torch.uint8)

                    k_encoded = hamming84_encode(k_int4.flatten()).view(
                        num_heads, head_dim
                    )
                    v_encoded = hamming84_encode(v_int4.flatten()).view(
                        num_heads, head_dim
                    )

                    offset_start = slot * head_dim
                    offset_end = offset_start + head_dim
                    k_cache[
                        phys_block, layer_idx, :, offset_start:offset_end
                    ] = k_encoded
                    v_cache[
                        phys_block, layer_idx, :, offset_start:offset_end
                    ] = v_encoded
                    scales[phys_block, layer_idx, :, slot] = scale

        ref_out = reference_attention_ecc(
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
        triton_out = paged_attention_ecc(
            query,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            scales,
            layer_idx,
            block_size,
        )

        max_diff = (ref_out - triton_out).abs().max().item()
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"

    def test_with_injected_errors_corrected(self):
        from kv_cache.attention_ecc import paged_attention_ecc
        from ecc_codecs.triton_kernels import hamming84_encode, inject_bit_errors_triton

        batch_size = 1
        num_heads = 2
        head_dim = 32
        num_layers = 1
        layer_idx = 0
        block_size = 8
        context_len = 16

        num_blocks = 4
        max_blocks = 4
        codewords_per_head = block_size * head_dim

        device = "cuda"
        torch.manual_seed(789)

        query = torch.randn(
            batch_size, num_heads, head_dim, device=device, dtype=torch.float32
        )

        block_table = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device
        )
        block_table[0, 0] = 0
        block_table[0, 1] = 1

        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

        k_cache_clean = torch.zeros(
            num_blocks,
            num_layers,
            num_heads,
            codewords_per_head,
            dtype=torch.uint8,
            device=device,
        )
        v_cache_clean = torch.zeros(
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

        num_ctx_blocks = (context_len + block_size - 1) // block_size
        for blk_idx in range(num_ctx_blocks):
            phys_block = int(block_table[0, blk_idx].item())
            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, context_len)

            for slot in range(end_pos - start_pos):
                k_fp = torch.randn(num_heads, head_dim, device=device)
                v_fp = torch.randn(num_heads, head_dim, device=device)

                scale = (
                    torch.maximum(
                        k_fp.abs().max(dim=-1).values, v_fp.abs().max(dim=-1).values
                    )
                    / 7.0
                )
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)

                k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(
                    torch.uint8
                )
                v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(
                    torch.uint8
                )

                k_encoded = hamming84_encode(k_int4.flatten()).view(num_heads, head_dim)
                v_encoded = hamming84_encode(v_int4.flatten()).view(num_heads, head_dim)

                offset_start = slot * head_dim
                offset_end = offset_start + head_dim
                k_cache_clean[
                    phys_block, layer_idx, :, offset_start:offset_end
                ] = k_encoded
                v_cache_clean[
                    phys_block, layer_idx, :, offset_start:offset_end
                ] = v_encoded
                scales[phys_block, layer_idx, :, slot] = scale

        clean_output = paged_attention_ecc(
            query,
            k_cache_clean,
            v_cache_clean,
            block_table,
            context_lens,
            scales,
            layer_idx,
            block_size,
        )

        k_cache_corrupted = inject_bit_errors_triton(
            k_cache_clean.flatten(), ber=0.001, n_bits=8, seed=42
        ).view_as(k_cache_clean)
        v_cache_corrupted = inject_bit_errors_triton(
            v_cache_clean.flatten(), ber=0.001, n_bits=8, seed=43
        ).view_as(v_cache_clean)

        corrupted_output = paged_attention_ecc(
            query,
            k_cache_corrupted,
            v_cache_corrupted,
            block_table,
            context_lens,
            scales,
            layer_idx,
            block_size,
        )

        max_diff = (clean_output - corrupted_output).abs().max().item()

        assert max_diff < 0.5, f"Error correction failed, max_diff={max_diff}"


class TestOnlineSoftmax:
    def test_softmax_matches_pytorch(self):
        from kv_cache.attention_ecc import paged_attention_ecc
        from ecc_codecs.triton_kernels import hamming84_encode

        batch_size = 1
        num_heads = 1
        head_dim = 16
        num_layers = 1
        layer_idx = 0
        block_size = 4
        context_len = 8

        num_blocks = 4
        max_blocks = 4
        codewords_per_head = block_size * head_dim

        device = "cuda"
        torch.manual_seed(999)

        query = torch.ones(
            batch_size, num_heads, head_dim, device=device, dtype=torch.float32
        )

        block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

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
        scales = torch.ones(
            num_blocks,
            num_layers,
            num_heads,
            block_size,
            dtype=torch.float32,
            device=device,
        )

        for phys_block in range(2):
            for slot in range(block_size):
                k_int4 = torch.full(
                    (num_heads, head_dim), 8, dtype=torch.uint8, device=device
                )
                v_int4 = torch.full(
                    (num_heads, head_dim), 8 + slot, dtype=torch.uint8, device=device
                )

                k_encoded = hamming84_encode(k_int4.flatten()).view(num_heads, head_dim)
                v_encoded = hamming84_encode(v_int4.flatten()).view(num_heads, head_dim)

                offset_start = slot * head_dim
                offset_end = offset_start + head_dim
                k_cache[phys_block, layer_idx, :, offset_start:offset_end] = k_encoded
                v_cache[phys_block, layer_idx, :, offset_start:offset_end] = v_encoded

        output = paged_attention_ecc(
            query,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            scales,
            layer_idx,
            block_size,
        )

        expected_avg = 1.5
        output_mean = output.mean().item()
        assert (
            abs(output_mean - expected_avg) < 0.1
        ), f"Expected ~{expected_avg}, got {output_mean}"


class TestEdgeCases:
    def test_empty_context(self):
        from kv_cache.attention_ecc import paged_attention_ecc

        batch_size = 1
        num_heads = 2
        head_dim = 32
        num_layers = 1
        block_size = 8
        num_blocks = 4
        max_blocks = 4
        codewords_per_head = block_size * head_dim

        device = "cuda"

        query = torch.randn(
            batch_size, num_heads, head_dim, device=device, dtype=torch.float32
        )
        block_table = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device
        )
        context_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)

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
        scales = torch.ones(
            num_blocks,
            num_layers,
            num_heads,
            block_size,
            dtype=torch.float32,
            device=device,
        )

        output = paged_attention_ecc(
            query,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            scales,
            layer_idx=0,
            block_size=block_size,
        )

        assert output.shape == query.shape

    def test_single_token_context(self):
        from kv_cache.attention_ecc import (
            paged_attention_ecc,
            reference_attention_ecc,
        )
        from ecc_codecs.triton_kernels import hamming84_encode

        batch_size = 1
        num_heads = 2
        head_dim = 32
        num_layers = 1
        layer_idx = 0
        block_size = 8
        context_len = 1

        num_blocks = 4
        max_blocks = 4
        codewords_per_head = block_size * head_dim

        device = "cuda"
        torch.manual_seed(111)

        query = torch.randn(
            batch_size, num_heads, head_dim, device=device, dtype=torch.float32
        )
        block_table = torch.tensor([[0, -1, -1, -1]], dtype=torch.int32, device=device)
        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

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
        scales = torch.ones(
            num_blocks,
            num_layers,
            num_heads,
            block_size,
            dtype=torch.float32,
            device=device,
        )

        k_fp = torch.randn(num_heads, head_dim, device=device)
        v_fp = torch.randn(num_heads, head_dim, device=device)

        scale = (
            torch.maximum(k_fp.abs().max(dim=-1).values, v_fp.abs().max(dim=-1).values)
            / 7.0
        )
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(
            torch.uint8
        )
        v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(
            torch.uint8
        )

        k_encoded = hamming84_encode(k_int4.flatten()).view(num_heads, head_dim)
        v_encoded = hamming84_encode(v_int4.flatten()).view(num_heads, head_dim)

        k_cache[0, layer_idx, :, :head_dim] = k_encoded
        v_cache[0, layer_idx, :, :head_dim] = v_encoded
        scales[0, layer_idx, :, 0] = scale

        ref_out = reference_attention_ecc(
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
        triton_out = paged_attention_ecc(
            query,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            scales,
            layer_idx,
            block_size,
        )

        max_diff = (ref_out - triton_out).abs().max().item()
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"

    def test_large_head_dim(self):
        from kv_cache.attention_ecc import (
            paged_attention_ecc,
            reference_attention_ecc,
        )
        from ecc_codecs.triton_kernels import hamming84_encode

        batch_size = 1
        num_heads = 2
        head_dim = 128
        num_layers = 1
        layer_idx = 0
        block_size = 16
        context_len = 32

        num_blocks = 8
        max_blocks = 4
        codewords_per_head = block_size * head_dim

        device = "cuda"
        torch.manual_seed(222)

        query = torch.randn(
            batch_size, num_heads, head_dim, device=device, dtype=torch.float32
        )

        block_table = torch.tensor([[0, 1, -1, -1]], dtype=torch.int32, device=device)
        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

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

        num_ctx_blocks = (context_len + block_size - 1) // block_size
        for blk_idx in range(num_ctx_blocks):
            phys_block = int(block_table[0, blk_idx].item())
            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, context_len)

            for slot in range(end_pos - start_pos):
                k_fp = torch.randn(num_heads, head_dim, device=device)
                v_fp = torch.randn(num_heads, head_dim, device=device)

                scale = (
                    torch.maximum(
                        k_fp.abs().max(dim=-1).values, v_fp.abs().max(dim=-1).values
                    )
                    / 7.0
                )
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)

                k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(
                    torch.uint8
                )
                v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(
                    torch.uint8
                )

                k_encoded = hamming84_encode(k_int4.flatten()).view(num_heads, head_dim)
                v_encoded = hamming84_encode(v_int4.flatten()).view(num_heads, head_dim)

                offset_start = slot * head_dim
                offset_end = offset_start + head_dim
                k_cache[phys_block, layer_idx, :, offset_start:offset_end] = k_encoded
                v_cache[phys_block, layer_idx, :, offset_start:offset_end] = v_encoded
                scales[phys_block, layer_idx, :, slot] = scale

        ref_out = reference_attention_ecc(
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
        triton_out = paged_attention_ecc(
            query,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            scales,
            layer_idx,
            block_size,
        )

        max_diff = (ref_out - triton_out).abs().max().item()
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"


class TestAttentionGolay:
    def test_golay_reference_roundtrip(self):
        from kv_cache.attention_ecc import paged_attention_ecc
        from ecc_codecs.triton_kernels import golay_encode

        batch_size = 1
        num_heads = 2
        head_dim = 12
        num_layers = 1
        layer_idx = 0
        block_size = 4
        context_len = 8

        num_blocks = 8
        max_blocks = 4

        codewords_per_slot = (head_dim + 2) // 3
        codewords_per_head = block_size * codewords_per_slot

        device = "cuda"
        torch.manual_seed(42)

        query = torch.randn(
            batch_size, num_heads, head_dim, device=device, dtype=torch.float32
        )

        block_table = torch.tensor([[0, 1, -1, -1]], dtype=torch.int32, device=device)
        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

        k_cache = torch.zeros(
            num_blocks,
            num_layers,
            num_heads,
            codewords_per_head,
            dtype=torch.int32,
            device=device,
        )
        v_cache = torch.zeros(
            num_blocks,
            num_layers,
            num_heads,
            codewords_per_head,
            dtype=torch.int32,
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

        num_ctx_blocks = (context_len + block_size - 1) // block_size
        for blk_idx in range(num_ctx_blocks):
            phys_block = int(block_table[0, blk_idx].item())
            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, context_len)

            for slot in range(end_pos - start_pos):
                k_fp = torch.randn(num_heads, head_dim, device=device)
                v_fp = torch.randn(num_heads, head_dim, device=device)

                scale = (
                    torch.maximum(
                        k_fp.abs().max(dim=-1).values, v_fp.abs().max(dim=-1).values
                    )
                    / 7.0
                )
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)

                k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(
                    torch.uint8
                )
                v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(
                    torch.uint8
                )

                for h in range(num_heads):
                    k_vals = k_int4[h]
                    v_vals = v_int4[h]
                    pad_len = (3 - head_dim % 3) % 3
                    if pad_len > 0:
                        k_vals = torch.cat(
                            [
                                k_vals,
                                torch.zeros(pad_len, dtype=k_vals.dtype, device=device),
                            ]
                        )
                        v_vals = torch.cat(
                            [
                                v_vals,
                                torch.zeros(pad_len, dtype=v_vals.dtype, device=device),
                            ]
                        )

                    k_triplets = k_vals.view(-1, 3)
                    v_triplets = v_vals.view(-1, 3)

                    k_encoded = golay_encode(k_triplets)
                    v_encoded = golay_encode(v_triplets)

                    offset_start = slot * codewords_per_slot
                    offset_end = offset_start + codewords_per_slot
                    k_cache[
                        phys_block, layer_idx, h, offset_start:offset_end
                    ] = k_encoded
                    v_cache[
                        phys_block, layer_idx, h, offset_start:offset_end
                    ] = v_encoded

                scales[phys_block, layer_idx, :, slot] = scale

        output = paged_attention_ecc(
            query,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            scales,
            layer_idx,
            block_size,
            codec="golay",
        )

        assert output.shape == query.shape

        assert torch.isfinite(output).all()

    def test_golay_non_divisible_head_dim(self):
        from kv_cache.attention_ecc import paged_attention_ecc
        from ecc_codecs.triton_kernels import golay_encode

        batch_size = 1
        num_heads = 2
        head_dim = 10
        num_layers = 1
        layer_idx = 0
        block_size = 4
        context_len = 4

        num_blocks = 4
        max_blocks = 4
        codewords_per_slot = (head_dim + 2) // 3
        codewords_per_head = block_size * codewords_per_slot

        device = "cuda"
        torch.manual_seed(123)

        query = torch.randn(
            batch_size, num_heads, head_dim, device=device, dtype=torch.float32
        )

        block_table = torch.tensor([[0, -1, -1, -1]], dtype=torch.int32, device=device)
        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

        k_cache = torch.zeros(
            num_blocks,
            num_layers,
            num_heads,
            codewords_per_head,
            dtype=torch.int32,
            device=device,
        )
        v_cache = torch.zeros(
            num_blocks,
            num_layers,
            num_heads,
            codewords_per_head,
            dtype=torch.int32,
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

        for slot in range(context_len):
            k_fp = torch.randn(num_heads, head_dim, device=device)
            v_fp = torch.randn(num_heads, head_dim, device=device)

            scale = (
                torch.maximum(
                    k_fp.abs().max(dim=-1).values, v_fp.abs().max(dim=-1).values
                )
                / 7.0
            )
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)

            k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(
                torch.uint8
            )
            v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(
                torch.uint8
            )

            for h in range(num_heads):
                k_vals = k_int4[h]
                v_vals = v_int4[h]
                pad_len = (3 - head_dim % 3) % 3
                if pad_len > 0:
                    k_vals = torch.cat(
                        [
                            k_vals,
                            torch.zeros(pad_len, dtype=k_vals.dtype, device=device),
                        ]
                    )
                    v_vals = torch.cat(
                        [
                            v_vals,
                            torch.zeros(pad_len, dtype=v_vals.dtype, device=device),
                        ]
                    )

                k_triplets = k_vals.view(-1, 3)
                v_triplets = v_vals.view(-1, 3)

                k_encoded = golay_encode(k_triplets)
                v_encoded = golay_encode(v_triplets)

                offset_start = slot * codewords_per_slot
                offset_end = offset_start + codewords_per_slot
                k_cache[0, layer_idx, h, offset_start:offset_end] = k_encoded
                v_cache[0, layer_idx, h, offset_start:offset_end] = v_encoded

            scales[0, layer_idx, :, slot] = scale

        output = paged_attention_ecc(
            query,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            scales,
            layer_idx,
            block_size,
            codec="golay",
        )

        assert output.shape == query.shape
        assert torch.isfinite(output).all()

