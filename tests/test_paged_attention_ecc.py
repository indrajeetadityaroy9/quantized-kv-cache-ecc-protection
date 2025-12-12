"""
Tests for ECC-Integrated PagedAttention Read Kernel.
"""

import pytest
import torch
import math

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestAttentionHamming84:
    """Tests for PagedAttention with Hamming84 ECC."""

    def test_single_block_clean_data(self):
        """Test attention with single block, no errors."""
        from vllm_kernels.attention_ecc import paged_attention_ecc, reference_attention_ecc
        from hamming74.triton_kernels import hamming84_encode

        # Small config
        batch_size = 1
        num_heads = 2
        head_dim = 32
        num_layers = 1
        layer_idx = 0
        block_size = 16
        context_len = 12  # Less than one block

        num_blocks = 4
        max_blocks = 4
        codewords_per_head = block_size * head_dim

        device = "cuda"
        torch.manual_seed(42)

        # Create query
        query = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float32)

        # Create block table (1 block allocated)
        block_table = torch.full((batch_size, max_blocks), -1, dtype=torch.int32, device=device)
        block_table[0, 0] = 0

        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

        # Create KV cache
        k_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.uint8, device=device)
        v_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.uint8, device=device)
        scales = torch.zeros(num_blocks, num_layers, num_heads, block_size,
                             dtype=torch.float32, device=device)

        # Fill with random data
        for slot in range(context_len):
            k_fp = torch.randn(num_heads, head_dim, device=device)
            v_fp = torch.randn(num_heads, head_dim, device=device)

            scale = torch.maximum(
                k_fp.abs().max(dim=-1).values,
                v_fp.abs().max(dim=-1).values
            ) / 7.0
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)

            k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)
            v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)

            k_encoded = hamming84_encode(k_int4.flatten()).view(num_heads, head_dim)
            v_encoded = hamming84_encode(v_int4.flatten()).view(num_heads, head_dim)

            offset_start = slot * head_dim
            offset_end = offset_start + head_dim
            k_cache[0, layer_idx, :, offset_start:offset_end] = k_encoded
            v_cache[0, layer_idx, :, offset_start:offset_end] = v_encoded
            scales[0, layer_idx, :, slot] = scale

        # Run both implementations
        ref_out = reference_attention_ecc(
            query, k_cache, v_cache, block_table, context_lens, scales,
            layer_idx, block_size, head_dim
        )
        triton_out = paged_attention_ecc(
            query, k_cache, v_cache, block_table, context_lens, scales,
            layer_idx, block_size
        )

        # Compare
        max_diff = (ref_out - triton_out).abs().max().item()
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"

    def test_multi_block_clean_data(self):
        """Test attention spanning multiple blocks."""
        from vllm_kernels.attention_ecc import paged_attention_ecc, reference_attention_ecc
        from hamming74.triton_kernels import hamming84_encode

        batch_size = 2
        num_heads = 4
        head_dim = 64
        num_layers = 2
        layer_idx = 1
        block_size = 16
        context_len = 50  # ~3 blocks

        num_blocks = 32
        max_blocks = 8
        codewords_per_head = block_size * head_dim

        device = "cuda"
        torch.manual_seed(123)

        query = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float32)

        # Allocate blocks for each batch
        block_table = torch.full((batch_size, max_blocks), -1, dtype=torch.int32, device=device)
        num_ctx_blocks = (context_len + block_size - 1) // block_size
        for b in range(batch_size):
            for i in range(num_ctx_blocks):
                block_table[b, i] = b * num_ctx_blocks + i

        context_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device=device)

        k_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.uint8, device=device)
        v_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.uint8, device=device)
        scales = torch.zeros(num_blocks, num_layers, num_heads, block_size,
                             dtype=torch.float32, device=device)

        # Fill cache
        for b in range(batch_size):
            for blk_idx in range(num_ctx_blocks):
                phys_block = int(block_table[b, blk_idx].item())
                start_pos = blk_idx * block_size
                end_pos = min(start_pos + block_size, context_len)

                for slot in range(end_pos - start_pos):
                    k_fp = torch.randn(num_heads, head_dim, device=device)
                    v_fp = torch.randn(num_heads, head_dim, device=device)

                    scale = torch.maximum(
                        k_fp.abs().max(dim=-1).values,
                        v_fp.abs().max(dim=-1).values
                    ) / 7.0
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

        ref_out = reference_attention_ecc(
            query, k_cache, v_cache, block_table, context_lens, scales,
            layer_idx, block_size, head_dim
        )
        triton_out = paged_attention_ecc(
            query, k_cache, v_cache, block_table, context_lens, scales,
            layer_idx, block_size
        )

        max_diff = (ref_out - triton_out).abs().max().item()
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"

    def test_varying_context_lengths(self):
        """Test with different context lengths per batch."""
        from vllm_kernels.attention_ecc import paged_attention_ecc, reference_attention_ecc
        from hamming74.triton_kernels import hamming84_encode

        batch_size = 3
        num_heads = 2
        head_dim = 32
        num_layers = 1
        layer_idx = 0
        block_size = 8

        # Different context lengths
        context_lengths = [10, 24, 5]

        num_blocks = 32
        max_blocks = 8
        codewords_per_head = block_size * head_dim

        device = "cuda"
        torch.manual_seed(456)

        query = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float32)

        block_table = torch.full((batch_size, max_blocks), -1, dtype=torch.int32, device=device)
        next_block = 0
        for b in range(batch_size):
            num_ctx_blocks = (context_lengths[b] + block_size - 1) // block_size
            for i in range(num_ctx_blocks):
                block_table[b, i] = next_block
                next_block += 1

        context_lens = torch.tensor(context_lengths, dtype=torch.int32, device=device)

        k_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.uint8, device=device)
        v_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.uint8, device=device)
        scales = torch.zeros(num_blocks, num_layers, num_heads, block_size,
                             dtype=torch.float32, device=device)

        # Fill cache for each batch
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

                    scale = torch.maximum(
                        k_fp.abs().max(dim=-1).values,
                        v_fp.abs().max(dim=-1).values
                    ) / 7.0
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

        ref_out = reference_attention_ecc(
            query, k_cache, v_cache, block_table, context_lens, scales,
            layer_idx, block_size, head_dim
        )
        triton_out = paged_attention_ecc(
            query, k_cache, v_cache, block_table, context_lens, scales,
            layer_idx, block_size
        )

        max_diff = (ref_out - triton_out).abs().max().item()
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"

    def test_with_injected_errors_corrected(self):
        """Test that single-bit errors are corrected during attention."""
        from vllm_kernels.attention_ecc import paged_attention_ecc
        from hamming74.triton_kernels import hamming84_encode, inject_bit_errors_triton

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

        query = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float32)

        block_table = torch.full((batch_size, max_blocks), -1, dtype=torch.int32, device=device)
        block_table[0, 0] = 0
        block_table[0, 1] = 1

        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

        # Create clean cache
        k_cache_clean = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                                    dtype=torch.uint8, device=device)
        v_cache_clean = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                                    dtype=torch.uint8, device=device)
        scales = torch.zeros(num_blocks, num_layers, num_heads, block_size,
                             dtype=torch.float32, device=device)

        # Fill cache
        num_ctx_blocks = (context_len + block_size - 1) // block_size
        for blk_idx in range(num_ctx_blocks):
            phys_block = int(block_table[0, blk_idx].item())
            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, context_len)

            for slot in range(end_pos - start_pos):
                k_fp = torch.randn(num_heads, head_dim, device=device)
                v_fp = torch.randn(num_heads, head_dim, device=device)

                scale = torch.maximum(
                    k_fp.abs().max(dim=-1).values,
                    v_fp.abs().max(dim=-1).values
                ) / 7.0
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)

                k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)
                v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)

                k_encoded = hamming84_encode(k_int4.flatten()).view(num_heads, head_dim)
                v_encoded = hamming84_encode(v_int4.flatten()).view(num_heads, head_dim)

                offset_start = slot * head_dim
                offset_end = offset_start + head_dim
                k_cache_clean[phys_block, layer_idx, :, offset_start:offset_end] = k_encoded
                v_cache_clean[phys_block, layer_idx, :, offset_start:offset_end] = v_encoded
                scales[phys_block, layer_idx, :, slot] = scale

        # Run with clean cache
        clean_output = paged_attention_ecc(
            query, k_cache_clean, v_cache_clean, block_table, context_lens, scales,
            layer_idx, block_size
        )

        # Inject errors (low BER so most are single-bit, correctable)
        k_cache_corrupted = inject_bit_errors_triton(
            k_cache_clean.flatten(), ber=0.001, n_bits=8, seed=42
        ).view_as(k_cache_clean)
        v_cache_corrupted = inject_bit_errors_triton(
            v_cache_clean.flatten(), ber=0.001, n_bits=8, seed=43
        ).view_as(v_cache_clean)

        # Run with corrupted cache (ECC should correct)
        corrupted_output = paged_attention_ecc(
            query, k_cache_corrupted, v_cache_corrupted, block_table, context_lens, scales,
            layer_idx, block_size
        )

        # Outputs should be very close (single-bit errors corrected)
        max_diff = (clean_output - corrupted_output).abs().max().item()
        # Allow some tolerance for uncorrectable double errors
        assert max_diff < 0.5, f"Error correction failed, max_diff={max_diff}"


class TestOnlineSoftmax:
    """Tests for Online Softmax correctness."""

    def test_softmax_matches_pytorch(self):
        """Verify Online Softmax produces correct attention weights."""
        from vllm_kernels.attention_ecc import paged_attention_ecc
        from hamming74.triton_kernels import hamming84_encode

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

        # Create simple data for manual verification
        query = torch.ones(batch_size, num_heads, head_dim, device=device, dtype=torch.float32)

        block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

        k_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.uint8, device=device)
        v_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.uint8, device=device)
        scales = torch.ones(num_blocks, num_layers, num_heads, block_size,
                            dtype=torch.float32, device=device)

        # Fill with constant values (mid-range INT4 = 8 -> 0 after dequant)
        for phys_block in range(2):
            for slot in range(block_size):
                # All values = 8 (zero after dequant), except vary slightly
                k_int4 = torch.full((num_heads, head_dim), 8, dtype=torch.uint8, device=device)
                v_int4 = torch.full((num_heads, head_dim), 8 + slot, dtype=torch.uint8, device=device)

                k_encoded = hamming84_encode(k_int4.flatten()).view(num_heads, head_dim)
                v_encoded = hamming84_encode(v_int4.flatten()).view(num_heads, head_dim)

                offset_start = slot * head_dim
                offset_end = offset_start + head_dim
                k_cache[phys_block, layer_idx, :, offset_start:offset_end] = k_encoded
                v_cache[phys_block, layer_idx, :, offset_start:offset_end] = v_encoded

        output = paged_attention_ecc(
            query, k_cache, v_cache, block_table, context_lens, scales,
            layer_idx, block_size
        )

        # With uniform K (all zeros after dequant), attention weights should be uniform
        # Output should be average of V values
        # V values: 0, 1, 2, 3, 0, 1, 2, 3 (8 tokens, 2 blocks)
        # Average: (0+1+2+3+0+1+2+3)/8 = 12/8 = 1.5 per element
        # But since V was stored as int4_val and dequant is (v-8)*scale
        # v_int4 = 8, 9, 10, 11 for slots 0-3 in each block
        # dequant = (8-8)*1, (9-8)*1, (10-8)*1, (11-8)*1 = 0, 1, 2, 3
        # average = 1.5

        expected_avg = 1.5
        output_mean = output.mean().item()
        assert abs(output_mean - expected_avg) < 0.1, f"Expected ~{expected_avg}, got {output_mean}"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_context(self):
        """Test with zero context length."""
        from vllm_kernels.attention_ecc import paged_attention_ecc

        batch_size = 1
        num_heads = 2
        head_dim = 32
        num_layers = 1
        block_size = 8
        num_blocks = 4
        max_blocks = 4
        codewords_per_head = block_size * head_dim

        device = "cuda"

        query = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float32)
        block_table = torch.full((batch_size, max_blocks), -1, dtype=torch.int32, device=device)
        context_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)

        k_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.uint8, device=device)
        v_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.uint8, device=device)
        scales = torch.ones(num_blocks, num_layers, num_heads, block_size,
                            dtype=torch.float32, device=device)

        # Should not crash
        output = paged_attention_ecc(
            query, k_cache, v_cache, block_table, context_lens, scales,
            layer_idx=0, block_size=block_size
        )

        # Output should be zeros (or unchanged query, depending on implementation)
        assert output.shape == query.shape

    def test_single_token_context(self):
        """Test with exactly one token in context."""
        from vllm_kernels.attention_ecc import paged_attention_ecc, reference_attention_ecc
        from hamming74.triton_kernels import hamming84_encode

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

        query = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float32)
        block_table = torch.tensor([[0, -1, -1, -1]], dtype=torch.int32, device=device)
        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

        k_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.uint8, device=device)
        v_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.uint8, device=device)
        scales = torch.ones(num_blocks, num_layers, num_heads, block_size,
                            dtype=torch.float32, device=device)

        # Fill single token
        k_fp = torch.randn(num_heads, head_dim, device=device)
        v_fp = torch.randn(num_heads, head_dim, device=device)

        scale = torch.maximum(
            k_fp.abs().max(dim=-1).values,
            v_fp.abs().max(dim=-1).values
        ) / 7.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)
        v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)

        k_encoded = hamming84_encode(k_int4.flatten()).view(num_heads, head_dim)
        v_encoded = hamming84_encode(v_int4.flatten()).view(num_heads, head_dim)

        k_cache[0, layer_idx, :, :head_dim] = k_encoded
        v_cache[0, layer_idx, :, :head_dim] = v_encoded
        scales[0, layer_idx, :, 0] = scale

        ref_out = reference_attention_ecc(
            query, k_cache, v_cache, block_table, context_lens, scales,
            layer_idx, block_size, head_dim
        )
        triton_out = paged_attention_ecc(
            query, k_cache, v_cache, block_table, context_lens, scales,
            layer_idx, block_size
        )

        # With single token, output should equal V (softmax weight = 1.0)
        max_diff = (ref_out - triton_out).abs().max().item()
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"

    def test_large_head_dim(self):
        """Test with larger head dimensions (128)."""
        from vllm_kernels.attention_ecc import paged_attention_ecc, reference_attention_ecc
        from hamming74.triton_kernels import hamming84_encode

        batch_size = 1
        num_heads = 2
        head_dim = 128  # Larger head dim
        num_layers = 1
        layer_idx = 0
        block_size = 16
        context_len = 32

        num_blocks = 8
        max_blocks = 4
        codewords_per_head = block_size * head_dim

        device = "cuda"
        torch.manual_seed(222)

        query = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float32)

        block_table = torch.tensor([[0, 1, -1, -1]], dtype=torch.int32, device=device)
        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

        k_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.uint8, device=device)
        v_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.uint8, device=device)
        scales = torch.zeros(num_blocks, num_layers, num_heads, block_size,
                             dtype=torch.float32, device=device)

        num_ctx_blocks = (context_len + block_size - 1) // block_size
        for blk_idx in range(num_ctx_blocks):
            phys_block = int(block_table[0, blk_idx].item())
            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, context_len)

            for slot in range(end_pos - start_pos):
                k_fp = torch.randn(num_heads, head_dim, device=device)
                v_fp = torch.randn(num_heads, head_dim, device=device)

                scale = torch.maximum(
                    k_fp.abs().max(dim=-1).values,
                    v_fp.abs().max(dim=-1).values
                ) / 7.0
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

        ref_out = reference_attention_ecc(
            query, k_cache, v_cache, block_table, context_lens, scales,
            layer_idx, block_size, head_dim
        )
        triton_out = paged_attention_ecc(
            query, k_cache, v_cache, block_table, context_lens, scales,
            layer_idx, block_size
        )

        max_diff = (ref_out - triton_out).abs().max().item()
        assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"


class TestAttentionGolay:
    """Tests for PagedAttention with Golay ECC (via reference implementation)."""

    def test_golay_reference_roundtrip(self):
        """Test Golay attention via reference implementation."""
        from vllm_kernels.attention_ecc import paged_attention_ecc
        from hamming74.triton_kernels import golay_encode

        batch_size = 1
        num_heads = 2
        head_dim = 12  # Divisible by 3 for clean packing
        num_layers = 1
        layer_idx = 0
        block_size = 4
        context_len = 8

        num_blocks = 8
        max_blocks = 4
        # Golay packs 3 INT4 per codeword: ceil(head_dim / 3) codewords per slot
        codewords_per_slot = (head_dim + 2) // 3
        codewords_per_head = block_size * codewords_per_slot

        device = "cuda"
        torch.manual_seed(42)

        query = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float32)

        block_table = torch.tensor([[0, 1, -1, -1]], dtype=torch.int32, device=device)
        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

        k_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.int32, device=device)
        v_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.int32, device=device)
        scales = torch.zeros(num_blocks, num_layers, num_heads, block_size,
                             dtype=torch.float32, device=device)

        num_ctx_blocks = (context_len + block_size - 1) // block_size
        for blk_idx in range(num_ctx_blocks):
            phys_block = int(block_table[0, blk_idx].item())
            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, context_len)

            for slot in range(end_pos - start_pos):
                k_fp = torch.randn(num_heads, head_dim, device=device)
                v_fp = torch.randn(num_heads, head_dim, device=device)

                scale = torch.maximum(
                    k_fp.abs().max(dim=-1).values,
                    v_fp.abs().max(dim=-1).values
                ) / 7.0
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)

                k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)
                v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)

                # Encode with Golay (per head)
                for h in range(num_heads):
                    # Pad to multiple of 3
                    k_vals = k_int4[h]
                    v_vals = v_int4[h]
                    pad_len = (3 - head_dim % 3) % 3
                    if pad_len > 0:
                        k_vals = torch.cat([k_vals, torch.zeros(pad_len, dtype=k_vals.dtype, device=device)])
                        v_vals = torch.cat([v_vals, torch.zeros(pad_len, dtype=v_vals.dtype, device=device)])

                    k_triplets = k_vals.view(-1, 3)
                    v_triplets = v_vals.view(-1, 3)

                    k_encoded = golay_encode(k_triplets)
                    v_encoded = golay_encode(v_triplets)

                    offset_start = slot * codewords_per_slot
                    offset_end = offset_start + codewords_per_slot
                    k_cache[phys_block, layer_idx, h, offset_start:offset_end] = k_encoded
                    v_cache[phys_block, layer_idx, h, offset_start:offset_end] = v_encoded

                scales[phys_block, layer_idx, :, slot] = scale

        # Test via reference implementation (Golay not yet fused in kernel)
        output = paged_attention_ecc(
            query, k_cache, v_cache, block_table, context_lens, scales,
            layer_idx, block_size, codec="golay"
        )

        assert output.shape == query.shape
        # Output should be finite
        assert torch.isfinite(output).all()

    def test_golay_non_divisible_head_dim(self):
        """Test Golay with head_dim not divisible by 3."""
        from vllm_kernels.attention_ecc import paged_attention_ecc
        from hamming74.triton_kernels import golay_encode

        batch_size = 1
        num_heads = 2
        head_dim = 10  # Not divisible by 3
        num_layers = 1
        layer_idx = 0
        block_size = 4
        context_len = 4

        num_blocks = 4
        max_blocks = 4
        codewords_per_slot = (head_dim + 2) // 3  # = 4 codewords for 10 values
        codewords_per_head = block_size * codewords_per_slot

        device = "cuda"
        torch.manual_seed(123)

        query = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float32)

        block_table = torch.tensor([[0, -1, -1, -1]], dtype=torch.int32, device=device)
        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

        k_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.int32, device=device)
        v_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.int32, device=device)
        scales = torch.zeros(num_blocks, num_layers, num_heads, block_size,
                             dtype=torch.float32, device=device)

        for slot in range(context_len):
            k_fp = torch.randn(num_heads, head_dim, device=device)
            v_fp = torch.randn(num_heads, head_dim, device=device)

            scale = torch.maximum(
                k_fp.abs().max(dim=-1).values,
                v_fp.abs().max(dim=-1).values
            ) / 7.0
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)

            k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)
            v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)

            for h in range(num_heads):
                k_vals = k_int4[h]
                v_vals = v_int4[h]
                pad_len = (3 - head_dim % 3) % 3
                if pad_len > 0:
                    k_vals = torch.cat([k_vals, torch.zeros(pad_len, dtype=k_vals.dtype, device=device)])
                    v_vals = torch.cat([v_vals, torch.zeros(pad_len, dtype=v_vals.dtype, device=device)])

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
            query, k_cache, v_cache, block_table, context_lens, scales,
            layer_idx, block_size, codec="golay"
        )

        assert output.shape == query.shape
        assert torch.isfinite(output).all()


class TestAdaptiveUEP:
    """Tests for Adaptive Unequal Error Protection (UEP) routing."""

    def test_adaptive_uep_basic(self):
        """Test adaptive UEP routes sink blocks to Golay, context to Hamming84."""
        from vllm_kernels.attention_ecc import paged_attention_ecc_adaptive
        from hamming74.triton_kernels import hamming84_encode, golay_encode

        batch_size = 1
        num_heads = 2
        head_dim = 12  # Divisible by 3 for Golay
        num_layers = 1
        layer_idx = 0
        block_size = 4
        context_len = 16  # 4 blocks
        sink_boundary = 2  # First 2 blocks use Golay

        num_blocks = 8
        max_blocks = 4

        device = "cuda"
        torch.manual_seed(42)

        query = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float32)

        block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device)
        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

        # Hamming84 cache layout
        hamming_codewords_per_head = block_size * head_dim
        k_cache = torch.zeros(num_blocks, num_layers, num_heads, hamming_codewords_per_head,
                              dtype=torch.uint8, device=device)
        v_cache = torch.zeros(num_blocks, num_layers, num_heads, hamming_codewords_per_head,
                              dtype=torch.uint8, device=device)

        # Golay cache layout
        codewords_per_slot = (head_dim + 2) // 3
        golay_codewords_per_head = block_size * codewords_per_slot
        sink_k_cache = torch.zeros(num_blocks, num_layers, num_heads, golay_codewords_per_head,
                                   dtype=torch.int32, device=device)
        sink_v_cache = torch.zeros(num_blocks, num_layers, num_heads, golay_codewords_per_head,
                                   dtype=torch.int32, device=device)

        scales = torch.zeros(num_blocks, num_layers, num_heads, block_size,
                             dtype=torch.float32, device=device)

        num_ctx_blocks = (context_len + block_size - 1) // block_size

        for blk_idx in range(num_ctx_blocks):
            phys_block = int(block_table[0, blk_idx].item())
            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, context_len)

            use_golay = blk_idx < sink_boundary

            for slot in range(end_pos - start_pos):
                k_fp = torch.randn(num_heads, head_dim, device=device)
                v_fp = torch.randn(num_heads, head_dim, device=device)

                scale = torch.maximum(
                    k_fp.abs().max(dim=-1).values,
                    v_fp.abs().max(dim=-1).values
                ) / 7.0
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)

                k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)
                v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)

                if use_golay:
                    # Encode with Golay
                    for h in range(num_heads):
                        k_vals = k_int4[h]
                        v_vals = v_int4[h]
                        pad_len = (3 - head_dim % 3) % 3
                        if pad_len > 0:
                            k_vals = torch.cat([k_vals, torch.zeros(pad_len, dtype=k_vals.dtype, device=device)])
                            v_vals = torch.cat([v_vals, torch.zeros(pad_len, dtype=v_vals.dtype, device=device)])

                        k_triplets = k_vals.view(-1, 3)
                        v_triplets = v_vals.view(-1, 3)

                        k_encoded = golay_encode(k_triplets)
                        v_encoded = golay_encode(v_triplets)

                        offset_start = slot * codewords_per_slot
                        offset_end = offset_start + codewords_per_slot
                        sink_k_cache[phys_block, layer_idx, h, offset_start:offset_end] = k_encoded
                        sink_v_cache[phys_block, layer_idx, h, offset_start:offset_end] = v_encoded
                else:
                    # Encode with Hamming84
                    k_encoded = hamming84_encode(k_int4.flatten()).view(num_heads, head_dim)
                    v_encoded = hamming84_encode(v_int4.flatten()).view(num_heads, head_dim)

                    offset_start = slot * head_dim
                    offset_end = offset_start + head_dim
                    k_cache[phys_block, layer_idx, :, offset_start:offset_end] = k_encoded
                    v_cache[phys_block, layer_idx, :, offset_start:offset_end] = v_encoded

                scales[phys_block, layer_idx, :, slot] = scale

        # Run adaptive UEP attention
        output = paged_attention_ecc_adaptive(
            query=query,
            k_cache=k_cache,
            v_cache=v_cache,
            sink_k_cache=sink_k_cache,
            sink_v_cache=sink_v_cache,
            block_table=block_table,
            context_lens=context_lens,
            scales=scales,
            layer_idx=layer_idx,
            block_size=block_size,
            sink_boundary=sink_boundary,
        )

        assert output.shape == query.shape
        assert torch.isfinite(output).all()

    def test_adaptive_uep_sink_only(self):
        """Test when all blocks are within sink boundary."""
        from vllm_kernels.attention_ecc import paged_attention_ecc_adaptive
        from hamming74.triton_kernels import hamming84_encode, golay_encode

        batch_size = 1
        num_heads = 2
        head_dim = 12
        num_layers = 1
        layer_idx = 0
        block_size = 4
        context_len = 8  # 2 blocks
        sink_boundary = 4  # All blocks use Golay

        num_blocks = 8
        max_blocks = 4

        device = "cuda"
        torch.manual_seed(123)

        query = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float32)

        block_table = torch.tensor([[0, 1, -1, -1]], dtype=torch.int32, device=device)
        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

        # Empty Hamming cache (not used)
        hamming_codewords_per_head = block_size * head_dim
        k_cache = torch.zeros(num_blocks, num_layers, num_heads, hamming_codewords_per_head,
                              dtype=torch.uint8, device=device)
        v_cache = torch.zeros(num_blocks, num_layers, num_heads, hamming_codewords_per_head,
                              dtype=torch.uint8, device=device)

        # Golay cache
        codewords_per_slot = (head_dim + 2) // 3
        golay_codewords_per_head = block_size * codewords_per_slot
        sink_k_cache = torch.zeros(num_blocks, num_layers, num_heads, golay_codewords_per_head,
                                   dtype=torch.int32, device=device)
        sink_v_cache = torch.zeros(num_blocks, num_layers, num_heads, golay_codewords_per_head,
                                   dtype=torch.int32, device=device)

        scales = torch.zeros(num_blocks, num_layers, num_heads, block_size,
                             dtype=torch.float32, device=device)

        num_ctx_blocks = (context_len + block_size - 1) // block_size

        for blk_idx in range(num_ctx_blocks):
            phys_block = int(block_table[0, blk_idx].item())
            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, context_len)

            for slot in range(end_pos - start_pos):
                k_fp = torch.randn(num_heads, head_dim, device=device)
                v_fp = torch.randn(num_heads, head_dim, device=device)

                scale = torch.maximum(
                    k_fp.abs().max(dim=-1).values,
                    v_fp.abs().max(dim=-1).values
                ) / 7.0
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)

                k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)
                v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)

                # All blocks use Golay
                for h in range(num_heads):
                    k_vals = k_int4[h]
                    v_vals = v_int4[h]
                    pad_len = (3 - head_dim % 3) % 3
                    if pad_len > 0:
                        k_vals = torch.cat([k_vals, torch.zeros(pad_len, dtype=k_vals.dtype, device=device)])
                        v_vals = torch.cat([v_vals, torch.zeros(pad_len, dtype=v_vals.dtype, device=device)])

                    k_triplets = k_vals.view(-1, 3)
                    v_triplets = v_vals.view(-1, 3)

                    k_encoded = golay_encode(k_triplets)
                    v_encoded = golay_encode(v_triplets)

                    offset_start = slot * codewords_per_slot
                    offset_end = offset_start + codewords_per_slot
                    sink_k_cache[phys_block, layer_idx, h, offset_start:offset_end] = k_encoded
                    sink_v_cache[phys_block, layer_idx, h, offset_start:offset_end] = v_encoded

                scales[phys_block, layer_idx, :, slot] = scale

        output = paged_attention_ecc_adaptive(
            query=query,
            k_cache=k_cache,
            v_cache=v_cache,
            sink_k_cache=sink_k_cache,
            sink_v_cache=sink_v_cache,
            block_table=block_table,
            context_lens=context_lens,
            scales=scales,
            layer_idx=layer_idx,
            block_size=block_size,
            sink_boundary=sink_boundary,
        )

        assert output.shape == query.shape
        assert torch.isfinite(output).all()

    def test_adaptive_uep_no_sink(self):
        """Test when sink_boundary=0 (all Hamming84)."""
        from vllm_kernels.attention_ecc import paged_attention_ecc_adaptive, paged_attention_ecc
        from hamming74.triton_kernels import hamming84_encode

        batch_size = 1
        num_heads = 2
        head_dim = 32
        num_layers = 1
        layer_idx = 0
        block_size = 8
        context_len = 16
        sink_boundary = 0  # No Golay, all Hamming84

        num_blocks = 8
        max_blocks = 4

        device = "cuda"
        torch.manual_seed(456)

        query = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float32)

        block_table = torch.tensor([[0, 1, -1, -1]], dtype=torch.int32, device=device)
        context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)

        codewords_per_head = block_size * head_dim
        k_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.uint8, device=device)
        v_cache = torch.zeros(num_blocks, num_layers, num_heads, codewords_per_head,
                              dtype=torch.uint8, device=device)

        # Empty Golay caches (not used with sink_boundary=0)
        sink_k_cache = torch.zeros(1, num_layers, num_heads, 1,
                                   dtype=torch.int32, device=device)
        sink_v_cache = torch.zeros(1, num_layers, num_heads, 1,
                                   dtype=torch.int32, device=device)

        scales = torch.zeros(num_blocks, num_layers, num_heads, block_size,
                             dtype=torch.float32, device=device)

        num_ctx_blocks = (context_len + block_size - 1) // block_size

        for blk_idx in range(num_ctx_blocks):
            phys_block = int(block_table[0, blk_idx].item())
            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, context_len)

            for slot in range(end_pos - start_pos):
                k_fp = torch.randn(num_heads, head_dim, device=device)
                v_fp = torch.randn(num_heads, head_dim, device=device)

                scale = torch.maximum(
                    k_fp.abs().max(dim=-1).values,
                    v_fp.abs().max(dim=-1).values
                ) / 7.0
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

        # Run with adaptive (should match regular Hamming84)
        adaptive_output = paged_attention_ecc_adaptive(
            query=query,
            k_cache=k_cache,
            v_cache=v_cache,
            sink_k_cache=sink_k_cache,
            sink_v_cache=sink_v_cache,
            block_table=block_table,
            context_lens=context_lens,
            scales=scales,
            layer_idx=layer_idx,
            block_size=block_size,
            sink_boundary=sink_boundary,
        )

        # Run regular Hamming84 for comparison
        regular_output = paged_attention_ecc(
            query, k_cache, v_cache, block_table, context_lens, scales,
            layer_idx, block_size, codec="hamming84"
        )

        # Should produce identical results
        max_diff = (adaptive_output - regular_output).abs().max().item()
        assert max_diff < 1e-5, f"Adaptive with sink_boundary=0 should match regular Hamming84, diff={max_diff}"
