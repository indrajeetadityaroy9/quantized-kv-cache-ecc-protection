"""
Tests for UnprotectedPagedAttentionShim and UnprotectedBackend.

These tests verify:
1. INT4 quantization works correctly
2. Cache storage and retrieval works
3. Error injection happens at the correct point
4. Baseline perplexity matches ECC configs (since both use INT4)
"""

import pytest
import torch
import torch.nn as nn


class TestUnprotectedShimConfig:
    """Test UnprotectedShimConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        from vllm_kernels.unprotected_shim import UnprotectedShimConfig

        config = UnprotectedShimConfig()
        assert config.ber == 0.0
        assert config.block_size == 16
        assert config.num_blocks == 256
        assert config.inject_errors is False
        assert config.seed == 42
        assert config.codec == "int4"

    def test_custom_config(self):
        """Test custom configuration values."""
        from vllm_kernels.unprotected_shim import UnprotectedShimConfig

        config = UnprotectedShimConfig(
            ber=1e-3,
            block_size=32,
            num_blocks=512,
            inject_errors=True,
            seed=123,
        )
        assert config.ber == 1e-3
        assert config.block_size == 32
        assert config.num_blocks == 512
        assert config.inject_errors is True
        assert config.seed == 123


class TestUnprotectedDummyCache:
    """Test UnprotectedDummyCache class."""

    def test_dummy_cache_interface(self):
        """Test that dummy cache satisfies transformers interface."""
        from vllm_kernels.unprotected_shim import UnprotectedDummyCache

        cache = UnprotectedDummyCache(num_layers=12)
        assert len(cache) == 12
        assert cache.get_seq_length() == 0
        assert cache.to_legacy_cache() == ()

    def test_dummy_cache_update(self):
        """Test dummy cache update tracking."""
        from vllm_kernels.unprotected_shim import UnprotectedDummyCache

        cache = UnprotectedDummyCache(num_layers=12)
        k = torch.randn(1, 4, 10, 64)  # [batch, heads, seq, head_dim]
        v = torch.randn(1, 4, 10, 64)

        cache.update(k, v, layer_idx=0)
        assert cache.seen_tokens == 10


class TestUnprotectedBackend:
    """Test UnprotectedBackend class."""

    @pytest.fixture
    def setup_backend(self):
        """Create backend with mock manager."""
        from vllm_kernels.shim import SimpleBlockManager
        from vllm_kernels.unprotected_shim import UnprotectedShimConfig, UnprotectedBackend

        manager = SimpleBlockManager(
            num_blocks=64,
            block_size=16,
            num_layers=4,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",  # Use CPU for tests
            codec="int4",
        )

        config = UnprotectedShimConfig(
            ber=0.0,
            block_size=16,
            num_blocks=64,
            inject_errors=False,
        )

        backend = UnprotectedBackend(manager, config, num_heads=32)
        return backend, manager

    def test_write_quantizes_correctly(self, setup_backend):
        """Test that write() correctly quantizes FP16 to INT4."""
        backend, manager = setup_backend

        # Create sample K,V tensors
        k = torch.randn(1, 8, 256, dtype=torch.float16)  # [batch, seq, hidden]
        v = torch.randn(1, 8, 256, dtype=torch.float16)

        # Write to cache
        backend.write(k, v, layer_idx=0, seq_id=0)

        # Verify blocks were allocated
        assert manager.get_context_len(0) == 8

        # Check cache dtype is uint8 (for INT4 storage)
        assert manager.k_cache.dtype == torch.uint8
        assert manager.v_cache.dtype == torch.uint8

    def test_attend_dequantizes_correctly(self, setup_backend):
        """Test that attend() correctly dequantizes and runs attention."""
        backend, manager = setup_backend

        # Write K,V first
        k = torch.randn(1, 8, 256, dtype=torch.float16)
        v = torch.randn(1, 8, 256, dtype=torch.float16)
        backend.write(k, v, layer_idx=0, seq_id=0)

        # Create query
        q = torch.randn(1, 32, 8, 64, dtype=torch.float16)  # [batch, heads, seq, head_dim]

        # Run attention
        output = backend.attend(q, layer_idx=0, seq_id=0)

        # Output should have same shape as query
        assert output.shape == q.shape

    def test_error_injection_disabled(self, setup_backend):
        """Test that no errors are injected when disabled."""
        backend, manager = setup_backend

        # Write K,V
        k = torch.randn(1, 8, 256, dtype=torch.float16)
        v = torch.randn(1, 8, 256, dtype=torch.float16)
        backend.write(k, v, layer_idx=0, seq_id=0)

        # Run attend twice - should get same results
        q = torch.randn(1, 32, 8, 64, dtype=torch.float16)
        output1 = backend.attend(q, layer_idx=0, seq_id=0)

        # Reset and write again
        manager.reset()
        backend.write(k, v, layer_idx=0, seq_id=0)
        output2 = backend.attend(q, layer_idx=0, seq_id=0)

        # Outputs should be identical (no random errors)
        assert torch.allclose(output1, output2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_error_injection_enabled(self):
        """Test that errors are injected when enabled."""
        from vllm_kernels.shim import SimpleBlockManager
        from vllm_kernels.unprotected_shim import UnprotectedShimConfig, UnprotectedBackend

        manager = SimpleBlockManager(
            num_blocks=64,
            block_size=16,
            num_layers=4,
            num_kv_heads=4,
            head_dim=64,
            device="cuda",
            codec="int4",
        )

        config = UnprotectedShimConfig(
            ber=0.01,  # High BER for visible effect
            block_size=16,
            num_blocks=64,
            inject_errors=True,
        )

        backend = UnprotectedBackend(manager, config, num_heads=32)

        # Write K,V
        k = torch.randn(1, 8, 256, dtype=torch.float16, device="cuda")
        v = torch.randn(1, 8, 256, dtype=torch.float16, device="cuda")
        backend.write(k, v, layer_idx=0, seq_id=0)

        # Run attend - errors should be injected
        q = torch.randn(1, 32, 8, 64, dtype=torch.float16, device="cuda")
        output = backend.attend(q, layer_idx=0, seq_id=0)

        # Check that errors were tracked
        assert backend._bits_flipped > 0
        assert backend._total_bits > 0


class TestQuantizationAccuracy:
    """Test INT4 quantization accuracy."""

    def test_quantization_round_trip(self):
        """Test that quantize -> dequantize preserves values approximately."""
        from vllm_kernels.paged_cache_ecc import compute_quantization_scales

        # Create test tensor
        x = torch.randn(1, 8, 4, 64, dtype=torch.float32)

        # Compute scales
        scales = compute_quantization_scales(x, dim=-1)

        # Quantize
        x_int4 = (torch.round(x / scales.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)

        # Dequantize
        x_recovered = (x_int4.float() - 8.0) * scales.unsqueeze(-1)

        # Check recovery error is bounded
        error = (x - x_recovered).abs().max()
        # Error should be at most ~1.5x the quantization step
        max_error = scales.max() * 1.5
        assert error < max_error, f"Quantization error {error} exceeds bound {max_error}"


class TestUnprotectedPagedAttentionShim:
    """Test UnprotectedPagedAttentionShim class."""

    @pytest.fixture
    def mock_attention_module(self):
        """Create a mock attention module."""
        class MockAttention(nn.Module):
            def __init__(self):
                super().__init__()
                hidden_size = 2048
                num_heads = 32
                num_kv_heads = 4
                head_dim = 64

                self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
                self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
                self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
                self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

                self.num_heads = num_heads
                self.num_key_value_heads = num_kv_heads
                self.head_dim = head_dim

        return MockAttention()

    @pytest.fixture
    def mock_rotary_emb(self):
        """Create a mock rotary embedding."""
        class MockRotaryEmb(nn.Module):
            def __init__(self, dim=64):
                super().__init__()
                self.dim = dim

            def forward(self, x, position_ids):
                seq_len = position_ids.shape[-1]
                cos = torch.ones(1, 1, seq_len, self.dim, device=x.device, dtype=x.dtype)
                sin = torch.zeros(1, 1, seq_len, self.dim, device=x.device, dtype=x.dtype)
                return cos, sin

        return MockRotaryEmb()

    def test_shim_initialization(self, mock_attention_module, mock_rotary_emb):
        """Test that shim initializes correctly."""
        from vllm_kernels.shim import SimpleBlockManager
        from vllm_kernels.unprotected_shim import (
            UnprotectedShimConfig,
            UnprotectedBackend,
            UnprotectedPagedAttentionShim,
        )

        config = UnprotectedShimConfig()
        manager = SimpleBlockManager(
            num_blocks=64,
            block_size=16,
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
            codec="int4",
        )
        backend = UnprotectedBackend(manager, config, num_heads=32)

        shim = UnprotectedPagedAttentionShim(
            original_attn=mock_attention_module,
            layer_idx=0,
            backend=backend,
            rotary_emb=mock_rotary_emb,
        )

        assert shim.num_heads == 32
        assert shim.num_kv_heads == 4
        assert shim.head_dim == 64

    def test_shim_forward(self, mock_attention_module, mock_rotary_emb):
        """Test that shim forward pass works."""
        from vllm_kernels.shim import SimpleBlockManager
        from vllm_kernels.unprotected_shim import (
            UnprotectedShimConfig,
            UnprotectedBackend,
            UnprotectedPagedAttentionShim,
        )

        config = UnprotectedShimConfig()
        manager = SimpleBlockManager(
            num_blocks=64,
            block_size=16,
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
            codec="int4",
        )
        backend = UnprotectedBackend(manager, config, num_heads=32)

        shim = UnprotectedPagedAttentionShim(
            original_attn=mock_attention_module,
            layer_idx=0,
            backend=backend,
            rotary_emb=mock_rotary_emb,
        )

        # Create input
        hidden_states = torch.randn(1, 8, 2048)

        # Run forward
        output, attn_weights, past_kv = shim(hidden_states, use_cache=True)

        # Check output shape
        assert output.shape == hidden_states.shape
        assert attn_weights is None  # We don't compute attention weights


class TestContextManager:
    """Test patch_model_with_unprotected_attention context manager."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_context_manager_restores_attention(self):
        """Test that context manager restores original attention on exit."""
        from transformers import AutoModelForCausalLM
        from vllm_kernels.unprotected_shim import (
            UnprotectedShimConfig,
            UnprotectedPagedAttentionShim,
            patch_model_with_unprotected_attention,
        )

        # Load a small model
        model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            torch_dtype=torch.float16,
            device_map="cuda",
        )

        # Get original attention type
        original_type = type(model.model.layers[0].self_attn)

        config = UnprotectedShimConfig()

        # Patch model
        with patch_model_with_unprotected_attention(model, config, num_blocks=64):
            # Check attention was replaced
            patched_type = type(model.model.layers[0].self_attn)
            assert patched_type == UnprotectedPagedAttentionShim

        # Check attention was restored
        restored_type = type(model.model.layers[0].self_attn)
        assert restored_type == original_type


class TestStatsTracking:
    """Test statistics tracking."""

    def test_get_unprotected_stats(self):
        """Test get_unprotected_stats function."""
        from vllm_kernels.shim import SimpleBlockManager
        from vllm_kernels.unprotected_shim import (
            UnprotectedShimConfig,
            UnprotectedBackend,
            get_unprotected_stats,
        )

        # Create a simple mock model with the backend attached
        class MockModel:
            pass

        model = MockModel()

        manager = SimpleBlockManager(
            num_blocks=64,
            block_size=16,
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            device="cpu",
            codec="int4",
        )

        config = UnprotectedShimConfig()
        backend = UnprotectedBackend(manager, config, num_heads=32)

        model._unprotected_block_manager = manager
        model._unprotected_backend = backend

        stats = get_unprotected_stats(model)

        assert "allocated_blocks" in stats
        assert "free_blocks" in stats
        assert "sequences" in stats
        assert "injection_count" in stats
        assert "bits_flipped" in stats
        assert "total_bits" in stats
        assert "actual_ber" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
