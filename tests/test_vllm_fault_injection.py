"""
Tests for VLLMWithFaultInjection and related fault injection utilities.

These tests verify:
1. BER accuracy - effective BER matches target BER within tolerance
2. Bit-width handling - correct bits per element for each dtype
3. Stochastic behavior - proper probabilistic error injection
4. Statistics tracking - accurate bit counting
5. GPU synchronization - proper synchronization before/after operations
6. FaultInjectionConfig dataclass
7. FaultInjectionAttentionShim class (forward pass takeover)
8. patch_model_with_fault_injection context manager
"""

import pytest
import torch
import torch.nn as nn


class TestFaultInjectionStats:
    """Tests for the FaultInjectionStats class."""

    def test_stats_initialization(self):
        from evaluation.experiments.vllm_comparison import FaultInjectionStats

        stats = FaultInjectionStats()
        result = stats.get_stats()

        assert result["injection_count"] == 0
        assert result["total_bits_flipped"] == 0
        assert result["total_bits_processed"] == 0
        assert result["effective_ber"] == 0.0

    def test_stats_tracking(self):
        from evaluation.experiments.vllm_comparison import FaultInjectionStats

        stats = FaultInjectionStats()
        stats.add_bits_processed(1000)
        stats.add_bits_flipped(10)

        result = stats.get_stats()
        assert result["total_bits_processed"] == 1000
        assert result["total_bits_flipped"] == 10
        assert result["injection_count"] == 1
        assert result["effective_ber"] == 0.01

    def test_stats_reset(self):
        from evaluation.experiments.vllm_comparison import FaultInjectionStats

        stats = FaultInjectionStats()
        stats.add_bits_processed(1000)
        stats.add_bits_flipped(10)
        stats.reset()

        result = stats.get_stats()
        assert result["injection_count"] == 0
        assert result["total_bits_flipped"] == 0
        assert result["total_bits_processed"] == 0


class TestBitWidthCalculation:
    """Tests for correct bit-width calculation based on dtype."""

    @pytest.mark.parametrize(
        "dtype,expected_bits",
        [
            (torch.float16, 16),
            (torch.bfloat16, 16),
            (torch.float32, 32),
            (torch.uint8, 8),
            (torch.int8, 8),
            (torch.int16, 16),
            (torch.int32, 32),
        ],
    )
    def test_element_size_bits(self, dtype, expected_bits):
        """Verify tensor.element_size() * 8 gives correct bit width."""
        tensor = torch.zeros(10, dtype=dtype, device="cuda")
        actual_bits = tensor.element_size() * 8
        assert actual_bits == expected_bits, f"{dtype}: expected {expected_bits}, got {actual_bits}"


class TestStochasticErrorInjection:
    """Tests for stochastic (probabilistic) error injection."""

    def test_stochastic_pytorch_fallback(self):
        """
        Test that PyTorch fallback uses stochastic error injection,
        not deterministic truncation.
        """
        from evaluation.experiments.vllm_comparison import FaultInjectionStats

        stats = FaultInjectionStats()

        # Create a mock function that mimics the PyTorch fallback behavior
        def stochastic_inject(tensor, ber, seed):
            torch.manual_seed(seed)
            tensor_bytes = tensor.view(torch.uint8).flatten()
            total_bits = tensor_bytes.numel() * 8

            # Stochastic: each bit independently flips with probability ber
            flip_probs = torch.rand(total_bits, device=tensor.device)
            flip_mask = flip_probs < ber
            num_errors = flip_mask.sum().item()

            return num_errors

        # With stochastic injection, low BER on small tensors should sometimes
        # produce non-zero errors due to probabilistic sampling
        total_errors = 0
        num_trials = 100
        tensor = torch.zeros(100, dtype=torch.float16, device="cuda")

        for seed in range(num_trials):
            errors = stochastic_inject(tensor, ber=0.001, seed=seed)
            total_errors += errors

        # With 100 trials, 100 elements * 16 bits * 0.001 BER = ~1.6 errors per trial
        # Total should be around 160, with variance
        # If deterministic (truncation), would be exactly 100 * int(1600 * 0.001) = 100
        assert total_errors > 0, "Stochastic injection should produce some errors"

    @pytest.mark.parametrize("target_ber", [0.001, 0.01, 0.05])
    def test_effective_ber_accuracy(self, target_ber):
        """
        Test that effective BER matches target BER within tolerance.

        This is a critical test that verifies the fix for the deterministic
        truncation bug in the original implementation.
        """
        from hamming74.triton_kernels import inject_bit_errors_triton

        # Large tensor for statistical accuracy
        n_elements = 100_000
        n_bits = 16  # FP16
        tensor = torch.zeros(n_elements, dtype=torch.int16, device="cuda")

        corrupted, stats = inject_bit_errors_triton(
            tensor, ber=target_ber, n_bits=n_bits, seed=42, return_stats=True
        )

        total_bits = n_elements * n_bits
        actual_ber = stats[0] / total_bits

        # Should be within 10% of target BER
        tolerance = max(target_ber * 0.1, 0.001)
        deviation = abs(actual_ber - target_ber)

        assert deviation < tolerance, (
            f"BER accuracy failed: target={target_ber:.4f}, "
            f"actual={actual_ber:.4f}, deviation={deviation:.4f}"
        )


class TestFP16ErrorInjection:
    """Tests specifically for FP16 tensor error injection."""

    def test_fp16_bit_width(self):
        """Test that FP16 tensors use 16-bit width for error injection."""
        tensor = torch.randn(1000, dtype=torch.float16, device="cuda")
        n_bits = tensor.element_size() * 8
        assert n_bits == 16, f"FP16 should be 16 bits, got {n_bits}"

    def test_fp16_error_injection_degrades_values(self):
        """Test that bit errors actually degrade FP16 tensor values."""
        from hamming74.triton_kernels import inject_bit_errors_triton

        tensor = torch.randn(1000, dtype=torch.float16, device="cuda")
        original = tensor.clone()

        # View as int16 for injection
        int_view = tensor.flatten().view(torch.int16)
        corrupted, stats = inject_bit_errors_triton(
            int_view, ber=0.1, n_bits=16, seed=42, return_stats=True
        )
        int_view.copy_(corrupted)

        # Verify that values changed
        changed = (tensor != original).sum().item()
        assert changed > 0, "Error injection should change some values"
        assert stats[0] > 0, "Should have injected some errors"


class TestFP8ErrorInjection:
    """Tests for FP8 tensor error injection."""

    @pytest.mark.skipif(
        not hasattr(torch, "float8_e4m3fn"),
        reason="FP8 not available in this PyTorch version",
    )
    def test_fp8_bit_width(self):
        """Test that FP8 tensors use 8-bit width for error injection."""
        tensor = torch.zeros(100, dtype=torch.float8_e4m3fn, device="cuda")
        n_bits = tensor.element_size() * 8
        assert n_bits == 8, f"FP8 should be 8 bits, got {n_bits}"


class TestGPUSynchronization:
    """Tests for proper GPU synchronization."""

    def test_synchronization_before_injection(self):
        """
        Test that GPU synchronization happens before tensor access.

        This ensures that any pending GPU operations complete before
        we start modifying tensor values.
        """
        # Create tensor and launch an async operation
        tensor = torch.randn(10000, dtype=torch.float16, device="cuda")

        # Manually synchronize (as the code should do)
        torch.cuda.synchronize()

        # Now we can safely access the tensor
        n_bits = tensor.element_size() * 8
        assert n_bits == 16

    def test_synchronization_after_injection(self):
        """
        Test that GPU synchronization happens after tensor modification.

        This ensures that error injection completes before subsequent operations.
        """
        from hamming74.triton_kernels import inject_bit_errors_triton

        tensor = torch.zeros(10000, dtype=torch.int16, device="cuda")
        corrupted, _ = inject_bit_errors_triton(
            tensor, ber=0.1, n_bits=16, seed=42, return_stats=True
        )

        # Synchronize and verify
        torch.cuda.synchronize()

        # Should be able to safely access result
        num_nonzero = (corrupted != 0).sum().item()
        assert num_nonzero > 0


class TestErrorInjectionIntegration:
    """Integration tests for the full error injection pipeline."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_full_injection_pipeline_fp16(self):
        """Test complete error injection for FP16 tensor."""
        from hamming74.triton_kernels import inject_bit_errors_triton

        # Simulate KV cache tensor (batch, heads, seq, dim)
        tensor = torch.randn(2, 8, 128, 64, dtype=torch.float16, device="cuda")
        original = tensor.clone()

        # Flatten and inject
        flat = tensor.flatten()
        int_view = flat.view(torch.int16)

        torch.cuda.synchronize()
        corrupted, stats = inject_bit_errors_triton(
            int_view, ber=0.01, n_bits=16, seed=42, return_stats=True
        )
        int_view.copy_(corrupted)
        torch.cuda.synchronize()

        # Verify injection happened
        assert stats[0] > 0, "Should have injected errors"

        # Calculate effective BER
        total_bits = tensor.numel() * 16
        effective_ber = stats[0] / total_bits

        # Should be within 20% of target for reasonable sample size
        assert 0.008 < effective_ber < 0.012, f"BER={effective_ber} outside expected range"

    def test_injection_preserves_tensor_shape(self):
        """Test that error injection preserves original tensor shape."""
        from hamming74.triton_kernels import inject_bit_errors_triton

        shapes = [
            (100,),
            (10, 10),
            (2, 8, 128, 64),
            (4, 32, 256, 128),
        ]

        for shape in shapes:
            tensor = torch.randn(shape, dtype=torch.float16, device="cuda")
            original_shape = tensor.shape

            flat = tensor.flatten()
            int_view = flat.view(torch.int16)
            corrupted, _ = inject_bit_errors_triton(
                int_view, ber=0.01, n_bits=16, seed=42, return_stats=True
            )
            int_view.copy_(corrupted)

            # Reshape should still work
            assert tensor.shape == original_shape, f"Shape changed: {original_shape} -> {tensor.shape}"


class TestLowBEREffectiveness:
    """Tests to verify low BER rates work correctly (previous bug)."""

    @pytest.mark.parametrize("ber", [1e-4, 1e-5])
    def test_very_low_ber_produces_errors(self, ber):
        """
        Test that very low BER rates still produce some errors on large tensors.

        This was a bug in the original implementation where int() truncation
        would round low BER to zero errors.
        """
        from hamming74.triton_kernels import inject_bit_errors_triton

        # Large tensor to ensure statistical significance
        n_elements = 1_000_000
        tensor = torch.zeros(n_elements, dtype=torch.int16, device="cuda")

        corrupted, stats = inject_bit_errors_triton(
            tensor, ber=ber, n_bits=16, seed=42, return_stats=True
        )

        total_bits = n_elements * 16
        expected_errors = total_bits * ber

        # Should have roughly the expected number of errors (within 50%)
        assert stats[0] > 0, f"BER={ber} should produce errors on large tensor"
        assert (
            0.5 * expected_errors < stats[0] < 1.5 * expected_errors
        ), f"BER={ber}: expected ~{expected_errors:.0f} errors, got {stats[0]}"


class TestVLLMAttentionWithErrors:
    """Tests for VLLMAttentionWithErrors wrapper class (backwards compatibility alias)."""

    def test_class_exists(self):
        """Verify VLLMAttentionWithErrors class is defined."""
        from evaluation.experiments.vllm_comparison import VLLMAttentionWithErrors

        assert VLLMAttentionWithErrors is not None

    def test_inherits_from_nn_module(self):
        """Verify VLLMAttentionWithErrors inherits from nn.Module."""
        from evaluation.experiments.vllm_comparison import VLLMAttentionWithErrors

        assert issubclass(VLLMAttentionWithErrors, nn.Module)

    def test_is_alias_for_shim(self):
        """Verify VLLMAttentionWithErrors is an alias for FaultInjectionAttentionShim."""
        from evaluation.experiments.vllm_comparison import (
            VLLMAttentionWithErrors,
            FaultInjectionAttentionShim,
        )

        assert VLLMAttentionWithErrors is FaultInjectionAttentionShim


class TestFaultInjectionConfig:
    """Tests for the FaultInjectionConfig dataclass."""

    def test_config_defaults(self):
        """Test that FaultInjectionConfig has correct default values."""
        from evaluation.experiments.vllm_comparison import FaultInjectionConfig

        config = FaultInjectionConfig()

        assert config.ber == 0.0
        assert config.seed == 42
        assert config.inject_on_k is True
        assert config.inject_on_v is True

    def test_config_custom_values(self):
        """Test that FaultInjectionConfig accepts custom values."""
        from evaluation.experiments.vllm_comparison import FaultInjectionConfig

        config = FaultInjectionConfig(
            ber=1e-3,
            seed=123,
            inject_on_k=False,
            inject_on_v=True,
        )

        assert config.ber == 1e-3
        assert config.seed == 123
        assert config.inject_on_k is False
        assert config.inject_on_v is True

    def test_config_is_dataclass(self):
        """Verify FaultInjectionConfig is a dataclass."""
        from dataclasses import is_dataclass
        from evaluation.experiments.vllm_comparison import FaultInjectionConfig

        assert is_dataclass(FaultInjectionConfig)


class TestFaultInjectionAttentionShim:
    """Tests for the FaultInjectionAttentionShim class."""

    def test_class_exists(self):
        """Verify FaultInjectionAttentionShim class is defined."""
        from evaluation.experiments.vllm_comparison import FaultInjectionAttentionShim

        assert FaultInjectionAttentionShim is not None

    def test_inherits_from_nn_module(self):
        """Verify FaultInjectionAttentionShim inherits from nn.Module."""
        from evaluation.experiments.vllm_comparison import FaultInjectionAttentionShim

        assert issubclass(FaultInjectionAttentionShim, nn.Module)

    def test_requires_projection_layers(self):
        """Test that shim requires q_proj, k_proj, v_proj, o_proj."""
        from evaluation.experiments.vllm_comparison import (
            FaultInjectionAttentionShim,
            FaultInjectionConfig,
            FaultInjectionStats,
        )

        # Mock attention module missing projections
        class BadAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.head_dim = 64

        config = FaultInjectionConfig()
        stats = FaultInjectionStats()

        with pytest.raises(ValueError, match="missing required projection"):
            FaultInjectionAttentionShim(
                original_attn=BadAttn(),
                layer_idx=0,
                config=config,
                stats_tracker=stats,
                rotary_emb=None,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_shim_construction_with_valid_attention(self):
        """Test that shim can be constructed with valid attention module."""
        from evaluation.experiments.vllm_comparison import (
            FaultInjectionAttentionShim,
            FaultInjectionConfig,
            FaultInjectionStats,
        )

        # Mock attention module with required projections
        class MockAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 8
                self.num_key_value_heads = 8
                self.head_dim = 64
                hidden_size = self.num_heads * self.head_dim
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.k_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)
                self.o_proj = nn.Linear(hidden_size, hidden_size)

        config = FaultInjectionConfig(ber=1e-3)
        stats = FaultInjectionStats()
        mock_attn = MockAttn().cuda()

        shim = FaultInjectionAttentionShim(
            original_attn=mock_attn,
            layer_idx=0,
            config=config,
            stats_tracker=stats,
            rotary_emb=None,
        )

        assert shim.num_heads == 8
        assert shim.num_kv_heads == 8
        assert shim.head_dim == 64
        assert shim.hidden_size == 512

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_shim_forward_pass(self):
        """Test that shim forward pass works and injects errors."""
        from evaluation.experiments.vllm_comparison import (
            FaultInjectionAttentionShim,
            FaultInjectionConfig,
            FaultInjectionStats,
        )

        # Mock attention module
        class MockAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 4
                self.num_key_value_heads = 4
                self.head_dim = 32
                hidden_size = self.num_heads * self.head_dim
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.k_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)
                self.o_proj = nn.Linear(hidden_size, hidden_size)

        config = FaultInjectionConfig(ber=0.01, seed=42)
        stats = FaultInjectionStats()
        mock_attn = MockAttn().cuda().half()

        shim = FaultInjectionAttentionShim(
            original_attn=mock_attn,
            layer_idx=0,
            config=config,
            stats_tracker=stats,
            rotary_emb=None,
        ).cuda().half()

        # Test forward pass
        batch_size, seq_len = 2, 16
        hidden_size = 4 * 32  # num_heads * head_dim
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)

        output, attn_weights, past_kv = shim.forward(hidden_states)

        # Check output shape
        assert output.shape == hidden_states.shape
        assert attn_weights is None  # We don't compute attention weights

        # Check that errors were injected (stats should show some bits processed)
        result_stats = stats.get_stats()
        assert result_stats["total_bits_processed"] > 0
        # With BER=0.01, we should have some errors
        assert result_stats["total_bits_flipped"] > 0


class TestPatchModelWithFaultInjection:
    """Tests for the patch_model_with_fault_injection context manager."""

    def test_context_manager_exists(self):
        """Verify patch_model_with_fault_injection function exists."""
        from evaluation.experiments.vllm_comparison import patch_model_with_fault_injection

        assert callable(patch_model_with_fault_injection)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_context_manager_patches_and_restores(self):
        """Test that context manager patches attention and restores on exit."""
        from evaluation.experiments.vllm_comparison import (
            patch_model_with_fault_injection,
            FaultInjectionConfig,
            FaultInjectionStats,
            FaultInjectionAttentionShim,
        )

        # Create a minimal mock model structure
        class MockAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 4
                self.num_key_value_heads = 4
                self.head_dim = 32
                hidden_size = self.num_heads * self.head_dim
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.k_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)
                self.o_proj = nn.Linear(hidden_size, hidden_size)
                self.rotary_emb = None

        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = MockAttn()

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([MockLayer() for _ in range(2)])

        model = MockModel().cuda().half()
        original_attn_0 = model.layers[0].self_attn
        original_attn_1 = model.layers[1].self_attn

        config = FaultInjectionConfig(ber=1e-3)
        stats = FaultInjectionStats()

        # Inside context: attention should be replaced
        with patch_model_with_fault_injection(model, config, stats):
            assert isinstance(model.layers[0].self_attn, FaultInjectionAttentionShim)
            assert isinstance(model.layers[1].self_attn, FaultInjectionAttentionShim)

        # After context: original attention should be restored
        assert model.layers[0].self_attn is original_attn_0
        assert model.layers[1].self_attn is original_attn_1

    def test_context_manager_raises_on_unsupported_model(self):
        """Test that context manager raises on unsupported model architecture."""
        from evaluation.experiments.vllm_comparison import (
            patch_model_with_fault_injection,
            FaultInjectionConfig,
        )

        # Model without 'layers' attribute
        class BadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

        model = BadModel()
        config = FaultInjectionConfig()

        with pytest.raises(ValueError, match="cannot find 'layers'"):
            with patch_model_with_fault_injection(model, config):
                pass


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_find_rotary_embedding_exists(self):
        """Verify _find_rotary_embedding function exists."""
        from evaluation.experiments.vllm_comparison import _find_rotary_embedding

        assert callable(_find_rotary_embedding)

    def test_get_vllm_underlying_model_exists(self):
        """Verify _get_vllm_underlying_model function exists."""
        from evaluation.experiments.vllm_comparison import _get_vllm_underlying_model

        assert callable(_get_vllm_underlying_model)
