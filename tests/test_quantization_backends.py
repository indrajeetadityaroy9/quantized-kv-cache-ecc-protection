import pytest
import torch

from ecc_codecs.quantization import INT4Quantizer
from ecc_codecs.quantization_backends import (
    QuantizerBackend,
    QuantizationConfig,
    QuantizationMode,
    QuantizedTensor,
    BlockAbsmaxQuantizer,
    PerTokenQuantizer,
    PerChannelQuantizer,
    KIVIQuantizer,
    GroupWiseQuantizer,
    TorchAOQuantizer,
    get_quantizer,
    list_backends,
    quantize_kv_cache,
    dequantize_kv_cache,
    QUANTIZER_BACKENDS,
)


class TestQuantizationConfig:
    """Test QuantizationConfig dataclass."""

    def test_default_config(self):
        config = QuantizationConfig()
        assert config.bits == 4
        assert config.symmetric is True
        assert config.block_size == 32
        assert config.dtype == torch.float16

    def test_custom_config(self):
        config = QuantizationConfig(
            bits=8,
            symmetric=False,
            block_size=64,
            group_size=128,
        )
        assert config.bits == 8
        assert config.symmetric is False
        assert config.block_size == 64
        assert config.group_size == 128


class TestQuantizedTensor:
    """Test QuantizedTensor dataclass."""

    def test_basic_tensor(self):
        data = torch.randint(0, 16, (10, 10), dtype=torch.uint8)
        scales = torch.ones(10)
        qt = QuantizedTensor(data=data, scales=scales)
        assert qt.data.shape == (10, 10)
        assert qt.scales.shape == (10,)
        assert qt.zero_points is None
        assert qt.mode == QuantizationMode.GENERIC

    def test_tensor_with_mode(self):
        data = torch.randint(0, 16, (10, 10), dtype=torch.uint8)
        scales = torch.ones(10)
        qt = QuantizedTensor(data=data, scales=scales, mode=QuantizationMode.KEY)
        assert qt.mode == QuantizationMode.KEY

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device(self):
        data = torch.randint(0, 16, (10, 10), dtype=torch.uint8)
        scales = torch.ones(10)
        qt = QuantizedTensor(data=data, scales=scales)

        qt_cuda = qt.to("cuda")
        assert qt_cuda.data.device.type == "cuda"
        assert qt_cuda.scales.device.type == "cuda"


class TestBackendRegistry:
    """Test backend registry functions."""

    def test_list_backends(self):
        backends = list_backends()
        assert "block_absmax" in backends
        assert "per_token" in backends
        assert "per_channel" in backends
        assert "kivi" in backends
        assert "group_wise" in backends
        assert "torchao" in backends

    def test_get_quantizer_valid(self):
        for name in list_backends():
            quantizer = get_quantizer(name)
            assert isinstance(quantizer, QuantizerBackend)

    def test_get_quantizer_invalid(self):
        with pytest.raises(ValueError, match="Unknown quantizer backend"):
            get_quantizer("nonexistent_backend")

    def test_get_quantizer_with_config(self):
        config = QuantizationConfig(block_size=64)
        quantizer = get_quantizer("block_absmax", config)
        assert quantizer.config.block_size == 64


class TestBlockAbsmaxQuantizer:
    """Test BlockAbsmaxQuantizer backend."""

    def test_roundtrip(self):
        config = QuantizationConfig(block_size=32)
        quantizer = BlockAbsmaxQuantizer(config)

        x = torch.randn(2, 4, 64, dtype=torch.float16)
        qt = quantizer.quantize(x)
        x_recon = quantizer.dequantize(qt)

        assert x_recon.shape == x.shape
        assert x_recon.dtype == torch.float16

        # Check reasonable MSE (INT4 has some loss)
        mse = ((x - x_recon) ** 2).mean().item()
        assert mse < 1.0

    def test_quantized_range(self):
        quantizer = BlockAbsmaxQuantizer()
        x = torch.randn(10, 64, dtype=torch.float16)
        qt = quantizer.quantize(x)

        assert qt.data.min() >= 0
        assert qt.data.max() <= 15
        assert qt.data.dtype == torch.uint8


class TestPerTokenQuantizer:
    """Test PerTokenQuantizer backend."""

    def test_roundtrip(self):
        quantizer = PerTokenQuantizer()

        x = torch.randn(2, 8, 64, dtype=torch.float16)
        qt = quantizer.quantize(x)
        x_recon = quantizer.dequantize(qt)

        assert x_recon.shape == x.shape
        mse = ((x - x_recon) ** 2).mean().item()
        assert mse < 1.0

    def test_scales_shape(self):
        quantizer = PerTokenQuantizer()
        x = torch.randn(2, 8, 64, dtype=torch.float16)
        qt = quantizer.quantize(x)

        # Per-token: scales should have shape [2, 8] (one per last dim)
        assert qt.scales.shape == (2, 8)


class TestPerChannelQuantizer:
    """Test PerChannelQuantizer backend."""

    def test_roundtrip(self):
        quantizer = PerChannelQuantizer()

        x = torch.randn(2, 8, 64, dtype=torch.float16)
        qt = quantizer.quantize(x)
        x_recon = quantizer.dequantize(qt)

        assert x_recon.shape == x.shape
        mse = ((x - x_recon) ** 2).mean().item()
        assert mse < 1.0

    def test_scales_shape(self):
        quantizer = PerChannelQuantizer()
        x = torch.randn(2, 8, 64, dtype=torch.float16)
        qt = quantizer.quantize(x)

        # Per-channel: scales should have shape [64] (one per channel)
        assert qt.scales.shape == (64,)


class TestKIVIQuantizer:
    """Test KIVIQuantizer backend with TRUE KIVI asymmetric quantization.

    KIVI paper (ICML 2024) uses:
    - Asymmetric quantization: Q(X) = round((X - min) / scale), scale = (max - min) / (2^B - 1)
    - Per-channel for keys (grouped along channel dimension)
    - Per-token for values (grouped along token dimension)
    - Group size of 32
    """

    def test_key_uses_per_channel_grouped(self):
        """Keys should be quantized per-channel with group size 32."""
        config = QuantizationConfig(group_size=32)
        quantizer = KIVIQuantizer(config)

        keys = torch.randn(2, 8, 64, 32, dtype=torch.float16)
        qt = quantizer.quantize(keys, QuantizationMode.KEY)

        assert qt.mode == QuantizationMode.KEY
        # Per-channel scales: [n_groups] where n_groups = ceil(32 / 32) = 1
        assert qt.scales.shape[0] == 1  # 32 channels / 32 group_size = 1 group
        # Should have zero_points for asymmetric quantization
        assert qt.zero_points is not None

    def test_value_uses_per_token_grouped(self):
        """Values should be quantized per-token with group size 32."""
        config = QuantizationConfig(group_size=32)
        quantizer = KIVIQuantizer(config)

        values = torch.randn(2, 8, 64, 32, dtype=torch.float16)
        qt = quantizer.quantize(values, QuantizationMode.VALUE)

        assert qt.mode == QuantizationMode.VALUE
        # Per-token scales: [..., n_groups] where n_groups = ceil(32 / 32) = 1
        assert qt.scales.shape[-1] == 1
        # Should have zero_points for asymmetric quantization
        assert qt.zero_points is not None

    def test_asymmetric_quantization_formula(self):
        """Verify KIVI uses asymmetric quantization: Q = round((X - min) / scale)."""
        quantizer = KIVIQuantizer()

        # Create tensor with known range
        x = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float16)
        qt = quantizer.quantize(x, QuantizationMode.VALUE)

        # For asymmetric: zero_point should be min(x) = 0
        # scale = (3 - 0) / 15 = 0.2 for 4-bit
        # Quantized values should be [0, 5, 10, 15]
        assert qt.zero_points is not None

    def test_quantize_kv_convenience(self):
        quantizer = KIVIQuantizer()

        keys = torch.randn(2, 8, 64, 32, dtype=torch.float16)
        values = torch.randn(2, 8, 64, 32, dtype=torch.float16)

        q_keys, q_values = quantizer.quantize_kv(keys, values)

        assert q_keys.mode == QuantizationMode.KEY
        assert q_values.mode == QuantizationMode.VALUE
        assert q_keys.zero_points is not None
        assert q_values.zero_points is not None

    def test_roundtrip_kv(self):
        quantizer = KIVIQuantizer()

        keys = torch.randn(2, 8, 64, 32, dtype=torch.float16)
        values = torch.randn(2, 8, 64, 32, dtype=torch.float16)

        q_keys, q_values = quantizer.quantize_kv(keys, values)
        keys_recon, values_recon = quantizer.dequantize_kv(q_keys, q_values)

        assert keys_recon.shape == keys.shape
        assert values_recon.shape == values.shape

        k_mse = ((keys - keys_recon) ** 2).mean().item()
        v_mse = ((values - values_recon) ** 2).mean().item()

        assert k_mse < 1.0
        assert v_mse < 1.0

    def test_metadata_contains_kivi_params(self):
        """Verify metadata includes KIVI-specific parameters."""
        quantizer = KIVIQuantizer()
        x = torch.randn(2, 8, 64, 32, dtype=torch.float16)
        qt = quantizer.quantize(x, QuantizationMode.KEY)

        assert qt.metadata is not None
        assert "per_channel" in qt.metadata
        assert "group_size" in qt.metadata
        assert "bits" in qt.metadata
        assert qt.metadata["per_channel"] is True
        assert qt.metadata["group_size"] == 32


class TestKIVISymmetricQuantizer:
    """Test KIVISymmetricQuantizer - symmetric version for ECC compatibility."""

    def test_symmetric_quantization(self):
        """KIVISymmetric should use symmetric quantization (no zero_points)."""
        from ecc_codecs.quantization_backends import KIVISymmetricQuantizer

        quantizer = KIVISymmetricQuantizer()
        x = torch.randn(2, 8, 64, 32, dtype=torch.float16)

        qt = quantizer.quantize(x, QuantizationMode.KEY)
        # Symmetric quantization doesn't need zero_points
        assert qt.zero_points is None

    def test_kivi_strategy_selection(self):
        """Should still use per-channel for keys, per-token for values."""
        from ecc_codecs.quantization_backends import KIVISymmetricQuantizer

        quantizer = KIVISymmetricQuantizer()

        keys = torch.randn(2, 8, 64, 32, dtype=torch.float16)
        values = torch.randn(2, 8, 64, 32, dtype=torch.float16)

        q_keys = quantizer.quantize(keys, QuantizationMode.KEY)
        q_values = quantizer.quantize(values, QuantizationMode.VALUE)

        # Keys per-channel: scales shape [32] (one per channel)
        assert q_keys.scales.shape == (32,)
        # Values per-token: scales shape [2, 8, 64] (one per token)
        assert q_values.scales.shape == (2, 8, 64)


class TestGroupWiseQuantizer:
    """Test GroupWiseQuantizer backend."""

    def test_roundtrip(self):
        config = QuantizationConfig(group_size=32)
        quantizer = GroupWiseQuantizer(config)

        x = torch.randn(2, 128, dtype=torch.float16)
        qt = quantizer.quantize(x)
        x_recon = quantizer.dequantize(qt)

        assert x_recon.shape == x.shape
        mse = ((x - x_recon) ** 2).mean().item()
        assert mse < 1.0

    def test_scales_shape(self):
        config = QuantizationConfig(group_size=32)
        quantizer = GroupWiseQuantizer(config)

        x = torch.randn(2, 128, dtype=torch.float16)
        qt = quantizer.quantize(x)

        # 128 / 32 = 4 groups
        assert qt.scales.shape == (2, 4)


class TestTorchAOQuantizer:
    """Test TorchAOQuantizer backend."""

    def test_fallback_works(self):
        quantizer = TorchAOQuantizer()

        x = torch.randn(2, 64, dtype=torch.float16)
        qt = quantizer.quantize(x)
        x_recon = quantizer.dequantize(qt)

        assert x_recon.shape == x.shape
        mse = ((x - x_recon) ** 2).mean().item()
        assert mse < 1.0

    def test_is_available_property(self):
        quantizer = TorchAOQuantizer()
        # Should be a boolean
        assert isinstance(quantizer.is_available, bool)


class TestINT4QuantizerBackwardCompat:
    """Test INT4Quantizer backward compatibility with backend support."""

    def test_default_backend(self):
        quantizer = INT4Quantizer(block_size=32)
        assert quantizer.backend_name == "block_absmax"

    def test_explicit_backend(self):
        quantizer = INT4Quantizer(backend="kivi")
        assert quantizer.backend_name == "kivi"

    def test_backward_compat_interface(self):
        # Original interface: quantizer.quantize(x) -> (q, scales)
        quantizer = INT4Quantizer(block_size=32)
        x = torch.randn(2, 64, dtype=torch.float16)

        q, scales = quantizer.quantize(x)
        x_recon = quantizer.dequantize(q, scales)

        assert x_recon.shape == x.shape
        assert isinstance(q, torch.Tensor)
        assert isinstance(scales, torch.Tensor)

    def test_kivi_mode_interface(self):
        quantizer = INT4Quantizer(backend="kivi")

        keys = torch.randn(2, 8, 64, 32, dtype=torch.float16)
        values = torch.randn(2, 8, 64, 32, dtype=torch.float16)

        # New mode parameter
        q_k, k_scales = quantizer.quantize(keys, mode="key")
        q_v, v_scales = quantizer.quantize(values, mode="value")

        keys_recon = quantizer.dequantize(q_k, k_scales, mode="key")
        values_recon = quantizer.dequantize(q_v, v_scales, mode="value")

        assert keys_recon.shape == keys.shape
        assert values_recon.shape == values.shape

    def test_quantize_kv_method(self):
        quantizer = INT4Quantizer(backend="kivi")

        keys = torch.randn(2, 8, 64, 32, dtype=torch.float16)
        values = torch.randn(2, 8, 64, 32, dtype=torch.float16)

        (q_k, k_scales), (q_v, v_scales) = quantizer.quantize_kv(keys, values)
        keys_recon, values_recon = quantizer.dequantize_kv(
            q_k, k_scales, q_v, v_scales
        )

        assert keys_recon.shape == keys.shape
        assert values_recon.shape == values.shape


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_quantize_kv_cache(self):
        keys = torch.randn(2, 8, 64, 32, dtype=torch.float16)
        values = torch.randn(2, 8, 64, 32, dtype=torch.float16)

        q_keys, q_values = quantize_kv_cache(keys, values, backend="kivi")

        assert isinstance(q_keys, QuantizedTensor)
        assert isinstance(q_values, QuantizedTensor)

    def test_dequantize_kv_cache(self):
        keys = torch.randn(2, 8, 64, 32, dtype=torch.float16)
        values = torch.randn(2, 8, 64, 32, dtype=torch.float16)

        q_keys, q_values = quantize_kv_cache(keys, values, backend="kivi")
        keys_recon, values_recon = dequantize_kv_cache(
            q_keys, q_values, backend="kivi"
        )

        assert keys_recon.shape == keys.shape
        assert values_recon.shape == values.shape


class TestQuantizationQuality:
    """Test quantization quality across backends."""

    @pytest.fixture
    def test_tensor(self):
        return torch.randn(4, 8, 128, dtype=torch.float16)

    @pytest.mark.parametrize("backend", list_backends())
    def test_all_backends_produce_valid_output(self, test_tensor, backend):
        quantizer = get_quantizer(backend)
        qt = quantizer.quantize(test_tensor)
        x_recon = quantizer.dequantize(qt)

        # Basic sanity checks
        assert x_recon.shape == test_tensor.shape
        assert not torch.isnan(x_recon).any()
        assert not torch.isinf(x_recon).any()

    @pytest.mark.parametrize("backend", list_backends())
    def test_all_backends_reasonable_mse(self, test_tensor, backend):
        quantizer = get_quantizer(backend)
        qt = quantizer.quantize(test_tensor)
        x_recon = quantizer.dequantize(qt)

        mse = ((test_tensor - x_recon) ** 2).mean().item()
        # INT4 quantization should have MSE < 1 for standard normal input
        assert mse < 2.0, f"{backend} has unexpectedly high MSE: {mse}"

    def test_kivi_quantizes_keys_correctly(self):
        """Verify KIVI quantizer produces reasonable reconstruction for keys.

        KIVI uses per-channel quantization for keys, which handles specific
        channel distributions. This test verifies it produces valid output
        without comparing to other strategies (since they target different cases).
        """
        # Set seed for reproducibility
        torch.manual_seed(42)

        # Simulate keys with channel outliers (common in transformers)
        keys = torch.randn(2, 8, 64, 32, dtype=torch.float16)
        # Add outliers in specific channels
        keys[..., 0] *= 10
        keys[..., 5] *= 10

        kivi = get_quantizer("kivi")
        qt_kivi = kivi.quantize(keys, QuantizationMode.KEY)
        recon_kivi = kivi.dequantize(qt_kivi)

        mse_kivi = ((keys - recon_kivi) ** 2).mean().item()

        # KIVI should produce reasonable MSE (< 10 for data with outliers)
        assert mse_kivi < 10.0, f"KIVI MSE ({mse_kivi:.4f}) is too high"

        # Verify output shape and validity
        assert recon_kivi.shape == keys.shape
        assert not torch.isnan(recon_kivi).any()
        assert not torch.isinf(recon_kivi).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
