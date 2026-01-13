"""
Tests for fused Triton kernels (quantize + encode).
"""

import pytest
import torch


class TestFusedQuantizeEncodeHamming84:
    """Tests for fused_quantize_encode_hamming84."""

    def test_matches_separate_ops(self):
        """Verify fused kernel matches separate scale + quantize + encode."""
        from ecc_codecs.triton_kernels import (
            fused_quantize_encode_hamming84,
            hamming84_encode,
        )
        from kv_cache.paged_cache_ecc import compute_quantization_scales

        # Generate test input
        x = torch.randn(1000, 128, device="cuda", dtype=torch.float16)

        # Reference implementation (separate ops)
        scales_ref = compute_quantization_scales(x.float(), dim=-1)
        q_ref = torch.round(x.float() / scales_ref.unsqueeze(-1)).clamp(-8, 7) + 8
        int4_ref = q_ref.to(torch.uint8)
        encoded_ref = hamming84_encode(int4_ref.flatten()).view(x.shape)

        # Fused implementation
        encoded_fused, scales_fused = fused_quantize_encode_hamming84(x)

        # Verify scales match
        torch.testing.assert_close(
            scales_fused, scales_ref, rtol=1e-3, atol=1e-5,
            msg="Scales should match reference implementation"
        )

        # Verify encoded codewords match (allowing for edge cases at 0.5 boundaries)
        # Extract INT4 values from codewords
        int4_fused = encoded_fused & 0x0F
        int4_expected = encoded_ref & 0x0F

        # Most values should match exactly
        exact_match_rate = (int4_fused == int4_expected).float().mean()
        assert exact_match_rate > 0.999, \
            f"Match rate too low: {exact_match_rate:.4f} (expected > 0.999)"

        # All differences should be at most 1 quantization level (edge cases)
        diff = (int4_fused.int() - int4_expected.int()).abs()
        assert diff.max() <= 1, \
            f"Found difference > 1 quantization level: max diff = {diff.max()}"

    def test_roundtrip_accuracy(self):
        """Verify encode -> decode roundtrip has reasonable accuracy."""
        from ecc_codecs.triton_kernels import (
            fused_quantize_encode_hamming84,
            hamming84_decode,
        )

        x = torch.randn(500, 64, device="cuda", dtype=torch.float16)

        # Encode with fused kernel
        encoded, scales = fused_quantize_encode_hamming84(x)

        # Decode
        decoded, stats = hamming84_decode(encoded.flatten())
        decoded = decoded.view(x.shape)

        # Dequantize
        dequantized = (decoded.float() - 8.0) * scales.unsqueeze(-1)

        # Check MSE is reasonable (quantization introduces some error)
        mse = ((x.float() - dequantized) ** 2).mean()
        assert mse < 1.0, f"Roundtrip MSE too high: {mse}"

    def test_1d_input(self):
        """Test with 1D input (single row)."""
        from ecc_codecs.triton_kernels import fused_quantize_encode_hamming84

        x = torch.randn(128, device="cuda", dtype=torch.float32)
        encoded, scales = fused_quantize_encode_hamming84(x)

        assert encoded.shape == (128,)
        # 1D input is treated as single row, so scale is shape (1,) not ()
        assert scales.shape == (1,) or scales.numel() == 1
        assert encoded.dtype == torch.uint8
        assert scales.dtype == torch.float32

    def test_3d_input(self):
        """Test with 3D input (batch, seq, hidden)."""
        from ecc_codecs.triton_kernels import fused_quantize_encode_hamming84

        x = torch.randn(4, 32, 128, device="cuda", dtype=torch.float16)
        encoded, scales = fused_quantize_encode_hamming84(x)

        assert encoded.shape == (4, 32, 128)
        assert scales.shape == (4, 32)
        assert encoded.dtype == torch.uint8

    def test_4d_input(self):
        """Test with 4D input (batch, heads, seq, head_dim)."""
        from ecc_codecs.triton_kernels import fused_quantize_encode_hamming84

        x = torch.randn(2, 8, 16, 64, device="cuda", dtype=torch.float16)
        encoded, scales = fused_quantize_encode_hamming84(x)

        assert encoded.shape == (2, 8, 16, 64)
        assert scales.shape == (2, 8, 16)

    def test_zero_input(self):
        """Test that zero input produces valid scale (1.0 to avoid div by zero)."""
        from ecc_codecs.triton_kernels import fused_quantize_encode_hamming84

        x = torch.zeros(100, 64, device="cuda", dtype=torch.float32)
        encoded, scales = fused_quantize_encode_hamming84(x)

        # Scales should be 1.0 for zero input
        assert (scales == 1.0).all(), "Zero input should produce scale=1.0"

        # Encoded should represent value 8 (quantized 0 + offset)
        # After decoding, data bits should be 8 = 0b1000
        decoded_data = encoded & 0x0F  # Extract data bits
        assert (decoded_data == 8).all(), "Zero input should encode to INT4 value 8"

    @pytest.mark.parametrize("row_size", [32, 64, 128, 256, 512])
    def test_various_row_sizes(self, row_size):
        """Test with various row sizes."""
        from ecc_codecs.triton_kernels import fused_quantize_encode_hamming84

        x = torch.randn(100, row_size, device="cuda", dtype=torch.float16)
        encoded, scales = fused_quantize_encode_hamming84(x)

        assert encoded.shape == (100, row_size)
        assert scales.shape == (100,)


class TestFusedQuantizeEncodeHamming74:
    """Tests for fused_quantize_encode_hamming74."""

    def test_matches_separate_ops(self):
        """Verify fused kernel matches separate scale + quantize + encode."""
        from ecc_codecs.triton_kernels import (
            fused_quantize_encode_hamming74,
            hamming74_encode,
        )
        from kv_cache.paged_cache_ecc import compute_quantization_scales

        x = torch.randn(1000, 128, device="cuda", dtype=torch.float16)

        # Reference
        scales_ref = compute_quantization_scales(x.float(), dim=-1)
        q_ref = torch.round(x.float() / scales_ref.unsqueeze(-1)).clamp(-8, 7) + 8
        int4_ref = q_ref.to(torch.uint8)
        encoded_ref = hamming74_encode(int4_ref.flatten()).view(x.shape)

        # Fused
        encoded_fused, scales_fused = fused_quantize_encode_hamming74(x)

        torch.testing.assert_close(scales_fused, scales_ref, rtol=1e-3, atol=1e-5)

        # Allow for rounding edge cases at 0.5 boundaries
        int4_fused = encoded_fused & 0x0F
        int4_expected = encoded_ref & 0x0F
        exact_match_rate = (int4_fused == int4_expected).float().mean()
        assert exact_match_rate > 0.999, f"Match rate too low: {exact_match_rate:.4f}"
        diff = (int4_fused.int() - int4_expected.int()).abs()
        assert diff.max() <= 1, f"Max diff > 1: {diff.max()}"

    def test_7bit_codeword(self):
        """Verify Hamming74 produces 7-bit codewords (bit 7 should be 0)."""
        from ecc_codecs.triton_kernels import fused_quantize_encode_hamming74

        x = torch.randn(100, 64, device="cuda", dtype=torch.float16)
        encoded, _ = fused_quantize_encode_hamming74(x)

        # Bit 7 should always be 0 for Hamming(7,4)
        bit7 = (encoded >> 7) & 1
        assert (bit7 == 0).all(), "Hamming74 should not set bit 7"


class TestFusedDecodeeDequantizeHamming84:
    """Tests for fused_decode_dequantize_hamming84."""

    def test_matches_separate_ops(self):
        """Verify fused decode+dequantize matches separate operations."""
        from ecc_codecs.triton_kernels import (
            fused_quantize_encode_hamming84,
            fused_decode_dequantize_hamming84,
            hamming84_decode,
        )

        x = torch.randn(500, 128, device="cuda", dtype=torch.float16)

        # Encode
        encoded, scales = fused_quantize_encode_hamming84(x)

        # Reference decode + dequantize
        decoded_ref, _ = hamming84_decode(encoded.flatten())
        decoded_ref = decoded_ref.view(x.shape)
        dequant_ref = (decoded_ref.float() - 8.0) * scales.unsqueeze(-1)

        # Fused decode + dequantize
        dequant_fused, _ = fused_decode_dequantize_hamming84(encoded, scales)

        torch.testing.assert_close(
            dequant_fused, dequant_ref.float(),
            rtol=1e-5, atol=1e-5,
            msg="Fused decode+dequantize should match reference"
        )

    def test_error_correction(self):
        """Verify fused decode corrects single-bit errors."""
        from ecc_codecs.triton_kernels import (
            fused_quantize_encode_hamming84,
            fused_decode_dequantize_hamming84,
        )

        x = torch.randn(100, 64, device="cuda", dtype=torch.float16)
        encoded, scales = fused_quantize_encode_hamming84(x)

        # Inject single-bit errors
        error_mask = torch.zeros_like(encoded)
        error_mask[::10, ::8] = 1  # Flip bit 0 in some positions
        corrupted = encoded ^ error_mask

        # Decode and verify correction
        decoded_clean, errors_clean = fused_decode_dequantize_hamming84(encoded, scales)
        decoded_corrupted, errors_corrected = fused_decode_dequantize_hamming84(corrupted, scales)

        # Should have corrected some errors
        assert errors_corrected > 0, "Should have corrected errors"

        # Output should be the same after correction
        torch.testing.assert_close(
            decoded_clean, decoded_corrupted,
            rtol=1e-5, atol=1e-5,
            msg="Single-bit errors should be corrected"
        )

    def test_output_dtype(self):
        """Test different output dtypes."""
        from ecc_codecs.triton_kernels import (
            fused_quantize_encode_hamming84,
            fused_decode_dequantize_hamming84,
        )

        x = torch.randn(50, 32, device="cuda", dtype=torch.float16)
        encoded, scales = fused_quantize_encode_hamming84(x)

        # Default float32
        out_f32, _ = fused_decode_dequantize_hamming84(encoded, scales)
        assert out_f32.dtype == torch.float32

        # Request float16
        out_f16, _ = fused_decode_dequantize_hamming84(encoded, scales, output_dtype=torch.float16)
        assert out_f16.dtype == torch.float16


class TestFusedKernelPerformance:
    """Performance tests for fused kernels."""

    def test_fused_is_not_slower_than_separate(self):
        """Ensure fused kernel is at least as fast as separate operations."""
        import time
        from ecc_codecs.triton_kernels import (
            fused_quantize_encode_hamming84,
            hamming84_encode,
        )
        from kv_cache.paged_cache_ecc import compute_quantization_scales

        x = torch.randn(10000, 128, device="cuda", dtype=torch.float16)

        # Warmup
        for _ in range(5):
            fused_quantize_encode_hamming84(x)
            scales = compute_quantization_scales(x.float(), dim=-1)
            q = torch.round(x.float() / scales.unsqueeze(-1)).clamp(-8, 7) + 8
            hamming84_encode(q.to(torch.uint8).flatten())

        torch.cuda.synchronize()

        # Benchmark fused
        start = time.perf_counter()
        for _ in range(100):
            fused_quantize_encode_hamming84(x)
        torch.cuda.synchronize()
        fused_time = time.perf_counter() - start

        # Benchmark separate
        start = time.perf_counter()
        for _ in range(100):
            scales = compute_quantization_scales(x.float(), dim=-1)
            q = torch.round(x.float() / scales.unsqueeze(-1)).clamp(-8, 7) + 8
            hamming84_encode(q.to(torch.uint8).flatten())
        torch.cuda.synchronize()
        separate_time = time.perf_counter() - start

        print(f"\nFused: {fused_time*1000:.2f}ms, Separate: {separate_time*1000:.2f}ms")
        print(f"Speedup: {separate_time/fused_time:.2f}x")

        # Allow 20% tolerance (fused should not be much slower)
        assert fused_time < separate_time * 1.2, \
            f"Fused kernel is significantly slower: {fused_time:.3f}s vs {separate_time:.3f}s"
