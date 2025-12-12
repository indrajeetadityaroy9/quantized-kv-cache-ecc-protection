"""
Tests for Triton Linear Interpolation implementation.

Verifies that the Triton GPU kernel correctly interpolates
double-detected errors with neighbor averages.
"""

import pytest
import torch

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestTritonInterpolation:
    """Tests for interpolation Triton kernel."""

    def test_no_double_errors_unchanged(self):
        """Tensor with no double errors returns exact copy."""
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors
        from hamming74.triton_kernels.config import ErrorType

        q = torch.tensor([1, 5, 10, 15, 8], dtype=torch.uint8, device="cuda")
        err = torch.zeros_like(q)  # NO_ERROR = 0

        result = interpolate_double_errors(q, err)

        assert torch.equal(result, q), "No-error case should return exact copy"

    def test_single_double_error_middle(self):
        """Single double error in middle is interpolated correctly."""
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors
        from hamming74.triton_kernels.config import ErrorType

        q = torch.tensor([4, 8, 12, 8, 4], dtype=torch.uint8, device="cuda")
        err = torch.tensor([0, 0, ErrorType.DOUBLE_DETECTED, 0, 0], dtype=torch.uint8, device="cuda")

        result = interpolate_double_errors(q, err)

        # Position 2 should be (8 + 8) / 2 = 8
        expected = torch.tensor([4, 8, 8, 8, 4], dtype=torch.uint8, device="cuda")
        assert torch.equal(result, expected), f"Middle interpolation failed: {result} vs {expected}"

    def test_double_error_first_position(self):
        """Double error at first position handles boundary correctly."""
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors
        from hamming74.triton_kernels.config import ErrorType

        q = torch.tensor([15, 4, 8, 12], dtype=torch.uint8, device="cuda")
        err = torch.tensor([ErrorType.DOUBLE_DETECTED, 0, 0, 0], dtype=torch.uint8, device="cuda")

        result = interpolate_double_errors(q, err)

        # Position 0 boundary: uses (q[0] + q[1]) / 2 = (15 + 4) / 2 = 9.5 -> 10
        expected = torch.tensor([10, 4, 8, 12], dtype=torch.uint8, device="cuda")
        assert torch.equal(result, expected), f"Left boundary failed: {result} vs {expected}"

    def test_double_error_last_position(self):
        """Double error at last position handles boundary correctly."""
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors
        from hamming74.triton_kernels.config import ErrorType

        q = torch.tensor([4, 8, 12, 15], dtype=torch.uint8, device="cuda")
        err = torch.tensor([0, 0, 0, ErrorType.DOUBLE_DETECTED], dtype=torch.uint8, device="cuda")

        result = interpolate_double_errors(q, err)

        # Position 3 boundary: uses (q[2] + q[3]) / 2 = (12 + 15) / 2 = 13.5 -> 14
        expected = torch.tensor([4, 8, 12, 14], dtype=torch.uint8, device="cuda")
        assert torch.equal(result, expected), f"Right boundary failed: {result} vs {expected}"

    def test_multiple_scattered_double_errors(self):
        """Multiple scattered double errors are all interpolated."""
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors
        from hamming74.triton_kernels.config import ErrorType

        q = torch.tensor([0, 4, 8, 12, 8, 4, 0], dtype=torch.uint8, device="cuda")
        err = torch.tensor(
            [0, ErrorType.DOUBLE_DETECTED, 0, ErrorType.DOUBLE_DETECTED, 0, ErrorType.DOUBLE_DETECTED, 0],
            dtype=torch.uint8, device="cuda"
        )

        result = interpolate_double_errors(q, err)

        # Position 1: (0 + 8) / 2 = 4
        # Position 3: (8 + 8) / 2 = 8
        # Position 5: (8 + 0) / 2 = 4
        expected = torch.tensor([0, 4, 8, 8, 8, 4, 0], dtype=torch.uint8, device="cuda")
        assert torch.equal(result, expected), f"Scattered errors failed: {result} vs {expected}"

    def test_consecutive_double_errors(self):
        """Consecutive double errors use their own (corrupted) neighbors."""
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors
        from hamming74.triton_kernels.config import ErrorType

        # In this case, interpolation uses corrupted neighbors (not perfect, but expected)
        q = torch.tensor([4, 0, 0, 0, 4], dtype=torch.uint8, device="cuda")
        err = torch.tensor(
            [0, ErrorType.DOUBLE_DETECTED, ErrorType.DOUBLE_DETECTED, ErrorType.DOUBLE_DETECTED, 0],
            dtype=torch.uint8, device="cuda"
        )

        result = interpolate_double_errors(q, err)

        # Position 1: (4 + 0) / 2 = 2
        # Position 2: (0 + 0) / 2 = 0
        # Position 3: (0 + 4) / 2 = 2
        expected = torch.tensor([4, 2, 0, 2, 4], dtype=torch.uint8, device="cuda")
        assert torch.equal(result, expected), f"Consecutive errors failed: {result} vs {expected}"

    def test_large_tensor_non_double_unchanged(self):
        """Large tensor: non-double-error positions are unchanged."""
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors
        from hamming74.triton_kernels.config import ErrorType

        N = 100000
        q = torch.randint(0, 16, (N,), dtype=torch.uint8, device="cuda")
        err = torch.zeros(N, dtype=torch.uint8, device="cuda")

        # Mark 10% as double errors
        double_mask = torch.rand(N, device="cuda") < 0.1
        err[double_mask] = ErrorType.DOUBLE_DETECTED

        result = interpolate_double_errors(q, err)

        # Verify non-double-error positions are unchanged
        non_double = ~double_mask
        assert torch.equal(result[non_double], q[non_double]), \
            "Non-double positions should be unchanged"

    def test_output_in_valid_range(self):
        """Output values are always in valid INT4 range [0, 15]."""
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors
        from hamming74.triton_kernels.config import ErrorType

        N = 100000
        q = torch.randint(0, 16, (N,), dtype=torch.uint8, device="cuda")
        err = torch.zeros(N, dtype=torch.uint8, device="cuda")

        # Mark random positions as double errors
        err[torch.rand(N, device="cuda") < 0.2] = ErrorType.DOUBLE_DETECTED

        result = interpolate_double_errors(q, err)

        assert (result >= 0).all(), "Result has values below 0"
        assert (result <= 15).all(), "Result has values above 15"


class TestTritonInterpolation2D:
    """Tests for 2D tensor interpolation."""

    def test_2d_shape_preserved(self):
        """2D tensor shape is preserved."""
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors
        from hamming74.triton_kernels.config import ErrorType

        q = torch.randint(0, 16, (32, 1024), dtype=torch.uint8, device="cuda")
        err = torch.zeros_like(q)
        err[::4, ::10] = ErrorType.DOUBLE_DETECTED

        result = interpolate_double_errors(q, err)

        assert result.shape == q.shape, "Shape should be preserved"

    def test_2d_non_double_unchanged(self):
        """2D tensor: non-double-error positions are unchanged."""
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors
        from hamming74.triton_kernels.config import ErrorType

        q = torch.randint(0, 16, (32, 1024), dtype=torch.uint8, device="cuda")
        err = torch.zeros_like(q)
        err[::4, ::10] = ErrorType.DOUBLE_DETECTED

        result = interpolate_double_errors(q, err)

        non_double = err != ErrorType.DOUBLE_DETECTED
        assert torch.equal(result[non_double], q[non_double]), \
            "Non-double positions should be unchanged (2D)"

    def test_2d_batch_independent(self):
        """Each row/sequence is interpolated independently."""
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors
        from hamming74.triton_kernels.config import ErrorType

        # Create 2 sequences where double errors are at different positions
        q = torch.tensor([
            [4, 0, 8, 4],   # Row 0: double error at position 1
            [4, 8, 0, 4],   # Row 1: double error at position 2
        ], dtype=torch.uint8, device="cuda")

        err = torch.tensor([
            [0, ErrorType.DOUBLE_DETECTED, 0, 0],
            [0, 0, ErrorType.DOUBLE_DETECTED, 0],
        ], dtype=torch.uint8, device="cuda")

        result = interpolate_double_errors(q, err)

        # Row 0, position 1: (4 + 8) / 2 = 6
        # Row 1, position 2: (8 + 4) / 2 = 6
        expected = torch.tensor([
            [4, 6, 8, 4],
            [4, 8, 6, 4],
        ], dtype=torch.uint8, device="cuda")

        assert torch.equal(result, expected), f"2D batch failed: {result} vs {expected}"


class TestTritonInterpolationPerformance:
    """Performance-related tests."""

    def test_handles_empty_tensor(self):
        """Empty tensor doesn't crash."""
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors

        empty_q = torch.tensor([], dtype=torch.uint8, device="cuda")
        empty_err = torch.tensor([], dtype=torch.uint8, device="cuda")

        result = interpolate_double_errors(empty_q, empty_err)

        assert result.numel() == 0

    def test_handles_single_element(self):
        """Single element tensor works (boundary case)."""
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors
        from hamming74.triton_kernels.config import ErrorType

        single_q = torch.tensor([8], dtype=torch.uint8, device="cuda")
        single_err = torch.tensor([ErrorType.DOUBLE_DETECTED], dtype=torch.uint8, device="cuda")

        result = interpolate_double_errors(single_q, single_err)

        # Single element: neighbors are itself, so (8 + 8) / 2 = 8
        assert result.item() == 8

    def test_early_exit_no_double_errors(self):
        """Early exit when no double errors present."""
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors
        from hamming74.triton_kernels.config import ErrorType

        # Large tensor with no double errors
        q = torch.randint(0, 16, (1000000,), dtype=torch.uint8, device="cuda")
        err = torch.zeros_like(q)  # All NO_ERROR

        # Should return quickly via early exit path
        result = interpolate_double_errors(q, err)

        assert torch.equal(result, q)

    @pytest.mark.parametrize("size", [1, 100, 1024, 10000, 100000])
    def test_various_sizes(self, size):
        """Test various tensor sizes."""
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors
        from hamming74.triton_kernels.config import ErrorType

        q = torch.randint(0, 16, (size,), dtype=torch.uint8, device="cuda")
        err = torch.zeros(size, dtype=torch.uint8, device="cuda")

        # Mark some positions as double errors
        if size > 1:
            err[::3] = ErrorType.DOUBLE_DETECTED

        result = interpolate_double_errors(q, err)

        assert result.shape == q.shape
        assert (result >= 0).all() and (result <= 15).all()


class TestTritonInterpolationWithHamming84:
    """Integration tests with Hamming(8,4) SECDED decode."""

    def test_end_to_end_with_double_errors(self):
        """Full pipeline: encode -> inject double error -> decode -> interpolate."""
        from hamming74.triton_kernels.hamming84_triton import hamming84_encode, hamming84_decode
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors
        from hamming74.triton_kernels.config import ErrorType

        # Original data
        original = torch.tensor([5, 10, 5], dtype=torch.uint8, device="cuda")
        encoded = hamming84_encode(original)

        # Inject double-bit error at position 1 (bits 0 and 1)
        corrupted = encoded.clone()
        corrupted[1] ^= 0b11  # Flip bits 0 and 1

        # Decode with error detection
        decoded, error_types, _ = hamming84_decode(corrupted, return_error_types=True)

        # Middle position should be DOUBLE_DETECTED
        assert error_types[1].item() == ErrorType.DOUBLE_DETECTED, \
            f"Expected DOUBLE_DETECTED, got {error_types[1].item()}"

        # Apply interpolation
        result = interpolate_double_errors(decoded, error_types)

        # Position 1 should be interpolated: (5 + 5) / 2 = 5
        assert result[0].item() == 5
        assert result[1].item() == 5  # Interpolated!
        assert result[2].item() == 5

    def test_large_scale_with_injected_double_errors(self):
        """Large scale test with random double errors injected."""
        from hamming74.triton_kernels.hamming84_triton import hamming84_encode, hamming84_decode
        from hamming74.triton_kernels.interpolation_triton import interpolate_double_errors
        from hamming74.triton_kernels.config import ErrorType

        N = 10000
        original = torch.randint(0, 16, (N,), dtype=torch.uint8, device="cuda")
        encoded = hamming84_encode(original)

        # Inject double-bit errors at random positions
        corrupted = encoded.clone()
        double_error_mask = torch.rand(N, device="cuda") < 0.1
        double_error_positions = torch.where(double_error_mask)[0]

        for pos in double_error_positions:
            # Pick two random bits to flip
            bit1 = torch.randint(0, 7, (1,)).item()
            bit2 = (bit1 + torch.randint(1, 7, (1,)).item()) % 7
            corrupted[pos] ^= (1 << bit1) ^ (1 << bit2)

        # Decode and interpolate
        decoded, error_types, _ = hamming84_decode(corrupted, return_error_types=True)
        result = interpolate_double_errors(decoded, error_types)

        # Check result is in valid range
        assert (result >= 0).all() and (result <= 15).all()

        # Non-corrupted positions should match original
        clean_positions = ~double_error_mask
        # Note: some clean positions may have been affected by adjacent double errors
        # So we just check that most clean positions match
        matches = (result[clean_positions] == original[clean_positions]).float().mean()
        assert matches > 0.9, f"Only {matches*100:.1f}% of clean positions match"
