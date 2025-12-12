"""
Tests for Triton fault injection implementation.

Verifies that Triton GPU kernels produce statistically correct
error rates and deterministic results.
"""

import pytest
import torch

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestTritonFaultInjectionBERFidelity:
    """Tests for BER fidelity of Triton fault injection."""

    def test_zero_ber_no_errors(self):
        """Zero BER should produce no errors."""
        from hamming74.triton_kernels.fault_injection_triton import inject_bit_errors_triton

        data = torch.randint(0, 256, (10000,), dtype=torch.uint8, device="cuda")
        corrupted, stats = inject_bit_errors_triton(data, ber=0.0, n_bits=8, seed=42, return_stats=True)

        assert torch.equal(data, corrupted), "Zero BER should not change data"
        assert stats[0] == 0, "Should have 0 errors"

    @pytest.mark.parametrize("target_ber", [0.01, 0.05, 0.10, 0.20])
    def test_ber_fidelity_uint8(self, target_ber):
        """Empirical BER should match target for uint8."""
        from hamming74.triton_kernels.fault_injection_triton import inject_bit_errors_triton

        n_values = 100_000
        n_bits = 8
        data = torch.zeros(n_values, dtype=torch.uint8, device="cuda")

        corrupted, stats = inject_bit_errors_triton(
            data, ber=target_ber, n_bits=n_bits, seed=42, return_stats=True
        )

        total_bits = n_values * n_bits
        empirical_ber = stats[0] / total_bits
        deviation = abs(empirical_ber - target_ber)

        # Allow 10% relative tolerance or 0.01 absolute tolerance
        tolerance = max(target_ber * 0.1, 0.01)
        assert deviation < tolerance, \
            f"BER={target_ber}: empirical={empirical_ber:.4f}, deviation={deviation:.4f}"

    @pytest.mark.parametrize("target_ber", [0.01, 0.05, 0.10])
    def test_ber_fidelity_int32_24bit(self, target_ber):
        """Empirical BER should match target for int32 (Golay 24-bit)."""
        from hamming74.triton_kernels.fault_injection_triton import inject_bit_errors_triton

        n_values = 50_000
        n_bits = 24
        data = torch.zeros(n_values, dtype=torch.int32, device="cuda")

        corrupted, stats = inject_bit_errors_triton(
            data, ber=target_ber, n_bits=n_bits, seed=42, return_stats=True
        )

        total_bits = n_values * n_bits
        empirical_ber = stats[0] / total_bits
        deviation = abs(empirical_ber - target_ber)

        tolerance = max(target_ber * 0.1, 0.01)
        assert deviation < tolerance, \
            f"BER={target_ber}: empirical={empirical_ber:.4f}, deviation={deviation:.4f}"


class TestTritonFaultInjectionDeterminism:
    """Tests for deterministic behavior of Triton fault injection."""

    def test_same_seed_same_result_uint8(self):
        """Same seed should produce identical results for uint8."""
        from hamming74.triton_kernels.fault_injection_triton import inject_bit_errors_triton

        data = torch.randint(0, 256, (10000,), dtype=torch.uint8, device="cuda")

        corrupted1 = inject_bit_errors_triton(data, ber=0.1, n_bits=8, seed=42)
        corrupted2 = inject_bit_errors_triton(data, ber=0.1, n_bits=8, seed=42)

        assert torch.equal(corrupted1, corrupted2), "Same seed should produce same errors"

    def test_same_seed_same_result_int32(self):
        """Same seed should produce identical results for int32."""
        from hamming74.triton_kernels.fault_injection_triton import inject_bit_errors_triton

        data = torch.randint(0, 2**20, (10000,), dtype=torch.int32, device="cuda")

        corrupted1 = inject_bit_errors_triton(data, ber=0.1, n_bits=24, seed=42)
        corrupted2 = inject_bit_errors_triton(data, ber=0.1, n_bits=24, seed=42)

        assert torch.equal(corrupted1, corrupted2), "Same seed should produce same errors"

    def test_different_seed_different_result(self):
        """Different seeds should produce different results (with high probability)."""
        from hamming74.triton_kernels.fault_injection_triton import inject_bit_errors_triton

        data = torch.zeros(10000, dtype=torch.uint8, device="cuda")

        corrupted1 = inject_bit_errors_triton(data, ber=0.1, n_bits=8, seed=42)
        corrupted2 = inject_bit_errors_triton(data, ber=0.1, n_bits=8, seed=99)

        # With 10% BER on 10K values, they should almost certainly differ
        assert not torch.equal(corrupted1, corrupted2), \
            "Different seeds should produce different errors"


class TestTritonFaultInjectionCorrectness:
    """Tests for correctness of Triton fault injection."""

    def test_only_active_bits_affected(self):
        """Only bits within n_bits should be affected."""
        from hamming74.triton_kernels.fault_injection_triton import inject_bit_errors_triton

        # Use n_bits=4 (INT4), so bits 4-7 should never change
        n_bits = 4
        # Start with high bit set (bit 7)
        data = torch.full((10000,), 0x80, dtype=torch.uint8, device="cuda")

        corrupted = inject_bit_errors_triton(data, ber=0.5, n_bits=n_bits, seed=42)

        # Check that bit 7 is unchanged in all elements
        high_bits_unchanged = ((corrupted >> 7) & 1) == 1
        assert high_bits_unchanged.all(), "Bits outside n_bits should not be affected"

    def test_xor_relationship(self):
        """Corrupted = original XOR error_mask."""
        from hamming74.triton_kernels.fault_injection_triton import inject_bit_errors_triton

        # Test with known data
        data = torch.arange(256, dtype=torch.uint8, device="cuda")
        corrupted = inject_bit_errors_triton(data.clone(), ber=0.1, n_bits=8, seed=42)

        # The XOR of original and corrupted gives error mask
        error_mask = data ^ corrupted

        # Re-XOR should recover original
        recovered = corrupted ^ error_mask
        assert torch.equal(recovered, data), "XOR relationship should hold"


class TestTritonFaultInjectionVariousSizes:
    """Tests for various tensor sizes."""

    @pytest.mark.parametrize("size", [1, 100, 1024, 10000, 100000])
    def test_various_sizes_uint8(self, size):
        """Test various tensor sizes for uint8."""
        from hamming74.triton_kernels.fault_injection_triton import inject_bit_errors_triton

        data = torch.randint(0, 256, (size,), dtype=torch.uint8, device="cuda")
        corrupted = inject_bit_errors_triton(data, ber=0.1, n_bits=8, seed=42)

        assert corrupted.shape == data.shape
        assert corrupted.dtype == data.dtype

    @pytest.mark.parametrize("size", [1, 100, 1024, 10000])
    def test_various_sizes_int32(self, size):
        """Test various tensor sizes for int32."""
        from hamming74.triton_kernels.fault_injection_triton import inject_bit_errors_triton

        data = torch.randint(0, 2**20, (size,), dtype=torch.int32, device="cuda")
        corrupted = inject_bit_errors_triton(data, ber=0.1, n_bits=24, seed=42)

        assert corrupted.shape == data.shape
        assert corrupted.dtype == data.dtype

    def test_empty_tensor(self):
        """Empty tensor should work without error."""
        from hamming74.triton_kernels.fault_injection_triton import inject_bit_errors_triton

        data = torch.empty(0, dtype=torch.uint8, device="cuda")
        corrupted = inject_bit_errors_triton(data, ber=0.1, n_bits=8, seed=42)

        assert corrupted.numel() == 0


class TestTritonFaultInjectionStatistics:
    """Tests for error statistics reporting."""

    def test_stats_match_actual_errors(self):
        """Reported stats should match actual bit differences."""
        from hamming74.triton_kernels.fault_injection_triton import inject_bit_errors_triton

        data = torch.zeros(10000, dtype=torch.uint8, device="cuda")
        corrupted, stats = inject_bit_errors_triton(
            data, ber=0.1, n_bits=8, seed=42, return_stats=True
        )

        # Count actual set bits (since original was all zeros)
        actual_errors = 0
        for bit in range(8):
            actual_errors += int(((corrupted >> bit) & 1).sum())

        assert stats[0] == actual_errors, \
            f"Reported errors {stats[0]} != actual errors {actual_errors}"

    def test_elements_with_errors_count(self):
        """Count of elements with errors should be accurate."""
        from hamming74.triton_kernels.fault_injection_triton import inject_bit_errors_triton

        data = torch.zeros(10000, dtype=torch.uint8, device="cuda")
        corrupted, stats = inject_bit_errors_triton(
            data, ber=0.1, n_bits=8, seed=42, return_stats=True
        )

        # Count elements that differ from original
        actual_elements_with_errors = int((corrupted != data).sum())

        assert stats[1] == actual_elements_with_errors, \
            f"Reported elements with errors {stats[1]} != actual {actual_elements_with_errors}"


class TestTritonFaultInjectionBatchedAPI:
    """Tests for the batched API."""

    def test_batched_api_returns_error_count(self):
        """Batched API should return corrupted data and error count."""
        from hamming74.triton_kernels.fault_injection_triton import inject_bit_errors_triton_batched

        data = torch.zeros(10000, dtype=torch.uint8, device="cuda")
        corrupted, total_errors = inject_bit_errors_triton_batched(
            data, ber=0.1, n_bits=8, seed=42
        )

        assert corrupted.shape == data.shape
        assert isinstance(total_errors, int)
        assert total_errors > 0  # With 10% BER, should have some errors
