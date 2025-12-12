"""
Tests for Triton Hamming(7,4) SEC implementation.

Verifies that Triton GPU kernels produce identical results to the
CPU reference implementation.
"""

import pytest
import torch

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestTritonHamming74Encode:
    """Tests for Hamming(7,4) Triton encode kernel."""

    def test_encode_all_values(self):
        """Encode all 16 INT4 values and compare with CPU."""
        from hamming74.hamming74_sec import Hamming74
        from hamming74.triton_kernels.hamming74_triton import hamming74_encode

        cpu_codec = Hamming74(device="cpu")
        all_int4 = torch.arange(16, dtype=torch.uint8)

        cpu_codewords = cpu_codec.encode(all_int4)
        triton_codewords = hamming74_encode(all_int4.cuda())

        assert torch.equal(cpu_codewords, triton_codewords.cpu())

    def test_encode_large_tensor(self):
        """Encode large random tensor."""
        from hamming74.triton_kernels.hamming74_triton import hamming74_encode

        large_input = torch.randint(0, 16, (1000000,), dtype=torch.uint8, device="cuda")
        encoded = hamming74_encode(large_input)

        assert encoded.shape == large_input.shape
        assert encoded.dtype == torch.uint8

    def test_encode_preserves_shape(self):
        """Encode preserves input shape."""
        from hamming74.triton_kernels.hamming74_triton import hamming74_encode

        shapes = [(100,), (10, 10), (5, 4, 3), (2, 3, 4, 5)]
        for shape in shapes:
            input_tensor = torch.randint(0, 16, shape, dtype=torch.uint8, device="cuda")
            encoded = hamming74_encode(input_tensor)
            assert encoded.shape == shape, f"Shape mismatch for {shape}"

    def test_encode_deterministic(self):
        """Encode is deterministic."""
        from hamming74.triton_kernels.hamming74_triton import hamming74_encode

        input_tensor = torch.randint(0, 16, (10000,), dtype=torch.uint8, device="cuda")
        encoded1 = hamming74_encode(input_tensor)
        encoded2 = hamming74_encode(input_tensor)

        assert torch.equal(encoded1, encoded2)

    def test_encode_msb_is_zero(self):
        """Encoded values have MSB = 0 (7-bit codeword in 8-bit storage)."""
        from hamming74.triton_kernels.hamming74_triton import hamming74_encode

        all_int4 = torch.arange(16, dtype=torch.uint8, device="cuda")
        encoded = hamming74_encode(all_int4)

        # MSB should always be 0 for 7-bit codewords
        assert (encoded & 0x80).sum() == 0, "MSB should be 0 for all codewords"


class TestTritonHamming74Decode:
    """Tests for Hamming(7,4) Triton decode kernel."""

    def test_decode_clean_codewords(self):
        """Decode clean codewords recovers original data."""
        from hamming74.triton_kernels.hamming74_triton import hamming74_encode, hamming74_decode

        all_int4 = torch.arange(16, dtype=torch.uint8, device="cuda")
        encoded = hamming74_encode(all_int4)
        decoded, stats = hamming74_decode(encoded)

        assert torch.equal(all_int4, decoded)
        assert stats[0] == 0, "Should have 0 corrected errors"

    def test_single_bit_error_correction(self):
        """Single-bit errors are corrected for all 7 positions."""
        from hamming74.triton_kernels.hamming74_triton import hamming74_encode, hamming74_decode

        test_val = torch.tensor([7], dtype=torch.uint8, device="cuda")
        codeword = hamming74_encode(test_val)

        for bit_pos in range(7):
            corrupted = codeword ^ (1 << bit_pos)
            decoded, error_detected, stats = hamming74_decode(corrupted, return_error_detected=True)

            assert decoded.item() == 7, f"Failed to correct error at bit {bit_pos}"
            assert error_detected.item() == 1, f"Error at bit {bit_pos} not detected"

    def test_single_bit_all_values(self):
        """Single-bit error correction works for all 16 INT4 values."""
        from hamming74.triton_kernels.hamming74_triton import hamming74_encode, hamming74_decode

        for val in range(16):
            test_val = torch.tensor([val], dtype=torch.uint8, device="cuda")
            codeword = hamming74_encode(test_val)

            for bit_pos in range(7):
                corrupted = codeword ^ (1 << bit_pos)
                decoded, _ = hamming74_decode(corrupted)

                assert decoded.item() == val, \
                    f"Failed to correct value {val} with error at bit {bit_pos}"

    def test_roundtrip_large_tensor(self):
        """Large tensor roundtrip works correctly."""
        from hamming74.triton_kernels.hamming74_triton import hamming74_encode, hamming74_decode

        large_input = torch.randint(0, 16, (100000,), dtype=torch.uint8, device="cuda")
        encoded = hamming74_encode(large_input)
        decoded, stats = hamming74_decode(encoded)

        assert torch.equal(large_input, decoded)
        assert stats[0] == 0, "Should have no corrections for clean data"

    def test_decode_preserves_shape(self):
        """Decode preserves input shape."""
        from hamming74.triton_kernels.hamming74_triton import hamming74_encode, hamming74_decode

        shapes = [(100,), (10, 10), (5, 4, 3)]
        for shape in shapes:
            input_tensor = torch.randint(0, 16, shape, dtype=torch.uint8, device="cuda")
            encoded = hamming74_encode(input_tensor)
            decoded, _ = hamming74_decode(encoded)
            assert decoded.shape == shape

    def test_error_statistics(self):
        """Error statistics are correctly computed."""
        from hamming74.triton_kernels.hamming74_triton import hamming74_encode, hamming74_decode

        # Create clean data
        large_input = torch.randint(0, 16, (100000,), dtype=torch.uint8, device="cuda")
        encoded = hamming74_encode(large_input)

        # Inject single-bit errors into exactly half the codewords
        error_mask = torch.zeros(100000, dtype=torch.bool, device="cuda")
        error_mask[::2] = True  # Every other codeword
        bit_positions = torch.randint(0, 7, (50000,), device="cuda")
        error_pattern = (1 << bit_positions).to(torch.uint8)
        encoded_with_errors = encoded.clone()
        encoded_with_errors[::2] ^= error_pattern

        decoded, stats = hamming74_decode(encoded_with_errors)

        assert stats[0] == 50000, f"Expected 50000 corrections, got {stats[0]}"
        assert torch.equal(decoded, large_input), "Corrected data doesn't match original"


class TestTritonHamming74VsCPU:
    """Cross-validation tests between Triton and CPU implementations."""

    def test_encode_matches_cpu_all_values(self):
        """Triton encode matches CPU for all 16 values."""
        from hamming74.hamming74_sec import Hamming74
        from hamming74.triton_kernels.hamming74_triton import hamming74_encode

        cpu_codec = Hamming74(device="cpu")

        for val in range(16):
            input_tensor = torch.tensor([val], dtype=torch.uint8)
            cpu_result = cpu_codec.encode(input_tensor)
            triton_result = hamming74_encode(input_tensor.cuda())

            assert cpu_result.item() == triton_result.cpu().item(), \
                f"Mismatch for value {val}: CPU={cpu_result.item()}, Triton={triton_result.cpu().item()}"

    def test_decode_matches_cpu_with_errors(self):
        """Triton decode matches CPU behavior for corrupted codewords."""
        from hamming74.hamming74_sec import Hamming74
        from hamming74.triton_kernels.hamming74_triton import hamming74_encode, hamming74_decode

        cpu_codec = Hamming74(device="cpu")

        test_vals = [0, 5, 10, 15]
        for val in test_vals:
            input_tensor = torch.tensor([val], dtype=torch.uint8)
            codeword = cpu_codec.encode(input_tensor)

            # Test single-bit errors
            for bit_pos in range(7):
                corrupted = codeword ^ (1 << bit_pos)

                # CPU decode returns (int4_values, errors_detected) tuple
                cpu_decoded, cpu_errors = cpu_codec.decode(corrupted)
                triton_result, _ = hamming74_decode(corrupted.cuda())

                assert cpu_decoded.item() == triton_result.cpu().item(), \
                    f"Single-bit error mismatch at bit {bit_pos} for value {val}"


class TestTritonHamming74Performance:
    """Performance-related tests."""

    def test_handles_empty_tensor(self):
        """Empty tensor doesn't crash."""
        from hamming74.triton_kernels.hamming74_triton import hamming74_encode, hamming74_decode

        empty = torch.tensor([], dtype=torch.uint8, device="cuda")
        encoded = hamming74_encode(empty)
        decoded, stats = hamming74_decode(encoded)

        assert encoded.numel() == 0
        assert decoded.numel() == 0

    def test_handles_single_element(self):
        """Single element tensor works."""
        from hamming74.triton_kernels.hamming74_triton import hamming74_encode, hamming74_decode

        single = torch.tensor([7], dtype=torch.uint8, device="cuda")
        encoded = hamming74_encode(single)
        decoded, _ = hamming74_decode(encoded)

        assert decoded.item() == 7

    @pytest.mark.parametrize("size", [1, 100, 1024, 10000, 100000, 1000000])
    def test_various_sizes(self, size):
        """Test various tensor sizes."""
        from hamming74.triton_kernels.hamming74_triton import hamming74_encode, hamming74_decode

        input_tensor = torch.randint(0, 16, (size,), dtype=torch.uint8, device="cuda")
        encoded = hamming74_encode(input_tensor)
        decoded, _ = hamming74_decode(encoded)

        assert torch.equal(input_tensor, decoded)
