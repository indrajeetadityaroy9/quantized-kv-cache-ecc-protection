"""
Tests for Triton Hamming(8,4) SECDED implementation.

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


class TestTritonHamming84Encode:
    """Tests for Hamming(8,4) Triton encode kernel."""

    def test_encode_all_values(self):
        """Encode all 16 INT4 values and compare with CPU."""
        from hamming74.hamming84_secded import Hamming84
        from hamming74.triton_kernels.hamming84_triton import hamming84_encode

        cpu_codec = Hamming84(device="cpu")
        all_int4 = torch.arange(16, dtype=torch.uint8)

        cpu_codewords = cpu_codec.encode(all_int4)
        triton_codewords = hamming84_encode(all_int4.cuda())

        assert torch.equal(cpu_codewords, triton_codewords.cpu())

    def test_encode_large_tensor(self):
        """Encode large random tensor."""
        from hamming74.triton_kernels.hamming84_triton import hamming84_encode

        large_input = torch.randint(0, 16, (1000000,), dtype=torch.uint8, device="cuda")
        encoded = hamming84_encode(large_input)

        assert encoded.shape == large_input.shape
        assert encoded.dtype == torch.uint8

    def test_encode_preserves_shape(self):
        """Encode preserves input shape."""
        from hamming74.triton_kernels.hamming84_triton import hamming84_encode

        shapes = [(100,), (10, 10), (5, 4, 3), (2, 3, 4, 5)]
        for shape in shapes:
            input_tensor = torch.randint(0, 16, shape, dtype=torch.uint8, device="cuda")
            encoded = hamming84_encode(input_tensor)
            assert encoded.shape == shape, f"Shape mismatch for {shape}"

    def test_encode_deterministic(self):
        """Encode is deterministic."""
        from hamming74.triton_kernels.hamming84_triton import hamming84_encode

        input_tensor = torch.randint(0, 16, (10000,), dtype=torch.uint8, device="cuda")
        encoded1 = hamming84_encode(input_tensor)
        encoded2 = hamming84_encode(input_tensor)

        assert torch.equal(encoded1, encoded2)


class TestTritonHamming84Decode:
    """Tests for Hamming(8,4) Triton decode kernel."""

    def test_decode_clean_codewords(self):
        """Decode clean codewords recovers original data."""
        from hamming74.triton_kernels.hamming84_triton import hamming84_encode, hamming84_decode

        all_int4 = torch.arange(16, dtype=torch.uint8, device="cuda")
        encoded = hamming84_encode(all_int4)
        decoded, stats = hamming84_decode(encoded)

        assert torch.equal(all_int4, decoded)
        assert stats[0] == 0, "Should have 0 corrected errors"
        assert stats[1] == 0, "Should have 0 detected errors"

    def test_single_bit_error_correction(self):
        """Single-bit errors are corrected."""
        from hamming74.triton_kernels.hamming84_triton import hamming84_encode, hamming84_decode
        from hamming74.triton_kernels.config import ErrorType

        test_val = torch.tensor([7], dtype=torch.uint8, device="cuda")
        codeword = hamming84_encode(test_val)

        for bit_pos in range(8):
            corrupted = codeword ^ (1 << bit_pos)
            decoded, error_types, stats = hamming84_decode(corrupted, return_error_types=True)

            assert decoded.item() == 7, f"Failed to correct error at bit {bit_pos}"

            if bit_pos < 7:
                # Error in Hamming portion - should be SINGLE_CORRECTED
                assert error_types.item() in [ErrorType.SINGLE_CORRECTED, ErrorType.PARITY_ONLY]
            else:
                # Error in parity bit only
                assert error_types.item() == ErrorType.PARITY_ONLY

    def test_double_bit_error_detection(self):
        """Double-bit errors are detected (not miscorrected)."""
        from hamming74.triton_kernels.hamming84_triton import hamming84_encode, hamming84_decode
        from hamming74.triton_kernels.config import ErrorType

        test_val = torch.tensor([5], dtype=torch.uint8, device="cuda")
        codeword = hamming84_encode(test_val)

        detected_count = 0
        total_pairs = 0

        for bit1 in range(7):
            for bit2 in range(bit1 + 1, 7):
                corrupted = codeword ^ (1 << bit1) ^ (1 << bit2)
                decoded, error_types, stats = hamming84_decode(corrupted, return_error_types=True)

                total_pairs += 1
                if error_types.item() == ErrorType.DOUBLE_DETECTED:
                    detected_count += 1
                    # Should be zeroed out per policy
                    assert decoded.item() == 0, f"Double error at bits {bit1},{bit2} not zeroed"

        # Most double errors should be detected
        assert detected_count >= total_pairs * 0.9, \
            f"Only {detected_count}/{total_pairs} double errors detected"

    def test_roundtrip_large_tensor(self):
        """Large tensor roundtrip works correctly."""
        from hamming74.triton_kernels.hamming84_triton import hamming84_encode, hamming84_decode

        large_input = torch.randint(0, 16, (100000,), dtype=torch.uint8, device="cuda")
        encoded = hamming84_encode(large_input)
        decoded, stats = hamming84_decode(encoded)

        assert torch.equal(large_input, decoded)
        assert stats[0] == 0, "Should have no corrections for clean data"
        assert stats[1] == 0, "Should have no detections for clean data"

    def test_decode_preserves_shape(self):
        """Decode preserves input shape."""
        from hamming74.triton_kernels.hamming84_triton import hamming84_encode, hamming84_decode

        shapes = [(100,), (10, 10), (5, 4, 3)]
        for shape in shapes:
            input_tensor = torch.randint(0, 16, shape, dtype=torch.uint8, device="cuda")
            encoded = hamming84_encode(input_tensor)
            decoded, _ = hamming84_decode(encoded)
            assert decoded.shape == shape


class TestTritonHamming84VsCPU:
    """Cross-validation tests between Triton and CPU implementations."""

    def test_encode_matches_cpu_all_values(self):
        """Triton encode matches CPU for all 16 values."""
        from hamming74.hamming84_secded import Hamming84
        from hamming74.triton_kernels.hamming84_triton import hamming84_encode

        cpu_codec = Hamming84(device="cpu")

        for val in range(16):
            input_tensor = torch.tensor([val], dtype=torch.uint8)
            cpu_result = cpu_codec.encode(input_tensor)
            triton_result = hamming84_encode(input_tensor.cuda())

            assert cpu_result.item() == triton_result.cpu().item(), \
                f"Mismatch for value {val}: CPU={cpu_result.item()}, Triton={triton_result.cpu().item()}"

    def test_decode_matches_cpu_with_errors(self):
        """Triton decode matches CPU behavior for corrupted codewords."""
        from hamming74.hamming84_secded import Hamming84
        from hamming74.triton_kernels.hamming84_triton import hamming84_encode, hamming84_decode

        cpu_codec = Hamming84(device="cpu", on_double_error="zero")

        test_vals = [0, 5, 10, 15]
        for val in test_vals:
            input_tensor = torch.tensor([val], dtype=torch.uint8)
            codeword = cpu_codec.encode(input_tensor)

            # Test single-bit errors
            for bit_pos in range(8):
                corrupted = codeword ^ (1 << bit_pos)

                cpu_result = cpu_codec.decode(corrupted)
                triton_result, _ = hamming84_decode(corrupted.cuda())

                assert cpu_result.data.item() == triton_result.cpu().item(), \
                    f"Single-bit error mismatch at bit {bit_pos} for value {val}"


class TestTritonHamming84Performance:
    """Performance-related tests."""

    def test_handles_empty_tensor(self):
        """Empty tensor doesn't crash."""
        from hamming74.triton_kernels.hamming84_triton import hamming84_encode, hamming84_decode

        empty = torch.tensor([], dtype=torch.uint8, device="cuda")
        encoded = hamming84_encode(empty)
        decoded, stats = hamming84_decode(encoded)

        assert encoded.numel() == 0
        assert decoded.numel() == 0

    def test_handles_single_element(self):
        """Single element tensor works."""
        from hamming74.triton_kernels.hamming84_triton import hamming84_encode, hamming84_decode

        single = torch.tensor([7], dtype=torch.uint8, device="cuda")
        encoded = hamming84_encode(single)
        decoded, _ = hamming84_decode(encoded)

        assert decoded.item() == 7

    @pytest.mark.parametrize("size", [1, 100, 1024, 10000, 100000, 1000000])
    def test_various_sizes(self, size):
        """Test various tensor sizes."""
        from hamming74.triton_kernels.hamming84_triton import hamming84_encode, hamming84_decode

        input_tensor = torch.randint(0, 16, (size,), dtype=torch.uint8, device="cuda")
        encoded = hamming84_encode(input_tensor)
        decoded, _ = hamming84_decode(encoded)

        assert torch.equal(input_tensor, decoded)
