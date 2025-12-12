"""
Tests for Triton Golay(24,12) implementation.

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


class TestTritonGolayEncode:
    """Tests for Golay(24,12) Triton encode kernel."""

    def test_encode_sample_triplets(self):
        """Encode sample triplets and compare with CPU."""
        from hamming74.golay import Golay2412
        from hamming74.triton_kernels.golay_triton import golay_encode

        cpu_codec = Golay2412(device="cpu")
        triplets = torch.tensor([
            [5, 10, 3],
            [0, 0, 0],
            [15, 15, 15],
            [7, 8, 9],
        ], dtype=torch.uint8)

        cpu_codewords = cpu_codec.encode(triplets)
        triton_codewords = golay_encode(triplets.cuda())

        # CPU uses int64, Triton uses int32 (both hold 24-bit values)
        assert torch.equal(cpu_codewords.to(torch.int32), triton_codewords.cpu())

    def test_encode_all_possible_first_nibble(self):
        """Encode all 16 values for first nibble."""
        from hamming74.golay import Golay2412
        from hamming74.triton_kernels.golay_triton import golay_encode

        cpu_codec = Golay2412(device="cpu")

        for val in range(16):
            triplet = torch.tensor([[val, 0, 0]], dtype=torch.uint8)
            cpu_cw = cpu_codec.encode(triplet)
            triton_cw = golay_encode(triplet.cuda())
            assert cpu_cw.to(torch.int32).item() == triton_cw.cpu().item(), \
                f"Mismatch for first nibble {val}"

    def test_encode_large_tensor(self):
        """Encode large random tensor."""
        from hamming74.triton_kernels.golay_triton import golay_encode

        large_triplets = torch.randint(0, 16, (100000, 3), dtype=torch.uint8, device="cuda")
        encoded = golay_encode(large_triplets)

        assert encoded.shape == (100000,)
        assert encoded.dtype == torch.int32

    def test_encode_deterministic(self):
        """Encode is deterministic."""
        from hamming74.triton_kernels.golay_triton import golay_encode

        triplets = torch.randint(0, 16, (1000, 3), dtype=torch.uint8, device="cuda")
        encoded1 = golay_encode(triplets)
        encoded2 = golay_encode(triplets)

        assert torch.equal(encoded1, encoded2)


class TestTritonGolayDecode:
    """Tests for Golay(24,12) Triton decode kernel."""

    def test_decode_clean_codewords(self):
        """Decode clean codewords recovers original data."""
        from hamming74.triton_kernels.golay_triton import golay_encode, golay_decode

        triplets = torch.tensor([
            [5, 10, 3],
            [0, 0, 0],
            [15, 15, 15],
        ], dtype=torch.uint8, device="cuda")

        encoded = golay_encode(triplets)
        decoded, stats = golay_decode(encoded)

        assert torch.equal(triplets, decoded)
        assert stats[0] == 0, "Should have 0 corrected errors"
        assert stats[1] == 0, "Should have 0 uncorrectable errors"

    def test_single_bit_error_correction(self):
        """Single-bit errors are corrected (all 24 positions)."""
        from hamming74.triton_kernels.golay_triton import golay_encode, golay_decode

        triplet = torch.tensor([[7, 8, 9]], dtype=torch.uint8, device="cuda")
        codeword = golay_encode(triplet)

        for bit_pos in range(24):
            corrupted = codeword ^ (1 << bit_pos)
            decoded, stats = golay_decode(corrupted)

            assert torch.equal(decoded, triplet), \
                f"Failed to correct single-bit error at bit {bit_pos}"
            assert stats[0] == 1, f"Should report 1 error corrected for bit {bit_pos}"

    def test_double_bit_error_correction(self):
        """Double-bit errors are corrected."""
        from hamming74.triton_kernels.golay_triton import golay_encode, golay_decode

        triplet = torch.tensor([[5, 10, 3]], dtype=torch.uint8, device="cuda")
        codeword = golay_encode(triplet)

        # Test sample of double-bit error combinations
        test_cases = [
            (0, 1), (0, 23), (5, 15), (10, 20), (7, 19)
        ]
        for bit1, bit2 in test_cases:
            corrupted = codeword ^ (1 << bit1) ^ (1 << bit2)
            decoded, stats = golay_decode(corrupted)

            assert torch.equal(decoded, triplet), \
                f"Failed to correct double-bit error at bits {bit1},{bit2}"
            assert stats[0] == 2, \
                f"Should report 2 errors corrected for bits {bit1},{bit2}"

    def test_triple_bit_error_correction(self):
        """Triple-bit errors are corrected (Golay's strength)."""
        from hamming74.triton_kernels.golay_triton import golay_encode, golay_decode

        triplet = torch.tensor([[5, 10, 3]], dtype=torch.uint8, device="cuda")
        codeword = golay_encode(triplet)

        # Test sample of triple-bit error combinations
        test_cases = [
            (0, 7, 15), (1, 8, 16), (3, 11, 19), (5, 13, 21)
        ]
        for bit1, bit2, bit3 in test_cases:
            corrupted = codeword ^ (1 << bit1) ^ (1 << bit2) ^ (1 << bit3)
            decoded, stats = golay_decode(corrupted)

            assert torch.equal(decoded, triplet), \
                f"Failed to correct triple-bit error at bits {bit1},{bit2},{bit3}"
            assert stats[0] == 3, \
                f"Should report 3 errors corrected for bits {bit1},{bit2},{bit3}"

    def test_roundtrip_large_tensor(self):
        """Large tensor roundtrip works correctly."""
        from hamming74.triton_kernels.golay_triton import golay_encode, golay_decode

        large_triplets = torch.randint(0, 16, (10000, 3), dtype=torch.uint8, device="cuda")
        encoded = golay_encode(large_triplets)
        decoded, stats = golay_decode(encoded)

        assert torch.equal(large_triplets, decoded)
        assert stats[0] == 0, "Should have no corrections for clean data"
        assert stats[1] == 0, "Should have no uncorrectable errors"


class TestTritonGolayVsCPU:
    """Cross-validation tests between Triton and CPU implementations."""

    def test_encode_matches_cpu_exhaustive_sample(self):
        """Triton encode matches CPU for many triplet combinations."""
        from hamming74.golay import Golay2412
        from hamming74.triton_kernels.golay_triton import golay_encode

        cpu_codec = Golay2412(device="cpu")

        # Test 1000 random triplets
        triplets = torch.randint(0, 16, (1000, 3), dtype=torch.uint8)

        cpu_codewords = cpu_codec.encode(triplets)
        triton_codewords = golay_encode(triplets.cuda())

        assert torch.equal(cpu_codewords.to(torch.int32), triton_codewords.cpu()), \
            "Triton encode does not match CPU for random triplets"

    def test_decode_matches_cpu_with_errors(self):
        """Triton decode matches CPU behavior for corrupted codewords."""
        from hamming74.golay import Golay2412
        from hamming74.triton_kernels.golay_triton import golay_encode, golay_decode

        cpu_codec = Golay2412(device="cpu")

        triplet = torch.tensor([[5, 10, 3]], dtype=torch.uint8)
        codeword = cpu_codec.encode(triplet)

        # Test single-bit errors
        for bit_pos in range(24):
            corrupted = codeword ^ (1 << bit_pos)

            cpu_result = cpu_codec.decode(corrupted.to(torch.int64))
            triton_decoded, _ = golay_decode(corrupted.to(torch.int32).cuda())

            assert torch.equal(cpu_result.data, triton_decoded.cpu()), \
                f"Single-bit error mismatch at bit {bit_pos}"


class TestTritonGolayPerformance:
    """Performance-related tests."""

    def test_handles_empty_tensor(self):
        """Empty tensor doesn't crash."""
        from hamming74.triton_kernels.golay_triton import golay_encode, golay_decode

        empty = torch.tensor([], dtype=torch.uint8, device="cuda").view(0, 3)
        encoded = golay_encode(empty)
        decoded, stats = golay_decode(encoded)

        assert encoded.numel() == 0
        assert decoded.numel() == 0

    def test_handles_single_triplet(self):
        """Single triplet works."""
        from hamming74.triton_kernels.golay_triton import golay_encode, golay_decode

        single = torch.tensor([[7, 8, 9]], dtype=torch.uint8, device="cuda")
        encoded = golay_encode(single)
        decoded, _ = golay_decode(encoded)

        assert torch.equal(single, decoded)

    @pytest.mark.parametrize("size", [1, 100, 1024, 10000, 100000])
    def test_various_sizes(self, size):
        """Test various tensor sizes."""
        from hamming74.triton_kernels.golay_triton import golay_encode, golay_decode

        triplets = torch.randint(0, 16, (size, 3), dtype=torch.uint8, device="cuda")
        encoded = golay_encode(triplets)
        decoded, _ = golay_decode(encoded)

        assert torch.equal(triplets, decoded)

    def test_syndrome_table_caching(self):
        """Syndrome table is cached across calls."""
        from hamming74.triton_kernels.golay_triton import golay_decode, _syndrome_table_cache

        triplets = torch.randint(0, 16, (100, 3), dtype=torch.uint8, device="cuda")
        from hamming74.triton_kernels.golay_triton import golay_encode
        encoded = golay_encode(triplets)

        # First decode builds table
        _ = golay_decode(encoded)
        assert "cuda:0" in _syndrome_table_cache or "cuda" in str(_syndrome_table_cache)

        # Second decode reuses cached table
        decoded, _ = golay_decode(encoded)
        assert torch.equal(triplets, decoded)
