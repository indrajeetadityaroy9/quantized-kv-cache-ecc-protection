"""
Tests for Timing Infrastructure.

Tests the timing utilities used for codec latency benchmarking.
"""

import time
import pytest
import torch

from evaluation.timing import (
    TimingStats,
    TimingContext,
    AggregatedTimingStats,
    cuda_transfer_timer,
    run_warmup,
)


class TestTimingStats:
    """Tests for TimingStats dataclass."""

    def test_initial_values(self):
        """Test that TimingStats initializes with zeros."""
        stats = TimingStats()
        assert stats.encode_ns == 0
        assert stats.decode_ns == 0
        assert stats.quantize_ns == 0
        assert stats.values_processed == 0

    def test_total_codec_ns(self):
        """Test total_codec_ns property."""
        stats = TimingStats()
        stats.encode_ns = 1_000_000  # 1ms
        stats.decode_ns = 500_000    # 0.5ms
        assert stats.total_codec_ns == 1_500_000

    def test_total_transfer_ns(self):
        """Test total_transfer_ns property."""
        stats = TimingStats()
        stats.cpu_to_gpu_transfer_ns = 100_000
        stats.gpu_to_cpu_transfer_ns = 200_000
        assert stats.total_transfer_ns == 300_000

    def test_encode_time_ms(self):
        """Test encode_time_ms conversion."""
        stats = TimingStats()
        stats.encode_ns = 5_000_000  # 5ms
        assert abs(stats.encode_time_ms - 5.0) < 0.001

    def test_decode_time_ms(self):
        """Test decode_time_ms conversion."""
        stats = TimingStats()
        stats.decode_ns = 2_500_000  # 2.5ms
        assert abs(stats.decode_time_ms - 2.5) < 0.001

    def test_throughput_mvalues_sec(self):
        """Test throughput calculation."""
        stats = TimingStats()
        stats.encode_ns = 100_000_000  # 100ms
        stats.decode_ns = 0
        stats.values_processed = 1_000_000  # 1M values in 100ms = 10 MVal/s
        assert abs(stats.throughput_mvalues_sec - 10.0) < 0.1

    def test_throughput_zero_time(self):
        """Test throughput returns 0 when no time recorded."""
        stats = TimingStats()
        stats.values_processed = 1000
        assert stats.throughput_mvalues_sec == 0.0

    def test_transfer_overhead_pct(self):
        """Test transfer overhead percentage."""
        stats = TimingStats()
        stats.cpu_to_gpu_transfer_ns = 100_000
        stats.gpu_to_cpu_transfer_ns = 100_000
        stats.encode_ns = 400_000
        stats.decode_ns = 400_000
        # Transfer = 200_000, Total = 1_000_000 -> 20%
        assert abs(stats.transfer_overhead_pct - 20.0) < 0.1


class TestTimingContext:
    """Tests for TimingContext context manager."""

    def test_timing_context_encode(self):
        """Test timing context for encode phase."""
        stats = TimingStats()
        with TimingContext(stats, "encode"):
            time.sleep(0.01)  # 10ms
        # Should be approximately 10ms
        assert stats.encode_ns > 5_000_000  # > 5ms
        assert stats.encode_ns < 50_000_000  # < 50ms

    def test_timing_context_decode(self):
        """Test timing context for decode phase."""
        stats = TimingStats()
        with TimingContext(stats, "decode"):
            time.sleep(0.005)  # 5ms
        assert stats.decode_ns > 2_000_000  # > 2ms
        assert stats.decode_ns < 20_000_000  # < 20ms

    def test_timing_context_accumulates(self):
        """Test that multiple timing contexts accumulate."""
        stats = TimingStats()
        with TimingContext(stats, "encode"):
            time.sleep(0.005)
        with TimingContext(stats, "encode"):
            time.sleep(0.005)
        # Should be approximately 10ms total
        assert stats.encode_ns > 5_000_000

    def test_timing_context_none_stats(self):
        """Test timing context with None stats (disabled timing)."""
        # Should not raise any errors
        with TimingContext(None, "encode"):
            time.sleep(0.001)

    def test_timing_context_invalid_phase(self):
        """Test timing context raises error for invalid phase."""
        stats = TimingStats()
        with pytest.raises(ValueError):
            with TimingContext(stats, "invalid_phase"):
                pass

    def test_timing_context_all_phases(self):
        """Test all valid timing phases."""
        stats = TimingStats()
        phases = ["cpu_to_gpu", "gpu_to_cpu", "quantize", "encode", "decode", "dequantize"]

        for phase in phases:
            with TimingContext(stats, phase):
                pass  # Just test it doesn't raise

        # All should have been recorded (non-zero due to context manager overhead)
        assert stats.cpu_to_gpu_transfer_ns >= 0
        assert stats.gpu_to_cpu_transfer_ns >= 0
        assert stats.quantize_ns >= 0
        assert stats.encode_ns >= 0
        assert stats.decode_ns >= 0
        assert stats.dequantize_ns >= 0


class TestAggregatedTimingStats:
    """Tests for AggregatedTimingStats."""

    def test_initial_values(self):
        """Test initial values are zeros."""
        agg = AggregatedTimingStats()
        assert agg.n_operations == 0
        assert agg.total_values == 0
        assert agg.is_cpu_bound == True

    def test_add_stats(self):
        """Test adding stats to aggregate."""
        agg = AggregatedTimingStats()

        stats1 = TimingStats()
        stats1.encode_ns = 1_000_000
        stats1.decode_ns = 500_000
        stats1.values_processed = 1000
        agg.add(stats1)

        assert agg.n_operations == 1
        assert agg.total_encode_ns == 1_000_000
        assert agg.total_decode_ns == 500_000
        assert agg.total_values == 1000

    def test_add_multiple_stats(self):
        """Test adding multiple stats."""
        agg = AggregatedTimingStats()

        for i in range(5):
            stats = TimingStats()
            stats.encode_ns = 1_000_000 * (i + 1)
            stats.values_processed = 1000
            agg.add(stats)

        assert agg.n_operations == 5
        # Sum of 1+2+3+4+5 = 15 million ns
        assert agg.total_encode_ns == 15_000_000
        assert agg.total_values == 5000

    def test_mean_encode_ms(self):
        """Test mean encode time calculation."""
        agg = AggregatedTimingStats()

        for _ in range(4):
            stats = TimingStats()
            stats.encode_ns = 10_000_000  # 10ms
            agg.add(stats)

        assert abs(agg.mean_encode_ms - 10.0) < 0.001

    def test_std_encode_ms(self):
        """Test standard deviation calculation."""
        agg = AggregatedTimingStats()

        # Add stats with known values: 8, 10, 12 (mean=10, std=~1.63)
        for val in [8_000_000, 10_000_000, 12_000_000]:
            stats = TimingStats()
            stats.encode_ns = val
            agg.add(stats)

        # std should be sqrt((4 + 0 + 4) / 3) = sqrt(8/3) ≈ 1.63 ms
        assert agg.std_encode_ms > 1.0
        assert agg.std_encode_ms < 2.0

    def test_std_single_sample(self):
        """Test std returns 0 for single sample."""
        agg = AggregatedTimingStats()
        stats = TimingStats()
        stats.encode_ns = 10_000_000
        agg.add(stats)
        assert agg.std_encode_ms == 0.0

    def test_reset(self):
        """Test reset clears all values."""
        agg = AggregatedTimingStats()

        stats = TimingStats()
        stats.encode_ns = 1_000_000
        stats.values_processed = 1000
        agg.add(stats)

        agg.reset()

        assert agg.n_operations == 0
        assert agg.total_encode_ns == 0
        assert agg.total_values == 0
        assert len(agg.encode_times_ns) == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        agg = AggregatedTimingStats()

        stats = TimingStats()
        stats.encode_ns = 10_000_000
        stats.decode_ns = 5_000_000
        stats.values_processed = 1000
        stats.errors_corrected = 10
        agg.add(stats)

        d = agg.to_dict()

        assert "n_operations" in d
        assert "encode_ms_mean" in d
        assert "decode_ms_mean" in d
        assert "throughput_mvalues_sec" in d
        assert "is_cpu_bound" in d
        assert d["n_operations"] == 1
        assert d["is_cpu_bound"] == True


class TestCudaTransferTimer:
    """Tests for CUDA transfer timing."""

    def test_cuda_transfer_timer_cpu_fallback(self):
        """Test that cuda_transfer_timer falls back to CPU timing when CUDA unavailable."""
        stats = TimingStats()

        with cuda_transfer_timer(stats, "cpu_to_gpu"):
            time.sleep(0.005)

        # Should have recorded some time
        assert stats.cpu_to_gpu_transfer_ns > 0

    def test_cuda_transfer_timer_none_stats(self):
        """Test cuda_transfer_timer with None stats."""
        # Should not raise
        with cuda_transfer_timer(None, "cpu_to_gpu"):
            time.sleep(0.001)


class TestWarmup:
    """Tests for warmup function."""

    def test_run_warmup(self):
        """Test that warmup runs function multiple times."""
        counter = [0]

        def increment():
            counter[0] += 1

        run_warmup(increment, n_iterations=3)
        assert counter[0] == 3

    def test_run_warmup_with_args(self):
        """Test warmup with function arguments."""
        results = []

        def append_value(val):
            results.append(val)

        run_warmup(append_value, n_iterations=2, val=42)
        assert results == [42, 42]


class TestIntegrationWithCodecs:
    """Integration tests with actual codec operations."""

    def test_timing_with_real_encode_decode(self):
        """Test timing infrastructure with real Hamming operations."""
        from hamming74.hamming74_sec import Hamming74

        hamming = Hamming74(device="cpu")
        data = torch.randint(0, 16, (1000,), dtype=torch.uint8)

        stats = TimingStats()

        with TimingContext(stats, "encode"):
            encoded = hamming.encode(data)

        with TimingContext(stats, "decode"):
            decoded, _ = hamming.decode(encoded)

        # Both should have recorded time
        assert stats.encode_ns > 0
        assert stats.decode_ns > 0

        # Sanity check: encode/decode should be fast for 1000 values
        assert stats.encode_time_ms < 100  # < 100ms
        assert stats.decode_time_ms < 100

    def test_aggregated_timing_multiple_runs(self):
        """Test aggregating timing over multiple runs."""
        from hamming74.hamming84_secded import Hamming84

        hamming = Hamming84(device="cpu", on_double_error="keep")
        data = torch.randint(0, 16, (1000,), dtype=torch.uint8)

        agg = AggregatedTimingStats()

        for _ in range(5):
            stats = TimingStats()

            with TimingContext(stats, "encode"):
                encoded = hamming.encode(data)

            with TimingContext(stats, "decode"):
                _ = hamming.decode(encoded)

            stats.values_processed = data.numel()
            agg.add(stats)

        assert agg.n_operations == 5
        assert agg.total_values == 5000
        assert agg.mean_encode_ms > 0
        assert agg.mean_decode_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
