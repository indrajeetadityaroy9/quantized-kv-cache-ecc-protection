import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from contextlib import contextmanager
import torch


@dataclass
class TimingStats:
    cpu_to_gpu_transfer_ns: int = 0
    gpu_to_cpu_transfer_ns: int = 0
    quantize_ns: int = 0
    encode_ns: int = 0
    decode_ns: int = 0
    dequantize_ns: int = 0

    values_processed: int = 0
    bytes_transferred: int = 0

    errors_corrected: int = 0
    errors_detected: int = 0

    @property
    def total_codec_ns(self) -> int:
        return self.encode_ns + self.decode_ns

    @property
    def total_transfer_ns(self) -> int:
        return self.cpu_to_gpu_transfer_ns + self.gpu_to_cpu_transfer_ns

    @property
    def total_ns(self) -> int:
        return (
            self.cpu_to_gpu_transfer_ns
            + self.quantize_ns
            + self.encode_ns
            + self.decode_ns
            + self.dequantize_ns
            + self.gpu_to_cpu_transfer_ns
        )

    @property
    def encode_time_ms(self) -> float:
        return self.encode_ns / 1_000_000

    @property
    def decode_time_ms(self) -> float:
        return self.decode_ns / 1_000_000

    @property
    def throughput_mvalues_sec(self) -> float:
        if self.total_codec_ns == 0:
            return 0.0
        seconds = self.total_codec_ns / 1_000_000_000
        return (self.values_processed / 1_000_000) / seconds

    @property
    def transfer_overhead_pct(self) -> float:
        if self.total_ns == 0:
            return 0.0
        return 100.0 * self.total_transfer_ns / self.total_ns


@dataclass
class AggregatedTimingStats:
    total_cpu_to_gpu_ns: int = 0
    total_gpu_to_cpu_ns: int = 0
    total_quantize_ns: int = 0
    total_encode_ns: int = 0
    total_decode_ns: int = 0
    total_dequantize_ns: int = 0
    total_values: int = 0
    total_bytes: int = 0
    total_errors_corrected: int = 0
    total_errors_detected: int = 0

    encode_times_ns: List[int] = field(default_factory=list)
    decode_times_ns: List[int] = field(default_factory=list)

    n_operations: int = 0

    is_cpu_bound: bool = True

    def add(self, stats: TimingStats) -> None:
        self.total_cpu_to_gpu_ns += stats.cpu_to_gpu_transfer_ns
        self.total_gpu_to_cpu_ns += stats.gpu_to_cpu_transfer_ns
        self.total_quantize_ns += stats.quantize_ns
        self.total_encode_ns += stats.encode_ns
        self.total_decode_ns += stats.decode_ns
        self.total_dequantize_ns += stats.dequantize_ns
        self.total_values += stats.values_processed
        self.total_bytes += stats.bytes_transferred
        self.total_errors_corrected += stats.errors_corrected
        self.total_errors_detected += stats.errors_detected

        self.encode_times_ns.append(stats.encode_ns)
        self.decode_times_ns.append(stats.decode_ns)
        self.n_operations += 1

    def reset(self) -> None:
        self.total_cpu_to_gpu_ns = 0
        self.total_gpu_to_cpu_ns = 0
        self.total_quantize_ns = 0
        self.total_encode_ns = 0
        self.total_decode_ns = 0
        self.total_dequantize_ns = 0
        self.total_values = 0
        self.total_bytes = 0
        self.total_errors_corrected = 0
        self.total_errors_detected = 0
        self.encode_times_ns = []
        self.decode_times_ns = []
        self.n_operations = 0

    @property
    def mean_encode_ms(self) -> float:
        if not self.encode_times_ns:
            return 0.0
        return (sum(self.encode_times_ns) / len(self.encode_times_ns)) / 1_000_000

    @property
    def std_encode_ms(self) -> float:
        if len(self.encode_times_ns) < 2:
            return 0.0
        mean = sum(self.encode_times_ns) / len(self.encode_times_ns)
        variance = sum((x - mean) ** 2 for x in self.encode_times_ns) / len(
            self.encode_times_ns
        )
        return (variance**0.5) / 1_000_000

    @property
    def mean_decode_ms(self) -> float:
        if not self.decode_times_ns:
            return 0.0
        return (sum(self.decode_times_ns) / len(self.decode_times_ns)) / 1_000_000

    @property
    def std_decode_ms(self) -> float:
        if len(self.decode_times_ns) < 2:
            return 0.0
        mean = sum(self.decode_times_ns) / len(self.decode_times_ns)
        variance = sum((x - mean) ** 2 for x in self.decode_times_ns) / len(
            self.decode_times_ns
        )
        return (variance**0.5) / 1_000_000

    @property
    def throughput_mvalues_sec(self) -> float:
        total_codec_ns = self.total_encode_ns + self.total_decode_ns
        if total_codec_ns == 0:
            return 0.0
        seconds = total_codec_ns / 1_000_000_000
        return (self.total_values / 1_000_000) / seconds

    @property
    def transfer_overhead_pct(self) -> float:
        total_ns = (
            self.total_cpu_to_gpu_ns
            + self.total_quantize_ns
            + self.total_encode_ns
            + self.total_decode_ns
            + self.total_dequantize_ns
            + self.total_gpu_to_cpu_ns
        )
        if total_ns == 0:
            return 0.0
        transfer_ns = self.total_cpu_to_gpu_ns + self.total_gpu_to_cpu_ns
        return 100.0 * transfer_ns / total_ns

    def to_dict(self) -> Dict:
        return {
            "n_operations": self.n_operations,
            "total_values": self.total_values,
            "is_cpu_bound": self.is_cpu_bound,
            "encode_ms_mean": self.mean_encode_ms,
            "encode_ms_std": self.std_encode_ms,
            "decode_ms_mean": self.mean_decode_ms,
            "decode_ms_std": self.std_decode_ms,
            "throughput_mvalues_sec": self.throughput_mvalues_sec,
            "transfer_overhead_pct": self.transfer_overhead_pct,
            "errors_corrected": self.total_errors_corrected,
            "errors_detected": self.total_errors_detected,
        }


class TimingContext:
    PHASE_ATTRS = {
        "cpu_to_gpu": "cpu_to_gpu_transfer_ns",
        "gpu_to_cpu": "gpu_to_cpu_transfer_ns",
        "quantize": "quantize_ns",
        "encode": "encode_ns",
        "decode": "decode_ns",
        "dequantize": "dequantize_ns",
    }

    def __init__(self, stats: Optional[TimingStats], phase: str):
        self.stats = stats
        self.phase = phase
        self.start_time: int = 0

        if phase not in self.PHASE_ATTRS:
            raise ValueError(
                f"Unknown phase: {phase}. Valid: {list(self.PHASE_ATTRS.keys())}"
            )

    def __enter__(self):
        if self.stats is not None:
            self.start_time = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stats is not None:
            elapsed_ns = time.perf_counter_ns() - self.start_time
            attr_name = self.PHASE_ATTRS[self.phase]
            current_value = getattr(self.stats, attr_name)
            setattr(self.stats, attr_name, current_value + elapsed_ns)
        return False


@contextmanager
def cuda_transfer_timer(stats: Optional[TimingStats], phase: str):
    if stats is None:
        yield
        return

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    yield
    end_event.record()

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_ns = int(elapsed_ms * 1_000_000)

    attr_name = TimingContext.PHASE_ATTRS[phase]
    current_value = getattr(stats, attr_name)
    setattr(stats, attr_name, current_value + elapsed_ns)


def run_warmup(func, n_iterations: int = 3, *args, **kwargs) -> None:
    for _ in range(n_iterations):
        func(*args, **kwargs)

    torch.cuda.synchronize()


if __name__ == "__main__":
    print("Timing Infrastructure Test")
    print("=" * 50)

    stats = TimingStats()

    with TimingContext(stats, "encode"):
        time.sleep(0.01)

    with TimingContext(stats, "decode"):
        time.sleep(0.005)

    stats.values_processed = 1_000_000

    print(f"Encode time: {stats.encode_time_ms:.3f}ms")
    print(f"Decode time: {stats.decode_time_ms:.3f}ms")
    print(f"Total codec time: {stats.total_codec_ns / 1_000_000:.3f}ms")
    print(f"Throughput: {stats.throughput_mvalues_sec:.2f} MValues/sec")

    agg = AggregatedTimingStats()

    for _ in range(5):
        trial_stats = TimingStats()

        with TimingContext(trial_stats, "encode"):
            time.sleep(0.008 + 0.002 * (torch.rand(1).item()))

        with TimingContext(trial_stats, "decode"):
            time.sleep(0.004 + 0.002 * (torch.rand(1).item()))

        trial_stats.values_processed = 1_000_000
        agg.add(trial_stats)

    print(f"\nAggregated ({agg.n_operations} trials):")
    print(f"  Encode: {agg.mean_encode_ms:.3f} +/- {agg.std_encode_ms:.3f}ms")
    print(f"  Decode: {agg.mean_decode_ms:.3f} +/- {agg.std_decode_ms:.3f}ms")
    print(f"  Throughput: {agg.throughput_mvalues_sec:.2f} MValues/sec")
    print(f"  Is CPU-bound: {agg.is_cpu_bound}")

    print("\nTiming infrastructure test passed!")
