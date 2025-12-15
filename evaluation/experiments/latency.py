import time
from dataclasses import dataclass, field
import torch

from hamming74 import Hamming74, Hamming84, Golay2412, INT4Quantizer
import warnings

from ..timing import (
    TimingStats,
    TimingContext,
    AggregatedTimingStats,
    run_warmup,
)
from ..constants import (
    CACHE_MODES,
    GPU_BANDWIDTH_GBPS,
    get_gpu_bandwidth,
    compute_bandwidth_efficiency,
)


@dataclass
class CodecBenchmarkConfig:
    tensor_sizes: list = field(
        default_factory=lambda: [
            (1, 256, 768),
            (8, 256, 768),
            (1, 1024, 768),
            (1, 256, 4096),
            (8, 1024, 4096),
        ]
    )

    n_iterations: int = 100
    warmup_iterations: int = 10

    codecs: list = field(
        default_factory=lambda: [
            "int4",
            "int4-hamming",
            "int4-hamming84",
            "int12-golay",
        ]
    )

    ber: float = 0.0

    device: str = "cuda"

    block_size: int = 32


@dataclass
class CodecBenchmarkResult:
    codec: str
    tensor_shape: tuple
    n_values: int
    n_iterations: int

    encode_time_mean_ms: float
    encode_time_std_ms: float
    decode_time_mean_ms: float
    decode_time_std_ms: float

    gpu_to_cpu_time_mean_ms: float = 0.0
    cpu_to_gpu_time_mean_ms: float = 0.0

    throughput_mvalues_sec: float = 0.0
    bandwidth_efficiency_pct: float = 0.0

    is_cpu_bound: bool = True
    device: str = "cuda"


@dataclass
class CodecBenchmarkReport:
    results: list = field(default_factory=list)
    config: CodecBenchmarkConfig = None

    def get_summary_table(self):
        lines = [
            "=" * 100,
            "CODEC LATENCY BENCHMARK RESULTS (CPU-BOUND BASELINE)",
            "=" * 100,
            "",
            f"{'Codec':<20} | {'Shape':<20} | {'Encode (ms)':<15} | {'Decode (ms)':<15} | {'MVal/s':<12} | {'Transfer %':<10}",
            "-" * 100,
        ]

        for r in self.results:
            shape_str = str(r.tensor_shape)
            encode_str = f"{r.encode_time_mean_ms:.3f}±{r.encode_time_std_ms:.3f}"
            decode_str = f"{r.decode_time_mean_ms:.3f}±{r.decode_time_std_ms:.3f}"
            transfer_pct = (
                100
                * (r.gpu_to_cpu_time_mean_ms + r.cpu_to_gpu_time_mean_ms)
                / max(
                    r.encode_time_mean_ms
                    + r.decode_time_mean_ms
                    + r.gpu_to_cpu_time_mean_ms
                    + r.cpu_to_gpu_time_mean_ms,
                    0.001,
                )
            )
            lines.append(
                f"{r.codec:<20} | {shape_str:<20} | {encode_str:<15} | {decode_str:<15} | {r.throughput_mvalues_sec:<12.2f} | {transfer_pct:<10.1f}"
            )

        lines.append("-" * 100)
        lines.append(
            "Note: These are CPU-bound baseline metrics. Phase 3 (Triton) will provide GPU-native metrics."
        )
        lines.append("")

        return "\n".join(lines)

    def to_dict(self):
        return {
            "results": [
                {
                    "codec": r.codec,
                    "tensor_shape": r.tensor_shape,
                    "n_values": r.n_values,
                    "n_iterations": r.n_iterations,
                    "encode_time_mean_ms": r.encode_time_mean_ms,
                    "encode_time_std_ms": r.encode_time_std_ms,
                    "decode_time_mean_ms": r.decode_time_mean_ms,
                    "decode_time_std_ms": r.decode_time_std_ms,
                    "gpu_to_cpu_time_mean_ms": r.gpu_to_cpu_time_mean_ms,
                    "cpu_to_gpu_time_mean_ms": r.cpu_to_gpu_time_mean_ms,
                    "throughput_mvalues_sec": r.throughput_mvalues_sec,
                    "bandwidth_efficiency_pct": r.bandwidth_efficiency_pct,
                    "is_cpu_bound": r.is_cpu_bound,
                    "device": r.device,
                }
                for r in self.results
            ],
            "config": {
                "tensor_sizes": self.config.tensor_sizes if self.config else [],
                "n_iterations": self.config.n_iterations if self.config else 0,
                "codecs": self.config.codecs if self.config else [],
                "ber": self.config.ber if self.config else 0.0,
            }
            if self.config
            else None,
        }


def benchmark_codec(codec_name, tensor_shape, config):
    hamming74 = Hamming74(device="cuda")
    hamming84 = Hamming84(device="cuda", on_double_error="keep")
    golay = Golay2412(device="cuda")
    quantizer = INT4Quantizer(block_size=config.block_size)

    device = config.device
    x = torch.randn(tensor_shape, dtype=torch.float16, device=device)
    n_values = x.numel()

    agg_timing = AggregatedTimingStats()

    for _ in range(config.warmup_iterations):
        x_cpu = x.cpu()
        q, scales = quantizer.quantize(x_cpu)

        if codec_name == "int4":
            pass
        elif codec_name == "int4-hamming":
            cw = hamming74.encode(q)
            _, _ = hamming74.decode(cw)
        elif codec_name == "int4-hamming84":
            cw = hamming84.encode(q)
            _ = hamming84.decode(cw)
        elif codec_name == "int12-golay":
            remainder = q.numel() % 3
            if remainder != 0:
                pad_count = 3 - remainder
                q_padded = torch.cat(
                    [q.flatten(), torch.zeros(pad_count, dtype=q.dtype)]
                )
            else:
                q_padded = q.flatten()
            q_triplets = q_padded.reshape(-1, 3)
            cw = golay.encode(q_triplets)
            _ = golay.decode(cw)

    torch.cuda.synchronize()

    for _ in range(config.n_iterations):
        trial_timing = TimingStats()

        with TimingContext(trial_timing, "gpu_to_cpu"):
            x_cpu = x.cpu()

        with TimingContext(trial_timing, "quantize"):
            q, scales = quantizer.quantize(x_cpu)

        with TimingContext(trial_timing, "encode"):
            if codec_name == "int4":
                cw = q
            elif codec_name == "int4-hamming":
                cw = hamming74.encode(q)
            elif codec_name == "int4-hamming84":
                cw = hamming84.encode(q)
            elif codec_name == "int12-golay":
                remainder = q.numel() % 3
                if remainder != 0:
                    pad_count = 3 - remainder
                    q_padded = torch.cat(
                        [q.flatten(), torch.zeros(pad_count, dtype=q.dtype)]
                    )
                else:
                    q_padded = q.flatten()
                q_triplets = q_padded.reshape(-1, 3)
                cw = golay.encode(q_triplets)

        if config.ber > 0:
            warnings.warn(
                "BER injection is not supported in GPU latency baseline; skipping injection.",
                RuntimeWarning,
            )

        with TimingContext(trial_timing, "decode"):
            if codec_name == "int4":
                q_decoded = cw
            elif codec_name == "int4-hamming":
                q_decoded, _ = hamming74.decode(cw)
            elif codec_name == "int4-hamming84":
                result = hamming84.decode(cw)
                q_decoded = result.data
            elif codec_name == "int12-golay":
                result = golay.decode(cw)
                q_decoded = result.data.flatten()
                if remainder != 0:
                    q_decoded = q_decoded[:-pad_count]
                q_decoded = q_decoded.reshape(q.shape)

        with TimingContext(trial_timing, "dequantize"):
            result = quantizer.dequantize(
                q_decoded if codec_name != "int12-golay" else q_decoded, scales
            )

        with TimingContext(trial_timing, "cpu_to_gpu"):
            result = result.to(device)

        trial_timing.values_processed = n_values
        agg_timing.add(trial_timing)

    total_codec_ns = agg_timing.total_encode_ns + agg_timing.total_decode_ns
    if total_codec_ns > 0:
        total_seconds = total_codec_ns / 1_000_000_000
        throughput = (agg_timing.total_values / 1_000_000) / total_seconds
    else:
        throughput = 0.0

    gpu_to_cpu_mean = (
        (agg_timing.total_gpu_to_cpu_ns / agg_timing.n_operations) / 1_000_000
        if agg_timing.n_operations > 0
        else 0
    )
    cpu_to_gpu_mean = (
        (agg_timing.total_cpu_to_gpu_ns / agg_timing.n_operations) / 1_000_000
        if agg_timing.n_operations > 0
        else 0
    )

    return CodecBenchmarkResult(
        codec=codec_name,
        tensor_shape=tensor_shape,
        n_values=n_values,
        n_iterations=config.n_iterations,
        encode_time_mean_ms=agg_timing.mean_encode_ms,
        encode_time_std_ms=agg_timing.std_encode_ms,
        decode_time_mean_ms=agg_timing.mean_decode_ms,
        decode_time_std_ms=agg_timing.std_decode_ms,
        gpu_to_cpu_time_mean_ms=gpu_to_cpu_mean,
        cpu_to_gpu_time_mean_ms=cpu_to_gpu_mean,
        throughput_mvalues_sec=throughput,
        bandwidth_efficiency_pct=compute_bandwidth_efficiency(throughput),
        is_cpu_bound=True,
        device=device,
    )


def run_codec_benchmarks(config=None, progress_callback=None):
    if config is None:
        config = CodecBenchmarkConfig()

    report = CodecBenchmarkReport(config=config)

    total = len(config.codecs) * len(config.tensor_sizes)
    current = 0

    for codec in config.codecs:
        for shape in config.tensor_sizes:
            if progress_callback:
                progress_callback(f"Benchmarking {codec} @ {shape}", current, total)

            result = benchmark_codec(codec, shape, config)
            report.results.append(result)
            current += 1

    return report


def run_latency_experiment(
    batch_sizes=None,
    seq_lengths=None,
    hidden_sizes=None,
    n_iterations=100,
    progress_callback=None,
):
    if batch_sizes is None:
        batch_sizes = [1, 8, 32]
    if seq_lengths is None:
        seq_lengths = [128, 512, 1024]
    if hidden_sizes is None:
        hidden_sizes = [768, 4096]

    tensor_sizes = [
        (batch, seq, hidden)
        for batch in batch_sizes
        for seq in seq_lengths
        for hidden in hidden_sizes
    ]

    config = CodecBenchmarkConfig(
        tensor_sizes=tensor_sizes,
        n_iterations=n_iterations,
        codecs=["int4", "int4-hamming", "int4-hamming84", "int12-golay"],
    )

    return run_codec_benchmarks(config, progress_callback)


if __name__ == "__main__":
    print("Running Codec Latency Benchmarks")
    print("=" * 50)
    print("NOTE: Phase 1 metrics are CPU-bound baseline.")
    print("      Phase 3 (Triton) will provide GPU-native metrics.")
    print("=" * 50)
    print()

    config = CodecBenchmarkConfig(
        tensor_sizes=[
            (1, 128, 768),
            (8, 256, 768),
        ],
        n_iterations=50,
        codecs=["int4", "int4-hamming", "int4-hamming84", "int12-golay"],
    )

    def progress(msg, curr, total):
        print(f"  [{curr+1}/{total}] {msg}")

    report = run_codec_benchmarks(config, progress)
    print()
    print(report.get_summary_table())
