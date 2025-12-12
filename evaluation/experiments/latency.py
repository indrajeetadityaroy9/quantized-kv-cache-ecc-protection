"""
Latency Benchmarking Experiments.

Provides isolated codec latency benchmarks and serving simulation for
measuring encode/decode throughput and CPU/GPU transfer overhead.

NOTE: Phase 1 metrics are Baseline / CPU-Bound. Since Hamming codecs
currently run on CPU, throughput numbers are bottlenecked by CPU-GPU
transfers. Phase 3 (Triton GPU codecs) will provide production-grade
metrics. Do not conflate CPU-bound and GPU-native numbers in analysis.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch

from hamming74.hamming74_sec import Hamming74
from hamming74.hamming84_secded import Hamming84
from hamming74.golay import Golay2412
from hamming74.quantization import INT4Quantizer
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
    """Configuration for isolated codec benchmarks."""

    # Tensor sizes to benchmark
    tensor_sizes: List[Tuple[int, ...]] = field(
        default_factory=lambda: [
            (1, 256, 768),      # Single batch, short seq, GPT-2 hidden
            (8, 256, 768),      # 8 batch, short seq
            (1, 1024, 768),     # Single batch, long seq
            (1, 256, 4096),     # Single batch, short seq, LLaMA hidden
            (8, 1024, 4096),    # 8 batch, long seq, LLaMA hidden
        ]
    )

    # Number of iterations per benchmark
    n_iterations: int = 100
    warmup_iterations: int = 10

    # Which codecs to benchmark
    codecs: List[str] = field(
        default_factory=lambda: ["int4", "int4-hamming", "int4-hamming84", "int12-golay"]
    )

    # BER for error injection (0 = no errors, just timing)
    ber: float = 0.0

    # Device for GPU transfer benchmarks
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Block size for quantization
    block_size: int = 32


@dataclass
class CodecBenchmarkResult:
    """Results from a single codec benchmark."""

    codec: str
    tensor_shape: Tuple[int, ...]
    n_values: int
    n_iterations: int

    # Timing results (milliseconds)
    encode_time_mean_ms: float
    encode_time_std_ms: float
    decode_time_mean_ms: float
    decode_time_std_ms: float

    # Transfer times (CPU/GPU)
    gpu_to_cpu_time_mean_ms: float = 0.0
    cpu_to_gpu_time_mean_ms: float = 0.0

    # Throughput
    throughput_mvalues_sec: float = 0.0
    bandwidth_efficiency_pct: float = 0.0

    # Metadata
    is_cpu_bound: bool = True
    device: str = "cpu"


@dataclass
class CodecBenchmarkReport:
    """Complete report from codec benchmarking."""

    results: List[CodecBenchmarkResult] = field(default_factory=list)
    config: Optional[CodecBenchmarkConfig] = None

    def get_summary_table(self) -> str:
        """Generate summary table as formatted string."""
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
            transfer_pct = 100 * (r.gpu_to_cpu_time_mean_ms + r.cpu_to_gpu_time_mean_ms) / max(
                r.encode_time_mean_ms + r.decode_time_mean_ms + r.gpu_to_cpu_time_mean_ms + r.cpu_to_gpu_time_mean_ms,
                0.001
            )
            lines.append(
                f"{r.codec:<20} | {shape_str:<20} | {encode_str:<15} | {decode_str:<15} | {r.throughput_mvalues_sec:<12.2f} | {transfer_pct:<10.1f}"
            )

        lines.append("-" * 100)
        lines.append("Note: These are CPU-bound baseline metrics. Phase 3 (Triton) will provide GPU-native metrics.")
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
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
            } if self.config else None,
        }


def benchmark_codec(
    codec_name: str,
    tensor_shape: Tuple[int, ...],
    config: CodecBenchmarkConfig,
) -> CodecBenchmarkResult:
    """
    Benchmark a single codec on a specific tensor shape.

    Args:
        codec_name: Name of codec to benchmark (e.g., "int4-hamming")
        tensor_shape: Shape of tensor to benchmark
        config: Benchmark configuration

    Returns:
        CodecBenchmarkResult with timing statistics
    """
    # Initialize codecs
    hamming74 = Hamming74(device="cpu")
    hamming84 = Hamming84(device="cpu", on_double_error="keep")
    golay = Golay2412(device="cpu")
    quantizer = INT4Quantizer(block_size=config.block_size)

    # Create test tensor
    device = config.device if torch.cuda.is_available() else "cpu"
    x = torch.randn(tensor_shape, dtype=torch.float16, device=device)
    n_values = x.numel()

    # Aggregated timing
    agg_timing = AggregatedTimingStats()

    # Warmup
    for _ in range(config.warmup_iterations):
        x_cpu = x.cpu()
        q, scales = quantizer.quantize(x_cpu)

        if codec_name == "int4":
            pass  # No encoding
        elif codec_name == "int4-hamming":
            cw = hamming74.encode(q)
            _, _ = hamming74.decode(cw)
        elif codec_name == "int4-hamming84":
            cw = hamming84.encode(q)
            _ = hamming84.decode(cw)
        elif codec_name == "int12-golay":
            # Pad for Golay
            remainder = q.numel() % 3
            if remainder != 0:
                pad_count = 3 - remainder
                q_padded = torch.cat([q.flatten(), torch.zeros(pad_count, dtype=q.dtype)])
            else:
                q_padded = q.flatten()
            q_triplets = q_padded.reshape(-1, 3)
            cw = golay.encode(q_triplets)
            _ = golay.decode(cw)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark iterations
    for _ in range(config.n_iterations):
        trial_timing = TimingStats()

        # GPU to CPU transfer
        with TimingContext(trial_timing, "gpu_to_cpu"):
            x_cpu = x.cpu()

        # Quantize
        with TimingContext(trial_timing, "quantize"):
            q, scales = quantizer.quantize(x_cpu)

        # Encode
        with TimingContext(trial_timing, "encode"):
            if codec_name == "int4":
                cw = q  # No encoding
            elif codec_name == "int4-hamming":
                cw = hamming74.encode(q)
            elif codec_name == "int4-hamming84":
                cw = hamming84.encode(q)
            elif codec_name == "int12-golay":
                remainder = q.numel() % 3
                if remainder != 0:
                    pad_count = 3 - remainder
                    q_padded = torch.cat([q.flatten(), torch.zeros(pad_count, dtype=q.dtype)])
                else:
                    q_padded = q.flatten()
                q_triplets = q_padded.reshape(-1, 3)
                cw = golay.encode(q_triplets)

        # Inject errors if configured (GPU Triton path only)
        if config.ber > 0:
            warnings.warn("BER injection is not supported in GPU latency baseline; skipping injection.", RuntimeWarning)

        # Decode
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

        # Dequantize
        with TimingContext(trial_timing, "dequantize"):
            result = quantizer.dequantize(q_decoded if codec_name != "int12-golay" else q_decoded, scales)

        # CPU to GPU transfer
        with TimingContext(trial_timing, "cpu_to_gpu"):
            result = result.to(device)

        trial_timing.values_processed = n_values
        agg_timing.add(trial_timing)

    # Compute throughput
    total_codec_ns = agg_timing.total_encode_ns + agg_timing.total_decode_ns
    if total_codec_ns > 0:
        total_seconds = total_codec_ns / 1_000_000_000
        throughput = (agg_timing.total_values / 1_000_000) / total_seconds
    else:
        throughput = 0.0

    # Compute transfer times
    gpu_to_cpu_mean = (agg_timing.total_gpu_to_cpu_ns / agg_timing.n_operations) / 1_000_000 if agg_timing.n_operations > 0 else 0
    cpu_to_gpu_mean = (agg_timing.total_cpu_to_gpu_ns / agg_timing.n_operations) / 1_000_000 if agg_timing.n_operations > 0 else 0

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


def run_codec_benchmarks(
    config: CodecBenchmarkConfig = None,
    progress_callback=None,
) -> CodecBenchmarkReport:
    """
    Run complete codec benchmark suite.

    Args:
        config: Benchmark configuration (uses defaults if None)
        progress_callback: Optional callback(message, current, total) for progress

    Returns:
        CodecBenchmarkReport with all results
    """
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
    batch_sizes: List[int] = None,
    seq_lengths: List[int] = None,
    hidden_sizes: List[int] = None,
    n_iterations: int = 100,
    progress_callback=None,
) -> CodecBenchmarkReport:
    """
    Run latency experiment with common LLM configurations.

    This is the main entry point for latency benchmarking.

    Args:
        batch_sizes: List of batch sizes (default: [1, 8, 32])
        seq_lengths: List of sequence lengths (default: [128, 512, 1024])
        hidden_sizes: List of hidden sizes (default: [768, 4096])
        n_iterations: Number of iterations per benchmark
        progress_callback: Optional progress callback

    Returns:
        CodecBenchmarkReport with all results
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 32]
    if seq_lengths is None:
        seq_lengths = [128, 512, 1024]
    if hidden_sizes is None:
        hidden_sizes = [768, 4096]  # GPT-2, LLaMA-8B

    # Generate all tensor shapes
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

    # Quick benchmark with small config
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
