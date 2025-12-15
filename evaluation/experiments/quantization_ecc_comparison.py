"""
Comprehensive comparison of Quantization Backends × ECC Methods

This benchmark tests all combinations of:
- Quantization backends: block_absmax, per_token, per_channel, kivi, kivi_symmetric, group_wise
- ECC methods: none (int4), hamming74, hamming84, golay

Metrics measured:
- Quantization MSE (before ECC)
- Post-ECC MSE (after error injection and correction)
- Error correction rate
- Error detection rate
"""

import os
import sys
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class BenchmarkConfig:
    """Configuration for the quantization+ECC benchmark."""
    batch_size: int = 4
    num_heads: int = 8
    seq_length: int = 128
    head_dim: int = 64

    ber_levels: list = field(default_factory=lambda: [0.0, 1e-4, 1e-3, 1e-2])
    num_trials: int = 3
    seed: int = 42

    quantization_backends: list = field(default_factory=lambda: [
        "block_absmax",
        "per_token",
        "per_channel",
        "kivi",
        "kivi_symmetric",
        "group_wise",
    ])

    ecc_methods: list = field(default_factory=lambda: [
        "none",
        "hamming74",
        "hamming84",
        "golay",
    ])


@dataclass
class TrialResult:
    """Result of a single trial."""
    quantization_backend: str
    ecc_method: str
    ber: float
    trial: int

    quant_mse: float
    post_ecc_mse: float
    max_error: float

    bits_injected: int
    errors_corrected: int
    errors_detected: int
    uncorrected_errors: int

    encode_time_ms: float = 0.0
    decode_time_ms: float = 0.0


def run_quantization_ecc_benchmark(config=None):
    """Run comprehensive benchmark of quantization × ECC combinations."""
    import torch
    import time

    from hamming74 import Hamming74, Hamming84, Golay2412
    from hamming74.quantization_backends import (
        get_quantizer,
        QuantizationMode,
        QuantizationConfig,
    )
    from hamming74.triton_kernels import inject_bit_errors_triton

    if config is None:
        config = BenchmarkConfig()

    torch.manual_seed(config.seed)

    hamming74 = Hamming74(device="cuda")
    hamming84 = Hamming84(device="cuda", on_double_error="keep")
    golay = Golay2412(device="cuda")

    results = []

    keys = torch.randn(
        config.batch_size, config.num_heads, config.seq_length, config.head_dim,
        dtype=torch.float16, device="cuda"
    )
    values = torch.randn(
        config.batch_size, config.num_heads, config.seq_length, config.head_dim,
        dtype=torch.float16, device="cuda"
    )

    total_combos = (
        len(config.quantization_backends) *
        len(config.ecc_methods) *
        len(config.ber_levels) *
        config.num_trials
    )
    current = 0

    print("=" * 80)
    print("QUANTIZATION × ECC BENCHMARK")
    print("=" * 80)
    print(f"Tensor shape: [{config.batch_size}, {config.num_heads}, {config.seq_length}, {config.head_dim}]")
    print(f"Quantization backends: {config.quantization_backends}")
    print(f"ECC methods: {config.ecc_methods}")
    print(f"BER levels: {config.ber_levels}")
    print(f"Trials per combination: {config.num_trials}")
    print(f"Total trials: {total_combos}")
    print("=" * 80)

    for quant_backend in config.quantization_backends:
        qconfig = QuantizationConfig(block_size=32, group_size=32)
        quantizer = get_quantizer(quant_backend, qconfig)

        for ecc_method in config.ecc_methods:
            for ber in config.ber_levels:
                for trial in range(config.num_trials):
                    current += 1
                    trial_seed = config.seed + trial

                    mode = QuantizationMode.KEY if "kivi" in quant_backend else QuantizationMode.GENERIC
                    qt_keys = quantizer.quantize(keys, mode)
                    q_data = qt_keys.data.flatten()

                    keys_quant_only = quantizer.dequantize(qt_keys)
                    quant_mse = ((keys - keys_quant_only) ** 2).mean().item()

                    t0 = time.perf_counter()
                    if ecc_method == "none":
                        encoded = q_data.clone()
                        codeword_bits = 4
                    elif ecc_method == "hamming74":
                        encoded = hamming74.encode(q_data)
                        codeword_bits = 7
                    elif ecc_method == "hamming84":
                        encoded = hamming84.encode(q_data)
                        codeword_bits = 8
                    elif ecc_method == "golay":
                        remainder = q_data.numel() % 3
                        if remainder != 0:
                            pad_count = 3 - remainder
                            q_padded = torch.cat([q_data, torch.zeros(pad_count, dtype=q_data.dtype, device=q_data.device)])
                        else:
                            q_padded = q_data
                            pad_count = 0
                        triplets = q_padded.reshape(-1, 3)
                        encoded = golay.encode(triplets)
                        codeword_bits = 24
                    encode_time = (time.perf_counter() - t0) * 1000

                    if ber > 0:
                        if ecc_method == "golay":
                            encoded_flat = encoded.flatten()
                            corrupted, stats = inject_bit_errors_triton(
                                encoded_flat.to(torch.int32), ber, n_bits=24, seed=trial_seed, return_stats=True
                            )
                            injection_count = stats[0]
                            corrupted = corrupted.view(encoded.shape)
                        else:
                            corrupted, stats = inject_bit_errors_triton(
                                encoded.flatten(), ber, n_bits=codeword_bits, seed=trial_seed, return_stats=True
                            )
                            injection_count = stats[0]
                            corrupted = corrupted.view(encoded.shape)
                    else:
                        corrupted = encoded
                        injection_count = 0

                    t0 = time.perf_counter()
                    if ecc_method == "none":
                        decoded = corrupted.clone()
                        errors_corrected = 0
                        errors_detected = 0
                    elif ecc_method == "hamming74":
                        decoded, error_flags = hamming74.decode(corrupted.flatten())
                        errors_corrected = error_flags.sum().item()
                        errors_detected = 0
                    elif ecc_method == "hamming84":
                        result = hamming84.decode(corrupted.flatten())
                        decoded = result.data
                        errors_corrected = result.corrected_count
                        errors_detected = result.detected_count
                    elif ecc_method == "golay":
                        result = golay.decode(corrupted)
                        decoded = result.data.flatten()
                        if pad_count > 0:
                            decoded = decoded[:-pad_count]
                        errors_corrected = result.errors_corrected
                        errors_detected = 0
                    decode_time = (time.perf_counter() - t0) * 1000

                    decoded = decoded.view(qt_keys.data.shape)

                    qt_decoded = type(qt_keys)(
                        data=decoded,
                        scales=qt_keys.scales,
                        zero_points=qt_keys.zero_points,
                        mode=qt_keys.mode,
                        metadata=qt_keys.metadata,
                    )
                    keys_restored = quantizer.dequantize(qt_decoded)

                    post_ecc_mse = ((keys - keys_restored) ** 2).mean().item()
                    max_error = (keys - keys_restored).abs().max().item()

                    if ecc_method == "hamming84":
                        uncorrected = errors_detected
                    elif ecc_method == "golay":
                        uncorrected = max(0, injection_count - errors_corrected * 3)
                    else:
                        uncorrected = max(0, injection_count - errors_corrected)

                    result = TrialResult(
                        quantization_backend=quant_backend,
                        ecc_method=ecc_method,
                        ber=ber,
                        trial=trial,
                        quant_mse=quant_mse,
                        post_ecc_mse=post_ecc_mse,
                        max_error=max_error,
                        bits_injected=injection_count,
                        errors_corrected=errors_corrected,
                        errors_detected=errors_detected,
                        uncorrected_errors=uncorrected,
                        encode_time_ms=encode_time,
                        decode_time_ms=decode_time,
                    )
                    results.append(result)

                    if current % 10 == 0 or current == total_combos:
                        print(f"[{current}/{total_combos}] {quant_backend} + {ecc_method} @ BER={ber:.0e}: "
                              f"MSE={post_ecc_mse:.6f}, corrected={errors_corrected}")

    return results


def aggregate_results(results):
    """Aggregate results by (quantization_backend, ecc_method, ber)."""
    import statistics

    grouped = defaultdict(list)
    for r in results:
        key = (r.quantization_backend, r.ecc_method, r.ber)
        grouped[key].append(r)

    aggregated = {}
    for (quant, ecc, ber), trials in grouped.items():
        key = f"{quant}+{ecc}"
        if key not in aggregated:
            aggregated[key] = {}

        quant_mses = [t.quant_mse for t in trials]
        post_mses = [t.post_ecc_mse for t in trials]
        max_errors = [t.max_error for t in trials]

        aggregated[key][str(ber)] = {
            "quant_mse_mean": statistics.mean(quant_mses),
            "quant_mse_std": statistics.stdev(quant_mses) if len(quant_mses) > 1 else 0,
            "post_ecc_mse_mean": statistics.mean(post_mses),
            "post_ecc_mse_std": statistics.stdev(post_mses) if len(post_mses) > 1 else 0,
            "max_error_mean": statistics.mean(max_errors),
            "bits_injected": sum(t.bits_injected for t in trials),
            "errors_corrected": sum(t.errors_corrected for t in trials),
            "errors_detected": sum(t.errors_detected for t in trials),
            "num_trials": len(trials),
        }

    return aggregated


def format_comparison_table(results):
    """Format results as a comparison table."""
    aggregated = aggregate_results(results)

    ber_levels = sorted(set(r.ber for r in results))

    lines = []
    lines.append("=" * 100)
    lines.append("QUANTIZATION × ECC COMPARISON TABLE")
    lines.append("=" * 100)
    lines.append("")

    header = f"{'Backend + ECC':<35} |"
    for ber in ber_levels:
        header += f" BER={ber:.0e} |"
    lines.append(header)
    lines.append("-" * len(header))

    sorted_keys = sorted(aggregated.keys())

    for key in sorted_keys:
        ber_data = aggregated[key]
        row = f"{key:<35} |"
        for ber in ber_levels:
            data = ber_data.get(str(ber), {})
            mse = data.get("post_ecc_mse_mean", float("nan"))
            if mse > 10:
                row += f" {'FAIL':>12} |"
            else:
                row += f" {mse:>12.6f} |"
        lines.append(row)

    lines.append("")
    lines.append("=" * 100)
    lines.append("ERROR CORRECTION STATS (at max BER)")
    lines.append("=" * 100)
    lines.append("")

    max_ber = str(max(ber_levels))
    lines.append(f"{'Backend + ECC':<35} | {'Injected':>10} | {'Corrected':>10} | {'Detected':>10} | {'Rate':>8}")
    lines.append("-" * 85)

    for key in sorted_keys:
        ber_data = aggregated[key]
        data = ber_data.get(max_ber, {})
        injected = data.get("bits_injected", 0)
        corrected = data.get("errors_corrected", 0)
        detected = data.get("errors_detected", 0)
        rate = corrected / max(injected, 1) * 100
        lines.append(f"{key:<35} | {injected:>10} | {corrected:>10} | {detected:>10} | {rate:>7.1f}%")

    return "\n".join(lines)


def format_quantization_comparison(results):
    """Format quantization-only comparison (at BER=0)."""
    aggregated = aggregate_results(results)

    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("QUANTIZATION BACKEND COMPARISON (BER=0, no errors)")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"{'Backend':<20} | {'MSE':>12} | {'Max Error':>12}")
    lines.append("-" * 50)

    backend_mse = {}
    for key in aggregated:
        if "+none" in key:
            backend = key.replace("+none", "")
            data = aggregated[key].get("0.0", aggregated[key].get("0", {}))
            if data:
                backend_mse[backend] = (
                    data.get("quant_mse_mean", 0),
                    data.get("max_error_mean", 0),
                )

    for backend in sorted(backend_mse.keys()):
        mse, max_err = backend_mse[backend]
        lines.append(f"{backend:<20} | {mse:>12.6f} | {max_err:>12.4f}")

    return "\n".join(lines)


def run_benchmark_impl():
    """Entry point for Modal execution."""
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = BenchmarkConfig(
        batch_size=2,
        num_heads=8,
        seq_length=64,
        head_dim=32,
        ber_levels=[0.0, 1e-4, 1e-3, 1e-2],
        num_trials=3,
    )

    results = run_quantization_ecc_benchmark(config)

    comparison_table = format_comparison_table(results)
    quant_table = format_quantization_comparison(results)

    print("\n" + comparison_table)
    print("\n" + quant_table)

    aggregated = aggregate_results(results)

    return {
        "config": {
            "batch_size": config.batch_size,
            "num_heads": config.num_heads,
            "seq_length": config.seq_length,
            "head_dim": config.head_dim,
            "ber_levels": config.ber_levels,
            "num_trials": config.num_trials,
        },
        "aggregated": aggregated,
        "comparison_table": comparison_table,
        "quantization_table": quant_table,
    }


if __name__ == "__main__":
    run_benchmark_impl()
