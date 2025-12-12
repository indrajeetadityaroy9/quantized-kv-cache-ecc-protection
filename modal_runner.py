"""
Modal GPU Runner for Hamming(7,4) INT4 KV Cache Protection.

This is the Modal app entrypoint. Core logic lives in evaluation/runners/modal.py.

GPU acceleration benefits:
- LLaMA-3.1-8B tests: requires A100-80GB (~16GB VRAM in FP16)
- Verification tests with GPT-2: ~5x faster on T4
- Core Hamming tests: minimal benefit (CPU-bound operations)

Usage:
    # Quick verification on GPU
    modal run modal_runner.py --verify

    # Run all tests on GPU
    modal run modal_runner.py

    # Run specific test file
    modal run modal_runner.py --test-file tests/test_llama_integration.py

    # Unified benchmark command (replaces --sweep, --llama-demo, --architecture, --generation-demo)
    modal run modal_runner.py --benchmark --models gpt2           # GPT-2 sweep
    modal run modal_runner.py --benchmark --models llama          # LLaMA sweep
    modal run modal_runner.py --benchmark --models both           # Architecture comparison
    modal run modal_runner.py --benchmark --models llama --include-generation  # With generation demo
    modal run modal_runner.py --benchmark --models both --seeds 5 --output ./results

    # Latency benchmarks (Phase 1 - CPU-bound baseline)
    modal run modal_runner.py --latency                           # Isolated codec latency benchmarks
    modal run modal_runner.py --latency --latency-iterations 200  # More iterations for precision

    # Triton ECC Evaluation (Phase 4.3 - PARALLEL on A100s)
    modal run modal_runner.py --eval-triton                       # Run Triton PPL sweep on LLaMA-8B
    modal run modal_runner.py --eval-triton --triton-model meta-llama/Llama-3.1-8B --triton-seeds 3
    modal run modal_runner.py --eval-triton --triton-samples 100  # More samples per trial

    # Attention Kernel Benchmarks (Phase 4.1)
    modal run modal_runner.py --benchmark-kernels                 # Measure Triton kernel latency vs baseline

    # List and pull results via Modal SDK
    modal run modal_runner.py --list-results-flag         # List all benchmark results
    modal run modal_runner.py --pull-results <run_name>   # Pull specific run to ./results/
    modal run modal_runner.py --pull-latest               # Pull most recent results

Results are automatically saved to Modal volume 'hamming74-results'.
Alternative CLI commands:
    modal volume ls hamming74-results                    # List available results
    modal volume get hamming74-results <folder> ./results/  # Download specific run
"""

import modal

# Create Modal app
app = modal.App("hamming74-experiments")

# HuggingFace authentication secret (required for LLaMA access)
# Create via: modal secret create huggingface-secret HF_TOKEN=<token>
hf_secret = modal.Secret.from_name("huggingface-secret")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pytest>=7.0.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "accelerate",
        "huggingface-hub",
        "matplotlib",
    )
    .add_local_dir(".", remote_path="/app")
)

# Volume for caching model weights
model_cache = modal.Volume.from_name("hamming74-model-cache", create_if_missing=True)

# Volume for storing benchmark outputs (can be pulled locally)
output_volume = modal.Volume.from_name("hamming74-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/cache": model_cache},
)
def run_tests(test_file: str = None, verbose: bool = True):
    """Run pytest on GPU."""
    import os
    import sys
    os.chdir("/app")
    sys.path.insert(0, "/app")
    from evaluation.runners.modal import run_tests_impl
    return run_tests_impl(test_file=test_file, verbose=verbose)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/cache": model_cache},
    secrets=[hf_secret],
)
def run_llama_tests(test_file: str = None, verbose: bool = True):
    """Run LLaMA tests on A100 GPU with HF_TOKEN."""
    import os
    import sys
    os.chdir("/app")
    sys.path.insert(0, "/app")
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"
    from evaluation.runners.modal import run_tests_impl
    return run_tests_impl(test_file=test_file, verbose=verbose)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={"/cache": model_cache},
)
def quick_verify():
    """Quick verification that everything works on GPU."""
    import os
    import sys
    os.chdir("/app")
    sys.path.insert(0, "/app")
    from evaluation.runners.modal import quick_verify_impl
    return quick_verify_impl()


@app.function(
    image=image,
    volumes={"/output": output_volume},
    timeout=60,
)
def list_results() -> list:
    """List all benchmark results stored in the Modal volume."""
    import os

    results = []
    output_dir = "/output"

    if not os.path.exists(output_dir):
        return results

    for entry in sorted(os.listdir(output_dir), reverse=True):
        entry_path = os.path.join(output_dir, entry)
        if os.path.isdir(entry_path):
            # Check for results.json to confirm it's a valid benchmark run
            results_file = os.path.join(entry_path, "results.json")
            summary_file = os.path.join(entry_path, "summary.txt")
            metrics_file = os.path.join(entry_path, "metrics.txt")

            files = []
            if os.path.exists(results_file):
                files.append("results.json")
            if os.path.exists(summary_file):
                files.append("summary.txt")
            if os.path.exists(metrics_file):
                files.append("metrics.txt")

            if files:
                # Get modification time
                mtime = os.path.getmtime(entry_path)
                results.append({
                    "name": entry,
                    "files": files,
                    "timestamp": mtime,
                })

    return results


@app.function(
    image=image,
    volumes={"/output": output_volume},
    timeout=120,
)
def get_results(run_name: str) -> dict:
    """
    Retrieve benchmark results from the Modal volume.

    Args:
        run_name: Name of the benchmark run (e.g., 'llama_benchmark_20251209_134313')

    Returns:
        Dictionary containing:
        - results_json: The full JSON results (if available)
        - summary_txt: Human-readable summary (if available)
        - metrics_txt: Formatted metrics table (if available)
    """
    import os
    import json

    output_dir = f"/output/{run_name}"

    if not os.path.exists(output_dir):
        return {"error": f"Run '{run_name}' not found"}

    result = {"name": run_name, "files": {}}

    # Read results.json
    results_file = os.path.join(output_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            result["files"]["results.json"] = json.load(f)

    # Read summary.txt
    summary_file = os.path.join(output_dir, "summary.txt")
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            result["files"]["summary.txt"] = f.read()

    # Read metrics.txt
    metrics_file = os.path.join(output_dir, "metrics.txt")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            result["files"]["metrics.txt"] = f.read()

    return result


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/cache": model_cache, "/output": output_volume},
)
def run_latency_benchmark(
    n_iterations: int = 100,
    save_to_volume: bool = True,
):
    """
    Run isolated codec latency benchmarks.

    NOTE: Phase 1 metrics are CPU-bound baseline. Since Hamming codecs
    currently run on CPU, throughput numbers are bottlenecked by CPU-GPU
    transfers. Phase 3 (Triton) will provide GPU-native metrics.

    Args:
        n_iterations: Number of iterations per benchmark
        save_to_volume: Save results to Modal volume
    """
    import os
    import sys
    import json
    from datetime import datetime

    os.chdir("/app")
    sys.path.insert(0, "/app")

    from evaluation.experiments.latency import (
        CodecBenchmarkConfig,
        run_codec_benchmarks,
    )

    print("=" * 70)
    print("CODEC LATENCY BENCHMARKS (CPU-BOUND BASELINE)")
    print("=" * 70)
    print("\nNOTE: Phase 1 metrics are CPU-bound baseline.")
    print("      Phase 3 (Triton) will provide GPU-native metrics.")
    print("=" * 70)

    # Configure benchmarks with common LLM tensor sizes
    config = CodecBenchmarkConfig(
        tensor_sizes=[
            (1, 256, 768),      # GPT-2: single batch, short seq
            (8, 256, 768),      # GPT-2: 8 batch
            (1, 1024, 768),     # GPT-2: long seq
            (1, 256, 4096),     # LLaMA: single batch
            (8, 256, 4096),     # LLaMA: 8 batch
            (1, 1024, 4096),    # LLaMA: long seq
            (32, 512, 4096),    # LLaMA: high batch, medium seq
        ],
        n_iterations=n_iterations,
        codecs=["int4", "int4-hamming", "int4-hamming84", "int12-golay"],
    )

    def progress(msg, curr, total):
        print(f"  [{curr+1}/{total}] {msg}")

    report = run_codec_benchmarks(config, progress)

    print("\n" + report.get_summary_table())

    # Save results to volume
    if save_to_volume:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/output/latency_benchmark_{timestamp}"
        os.makedirs(output_path, exist_ok=True)

        # Save JSON results
        json_file = f"{output_path}/results.json"
        with open(json_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nResults saved to volume: {json_file}")

        # Save summary
        summary_file = f"{output_path}/summary.txt"
        with open(summary_file, "w") as f:
            f.write(report.get_summary_table())
        print(f"Summary saved to volume: {summary_file}")

        output_volume.commit()
        print(f"\nResults committed to 'hamming74-results' volume.")
        print(f"To download: modal volume get hamming74-results latency_benchmark_{timestamp} ./results/")

    return report.to_dict()


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,  # 1 hour per trial (plenty of margin)
    volumes={"/cache": model_cache, "/output": output_volume},
    secrets=[hf_secret],
)
def run_triton_worker(
    model_name: str,
    mode: str,
    ber: float,
    seed: int,
    max_samples: int,
) -> dict:
    """
    Single-trial worker for parallel Triton PPL evaluation.

    Designed for parallel execution via .starmap().
    Each invocation runs ONE trial (mode × BER × seed combination).
    """
    import os
    import sys
    os.chdir("/app")
    sys.path.insert(0, "/app")
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"

    from evaluation.runners.triton_eval import run_single_triton_trial
    return run_single_triton_trial(model_name, mode, ber, seed, max_samples)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/cache": model_cache, "/output": output_volume},
)
def run_kernel_benchmarks(
    batch_sizes: list = None,
    seq_lens: list = None,
    num_heads: int = 32,
    head_dim: int = 128,
    warmup: int = 10,
    repeat: int = 50,
    save_to_volume: bool = True,
) -> dict:
    """
    Run attention kernel latency benchmarks on A100.

    Measures overhead of ECC-protected attention vs PyTorch SDPA baseline.
    """
    import os
    import sys
    import json
    from datetime import datetime

    os.chdir("/app")
    sys.path.insert(0, "/app")

    from vllm_kernels.benchmark_harness import run_attention_benchmark_suite

    if batch_sizes is None:
        batch_sizes = [1, 4, 16]
    if seq_lens is None:
        seq_lens = [128, 512, 2048]

    print("=" * 70)
    print("ATTENTION KERNEL LATENCY BENCHMARKS")
    print("=" * 70)
    print(f"Batch sizes: {batch_sizes}")
    print(f"Sequence lengths: {seq_lens}")
    print(f"num_heads={num_heads}, head_dim={head_dim}")
    print(f"warmup={warmup}, repeat={repeat}")
    print("=" * 70)

    results = run_attention_benchmark_suite(
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_dim=head_dim,
        warmup=warmup,
        repeat=repeat,
    )

    # Convert results to dict for serialization
    results_dict = [r.__dict__ if hasattr(r, '__dict__') else r for r in results]

    # Save to volume
    if save_to_volume:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/output/kernel_benchmark_{timestamp}"
        os.makedirs(output_path, exist_ok=True)

        # Save JSON
        with open(f"{output_path}/results.json", "w") as f:
            json.dump({"attention_benchmarks": results_dict}, f, indent=2)

        # Save formatted table
        with open(f"{output_path}/latency_table.md", "w") as f:
            f.write("# Attention Kernel Latency Results\n\n")
            f.write("| Configuration | Latency (us) | Overhead vs Baseline |\n")
            f.write("|---------------|--------------|---------------------|\n")
            for r in results_dict:
                overhead = f"{r.get('overhead_vs_baseline', 'N/A'):.2f}x" if r.get('overhead_vs_baseline') else "N/A"
                f.write(f"| {r['name']} B={r['batch_size']} S={r['seq_len']} | {r['latency_us']:.2f} | {overhead} |\n")

        output_volume.commit()
        print(f"\nResults saved to: {output_path}")

    return {"attention_benchmarks": results_dict}


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=21600,  # 6 hours for full 150-config sweep with protected modes
    volumes={"/cache": model_cache, "/output": output_volume},
    secrets=[hf_secret],
)
def run_benchmark(
    models: str = "gpt2",
    seeds: int = 3,
    include_generation: bool = False,
    generation_ber: float = 0.05,
    output_dir: str = None,
    save_to_volume: bool = True,
):
    """
    Unified benchmark runner.

    Replaces: --sweep, --llama-demo, --architecture, --generation-demo

    Args:
        models: "gpt2", "llama", or "both"
        seeds: Number of Monte Carlo seeds
        include_generation: Run text generation demo (LLaMA only)
        generation_ber: BER for generation demo
        output_dir: Output directory for results
        save_to_volume: Save results to Modal volume for later retrieval
    """
    import os
    import sys
    import json
    from datetime import datetime

    os.chdir("/app")
    sys.path.insert(0, "/app")
    from evaluation.runners.modal import run_benchmark_impl

    results = run_benchmark_impl(
        models=models,
        seeds=seeds,
        include_generation=include_generation,
        generation_ber=generation_ber,
        output_dir=output_dir,
    )

    # Save results to Modal volume for retrieval
    if save_to_volume:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/output/{models}_benchmark_{timestamp}"
        os.makedirs(output_path, exist_ok=True)

        # Save JSON results
        json_file = f"{output_path}/results.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to volume: {json_file}")

        # Save human-readable summary
        summary_file = f"{output_path}/summary.txt"
        with open(summary_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(f"BENCHMARK RESULTS: {models.upper()}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Seeds: {seeds}\n")
            f.write(f"Include Generation: {include_generation}\n")
            f.write("=" * 70 + "\n\n")

            if results.get("sweep_results"):
                for model_key, sweep_data in results["sweep_results"].items():
                    f.write(f"\n{model_key.upper()} SWEEP RESULTS:\n")
                    f.write("-" * 50 + "\n")
                    for mode, ber_results in sweep_data.items():
                        f.write(f"\n  {mode}:\n")
                        for ber, metrics in ber_results.items():
                            ppl = metrics.get("ppl_mean", "N/A")
                            ppl_std = metrics.get("ppl_std", 0)
                            kl = metrics.get("kl_divergence_mean", "N/A")
                            f.write(f"    BER={ber}: PPL={ppl:.2f}±{ppl_std:.2f}, KL={kl:.4f}\n")

            if results.get("generation_results"):
                f.write("\n\nGENERATION DEMO RESULTS:\n")
                f.write("-" * 50 + "\n")
                for gen in results["generation_results"]:
                    f.write(f"\nPrompt: {gen.get('prompt', 'N/A')}\n")
                    f.write(f"Mode: {gen.get('mode', 'N/A')}\n")
                    f.write(f"Output: {gen.get('output', 'N/A')}\n")

        print(f"Summary saved to volume: {summary_file}")

        # Save formatted metrics table
        metrics_file = f"{output_path}/metrics.txt"
        with open(metrics_file, "w") as f:
            f.write("=" * 100 + "\n")
            f.write(f"BENCHMARK METRICS: {models.upper()} | {timestamp}\n")
            f.write("=" * 100 + "\n\n")

            if results.get("sweep_results"):
                for model_key, sweep_data in results["sweep_results"].items():
                    f.write(f"\n{'=' * 80}\n")
                    f.write(f"{model_key.upper()} PERPLEXITY TABLE\n")
                    f.write(f"{'=' * 80}\n\n")

                    # Get all BER levels
                    ber_levels = []
                    for mode_data in sweep_data.values():
                        ber_levels = list(mode_data.keys())
                        break

                    # Header
                    header = f"{'Cache Mode':<25}"
                    for ber in ber_levels:
                        header += f" | BER={ber:>8}"
                    f.write(header + "\n")
                    f.write("-" * len(header) + "\n")

                    # Data rows
                    for mode, ber_results in sweep_data.items():
                        row = f"{mode:<25}"
                        for ber in ber_levels:
                            metrics = ber_results.get(ber, {})
                            ppl = metrics.get("ppl_mean", 0)
                            ppl_std = metrics.get("ppl_std", 0)
                            row += f" | {ppl:>6.2f}±{ppl_std:<4.2f}"
                        f.write(row + "\n")

                    # KL Divergence table
                    f.write(f"\n\n{'=' * 80}\n")
                    f.write(f"{model_key.upper()} KL DIVERGENCE TABLE (nats)\n")
                    f.write(f"{'=' * 80}\n\n")

                    f.write(header + "\n")
                    f.write("-" * len(header) + "\n")

                    for mode, ber_results in sweep_data.items():
                        row = f"{mode:<25}"
                        for ber in ber_levels:
                            metrics = ber_results.get(ber, {})
                            kl = metrics.get("kl_divergence_mean", 0)
                            row += f" | {kl:>12.4f}"
                        f.write(row + "\n")

                    # Top-5 Accuracy table
                    f.write(f"\n\n{'=' * 80}\n")
                    f.write(f"{model_key.upper()} TOP-5 ACCURACY TABLE (%)\n")
                    f.write(f"{'=' * 80}\n\n")

                    f.write(header + "\n")
                    f.write("-" * len(header) + "\n")

                    for mode, ber_results in sweep_data.items():
                        row = f"{mode:<25}"
                        for ber in ber_levels:
                            metrics = ber_results.get(ber, {})
                            top5 = metrics.get("top5_accuracy_mean", 0) * 100
                            row += f" | {top5:>12.1f}"
                        f.write(row + "\n")

                    # Error correction stats
                    f.write(f"\n\n{'=' * 80}\n")
                    f.write(f"{model_key.upper()} ERROR CORRECTION STATS (at max BER)\n")
                    f.write(f"{'=' * 80}\n\n")

                    max_ber = ber_levels[-1] if ber_levels else "0"
                    f.write(f"{'Cache Mode':<25} | {'Errors Corrected':>18} | {'Errors Detected':>16}\n")
                    f.write("-" * 65 + "\n")

                    for mode, ber_results in sweep_data.items():
                        metrics = ber_results.get(max_ber, {})
                        corrected = metrics.get("errors_corrected", 0)
                        detected = metrics.get("errors_detected", 0)
                        f.write(f"{mode:<25} | {corrected:>18,.0f} | {detected:>16,.0f}\n")

        print(f"Metrics saved to volume: {metrics_file}")

        # Commit the volume to persist changes
        output_volume.commit()
        print(f"\nResults committed to 'hamming74-results' volume.")
        print(f"To download: modal volume get hamming74-results {models}_benchmark_{timestamp} ./results/")

    return results


def save_triton_results(results: list, output_path: str):
    """Save and format Triton PPL results."""
    import json
    import os
    from evaluation.runners.triton_eval import format_ppl_table, aggregate_results

    os.makedirs(output_path, exist_ok=True)

    # Raw JSON
    with open(f"{output_path}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Aggregated results
    aggregated = aggregate_results(results)
    with open(f"{output_path}/aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    # Formatted PPL table
    ppl_table = format_ppl_table(results)
    with open(f"{output_path}/ppl_table.md", "w") as f:
        f.write("# Triton ECC Perplexity Results\n\n")
        f.write(ppl_table)

    # Summary text
    with open(f"{output_path}/summary.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("TRITON ECC PERPLEXITY SWEEP RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total trials: {len(results)}\n")
        modes = set(r["mode"] for r in results)
        bers = sorted(set(r["ber"] for r in results))
        seeds = set(r["seed"] for r in results)

        f.write(f"Modes: {', '.join(modes)}\n")
        f.write(f"BER levels: {[f'{b:.0e}' for b in bers]}\n")
        f.write(f"Seeds: {seeds}\n\n")

        f.write("PPL Table:\n")
        f.write(ppl_table)
        f.write("\n\n")

        # Per-mode summary
        for mode, mode_data in aggregated.items():
            f.write(f"\n{mode.upper()}:\n")
            for ber_str, stats in mode_data.items():
                ppl_mean = stats["ppl_mean"]
                ppl_std = stats["ppl_std"]
                if ppl_mean > 1000:
                    f.write(f"  BER={ber_str}: PPL > 1000 (catastrophic)\n")
                else:
                    f.write(f"  BER={ber_str}: PPL = {ppl_mean:.2f} +/- {ppl_std:.2f}\n")


@app.local_entrypoint()
def main(
    # Test runner
    test_file: str = None,
    verify: bool = False,
    llama_tests: bool = False,

    # Unified benchmark
    benchmark: bool = False,
    models: str = "gpt2",
    seeds: int = 3,
    include_generation: bool = False,
    ber: float = 0.05,
    output: str = None,

    # Latency benchmark (Phase 1 - CPU-bound baseline)
    latency: bool = False,
    latency_iterations: int = 100,

    # Triton ECC evaluation (Phase 4.3)
    eval_triton: bool = False,
    triton_model: str = "meta-llama/Llama-3.1-8B",
    triton_seeds: int = 3,
    triton_samples: int = 50,

    # Kernel latency benchmarks
    benchmark_kernels: bool = False,

    # Results management
    list_results_flag: bool = False,
    pull_results: str = None,
    pull_latest: bool = False,
    results_dir: str = "./results",
):
    """
    Modal experiment runner entrypoint.

    Args:
        test_file: Specific test file to run (e.g., tests/test_verification.py)
        verify: Quick verification test
        llama_tests: Run LLaMA integration tests on A100 with HF_TOKEN

        benchmark: Run unified benchmark (replaces --sweep, --llama-demo, etc.)
        models: Model selection for benchmark: "gpt2", "llama", or "both"
        seeds: Number of Monte Carlo seeds for benchmark
        include_generation: Include text generation demo (LLaMA only)
        ber: Bit error rate for generation demo
        output: Output directory for results

        latency: Run isolated codec latency benchmarks (Phase 1 - CPU-bound baseline)
        latency_iterations: Number of iterations for latency benchmarks

        eval_triton: Run Triton ECC PPL sweep (Phase 4.3 - parallel on A100s)
        triton_model: Model for Triton eval (default: meta-llama/Llama-3.1-8B)
        triton_seeds: Number of Monte Carlo seeds for Triton sweep
        triton_samples: Number of WikiText-2 samples per trial

        benchmark_kernels: Run attention kernel latency benchmarks

        list_results_flag: List all available benchmark results
        pull_results: Pull specific benchmark run by name
        pull_latest: Pull the most recent benchmark results
        results_dir: Local directory to save pulled results (default: ./results)
    """
    import os
    import json
    from pathlib import Path
    from datetime import datetime

    # Handle results listing
    if list_results_flag:
        print("\n" + "=" * 70)
        print("AVAILABLE BENCHMARK RESULTS")
        print("=" * 70)

        results_list = list_results.remote()

        if not results_list:
            print("\nNo benchmark results found in Modal volume.")
            print("Run a benchmark first: modal run modal_runner.py --benchmark --models llama")
            return

        print(f"\n{'Run Name':<45} | {'Files':<30} | Timestamp")
        print("-" * 95)

        for r in results_list:
            ts = datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            files = ", ".join(r["files"])
            print(f"{r['name']:<45} | {files:<30} | {ts}")

        print(f"\nTotal: {len(results_list)} benchmark run(s)")
        print("\nTo pull results: modal run modal_runner.py --pull-results <run_name>")
        print("To pull latest:  modal run modal_runner.py --pull-latest")
        return

    # Handle pulling specific results
    if pull_results or pull_latest:
        if pull_latest:
            print("\nFetching latest results...")
            results_list = list_results.remote()
            if not results_list:
                print("No benchmark results found.")
                return
            pull_results = results_list[0]["name"]  # Already sorted by timestamp desc
            print(f"Latest run: {pull_results}")

        print(f"\nPulling results: {pull_results}")
        result_data = get_results.remote(pull_results)

        if "error" in result_data:
            print(f"Error: {result_data['error']}")
            return

        # Create output directory
        output_path = Path(results_dir) / pull_results
        output_path.mkdir(parents=True, exist_ok=True)

        # Save files locally
        files_saved = []
        for filename, content in result_data.get("files", {}).items():
            file_path = output_path / filename
            if filename.endswith(".json"):
                with open(file_path, "w") as f:
                    json.dump(content, f, indent=2)
            else:
                with open(file_path, "w") as f:
                    f.write(content)
            files_saved.append(filename)

        print(f"\nResults saved to: {output_path}")
        print(f"Files: {', '.join(files_saved)}")

        # Print summary if available
        if "summary.txt" in result_data.get("files", {}):
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(result_data["files"]["summary.txt"])

        return

    if llama_tests:
        # Run LLaMA tests on A100 with HF_TOKEN
        test_path = test_file if test_file else "tests/test_llama_integration.py"
        return_code = run_llama_tests.remote(test_file=test_path)
        print(f"\nLLaMA tests completed with return code: {return_code}")

    elif verify:
        result = quick_verify.remote()
        print(f"\nVerification: {'PASSED' if result else 'FAILED'}")

    elif latency:
        # Run isolated codec latency benchmarks
        print("\n" + "=" * 70)
        print("CODEC LATENCY BENCHMARKS (CPU-BOUND BASELINE)")
        print("=" * 70)
        print("\nNOTE: Phase 1 metrics are CPU-bound baseline.")
        print("      Phase 3 (Triton) will provide GPU-native metrics.")

        results = run_latency_benchmark.remote(
            n_iterations=latency_iterations,
        )

        print("\n" + "=" * 70)
        print("LATENCY BENCHMARK COMPLETE")
        print("=" * 70)
        print(f"\nBenchmarked {len(results.get('results', []))} configurations")

    elif benchmark_kernels:
        # Run attention kernel latency benchmarks (Triton)
        print("\n" + "=" * 70)
        print("ATTENTION KERNEL LATENCY BENCHMARKS (TRITON)")
        print("=" * 70)

        results = run_kernel_benchmarks.remote()

        print("\n" + "=" * 70)
        print("KERNEL BENCHMARK COMPLETE")
        print("=" * 70)

        if results.get("attention_benchmarks"):
            benchmarks = results["attention_benchmarks"]
            print(f"\nBenchmarked {len(benchmarks)} configurations")

            # Print summary table
            print("\n| Configuration | Latency (us) | Overhead |")
            print("|---------------|--------------|----------|")
            for r in benchmarks[:10]:  # Show first 10
                overhead = f"{r.get('overhead_vs_baseline', 0):.2f}x" if r.get('overhead_vs_baseline') else "N/A"
                print(f"| {r['name'][:25]} | {r['latency_us']:.2f} | {overhead} |")

    elif eval_triton:
        # Run Triton ECC PPL sweep with PARALLEL execution
        print("\n" + "=" * 70)
        print("TRITON ECC PERPLEXITY SWEEP (PARALLEL EXECUTION)")
        print("=" * 70)

        # Define sweep matrix - ALL 7 protection modes
        # fp16: No quantization (oracle baseline)
        # int4: INT4 quantization, no ECC protection
        # int4-hamming: Hamming(7,4) SEC - corrects 1 bit
        # int4-hamming84: Hamming(8,4) SECDED - corrects 1, detects 2
        # int4-hamming84-interp: SECDED + interpolation for detected double errors
        # int12-golay: Golay(24,12) - corrects up to 3 bits per triplet
        # adaptive: Golay for sink tokens, Hamming84 for context (UEP)
        # Define sweep modes - can be overridden via command line in future
        all_modes = [
            "fp16",
            "int4",
            "int4-hamming",
            "int4-hamming84",
            "int4-hamming84-interp",
            "int12-golay",
            "adaptive",
        ]
        # Use all modes for full sweep
        modes = all_modes
        bers = [0.0, 1e-4, 1e-3, 1e-2]
        seed_list = [42, 101, 997][:triton_seeds]

        # Build argument list for all trials
        args_list = [
            (triton_model, mode, ber, seed, triton_samples)
            for mode in modes
            for ber in bers
            for seed in seed_list
        ]

        print(f"\nModel: {triton_model}")
        print(f"Modes: {modes}")
        print(f"BER levels: {bers}")
        print(f"Seeds: {seed_list}")
        print(f"Samples per trial: {triton_samples}")
        print(f"\nTotal trials: {len(args_list)}")
        print("=" * 70)
        print(f"Spawning {len(args_list)} parallel A100 jobs...")
        print("=" * 70)

        # PARALLEL EXECUTION using .starmap()
        results = []
        for res in run_triton_worker.starmap(args_list):
            ppl_str = f"{res['ppl']:.2f}" if res['ppl'] < 1000 else ">1000"
            print(f"  [DONE] {res['mode']} @ BER={res['ber']:.0e} seed={res['seed']}: PPL={ppl_str}")
            results.append(res)

        print("\n" + "=" * 70)
        print("TRITON PPL SWEEP COMPLETE")
        print("=" * 70)

        # Save results locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(results_dir) / f"triton_ppl_{timestamp}"
        save_triton_results(results, str(output_path))

        print(f"\nResults saved to: {output_path}")

        # Print summary table
        from evaluation.runners.triton_eval import format_ppl_table
        print("\nPPL Summary:")
        print(format_ppl_table(results))

    elif benchmark:
        results = run_benchmark.remote(
            models=models,
            seeds=seeds,
            include_generation=include_generation,
            generation_ber=ber,
            output_dir=output,
        )

        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)

        # Print summary
        if results.get("sweep_results"):
            for model_key, sweep_data in results["sweep_results"].items():
                print(f"\n{model_key.upper()} Results:")
                for mode in list(sweep_data.keys())[:3]:  # Show first 3 modes
                    mode_data = sweep_data[mode]
                    # Get BER=0 result as baseline
                    if "0" in mode_data:
                        ppl = mode_data["0"]["ppl_mean"]
                        print(f"  {mode}: PPL={ppl:.2f} (BER=0)")

        if results.get("architecture_comparison"):
            arch = results["architecture_comparison"]
            print(f"\nArchitecture Comparison:")
            print(f"  GPT-2: {arch['gpt2_info'].get('n_kv_projections', 'N/A')} KV projections")
            print(f"  LLaMA: {arch['llama_info'].get('n_kv_projections', 'N/A')} KV projections")

        if results.get("generation_results"):
            print(f"\nGeneration Demo: {len(results['generation_results'])} samples generated")

    else:
        # Run tests
        return_code = run_tests.remote(test_file=test_file)
        print(f"\nTests completed with return code: {return_code}")
