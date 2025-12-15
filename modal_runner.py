import os
import sys
import json
from datetime import datetime
from pathlib import Path

import modal

app = modal.App("hamming74-experiments")
hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "triton>=2.1.0",
        "numpy>=1.21.0",
        "pytest>=7.0.0",
        "transformers>=4.40.0,<4.45.0",
        "datasets>=2.14.0",
        "accelerate",
        "huggingface-hub",
        "matplotlib",
    )
    .add_local_dir(".", remote_path="/app")
)

model_cache = modal.Volume.from_name("hamming74-model-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("hamming74-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/cache": model_cache},
)
def run_tests(test_file=None, verbose=True):
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
def run_llama_tests(test_file=None, verbose=True):
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
    os.chdir("/app")
    sys.path.insert(0, "/app")
    from evaluation.runners.modal import quick_verify_impl
    return quick_verify_impl()


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={"/cache": model_cache},
)
def verify_quantization():
    """Verify all quantization backends work correctly."""
    os.chdir("/app")
    sys.path.insert(0, "/app")
    from evaluation.runners.modal import verify_quantization_backends_impl
    return verify_quantization_backends_impl()


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    volumes={"/cache": model_cache},
)
def run_quantization_ecc_comparison():
    """Run comprehensive quantization × ECC comparison benchmark."""
    os.chdir("/app")
    sys.path.insert(0, "/app")
    from evaluation.experiments.quantization_ecc_comparison import run_benchmark_impl
    return run_benchmark_impl()


@app.function(
    image=image,
    volumes={"/output": output_volume},
    timeout=60,
)
def list_results():
    results = []
    output_dir = "/output"

    if not os.path.exists(output_dir):
        return results

    for entry in sorted(os.listdir(output_dir), reverse=True):
        entry_path = os.path.join(output_dir, entry)
        if os.path.isdir(entry_path):
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
                mtime = os.path.getmtime(entry_path)
                results.append({"name": entry, "files": files, "timestamp": mtime})
    return results


@app.function(
    image=image,
    volumes={"/output": output_volume},
    timeout=120,
)
def get_results(run_name):
    output_dir = f"/output/{run_name}"

    if not os.path.exists(output_dir):
        return {"error": f"Run '{run_name}' not found"}

    result = {"name": run_name, "files": {}}

    results_file = os.path.join(output_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            result["files"]["results.json"] = json.load(f)

    summary_file = os.path.join(output_dir, "summary.txt")
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            result["files"]["summary.txt"] = f.read()

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
def run_latency_benchmark(n_iterations=100, save_to_volume=True):
    os.chdir("/app")
    sys.path.insert(0, "/app")
    from evaluation.experiments.latency import CodecBenchmarkConfig, run_codec_benchmarks

    config = CodecBenchmarkConfig(
        tensor_sizes=[
            (1, 256, 768),
            (8, 256, 768),
            (1, 1024, 768),
            (1, 256, 4096),
            (8, 256, 4096),
            (1, 1024, 4096),
            (32, 512, 4096),
        ],
        n_iterations=n_iterations,
        codecs=["int4", "int4-hamming", "int4-hamming84", "int12-golay"],
    )

    def progress(msg, curr, total):
        print(f"[{curr+1}/{total}] {msg}")

    report = run_codec_benchmarks(config, progress)

    print(report.get_summary_table())

    if save_to_volume:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/output/latency_benchmark_{timestamp}"
        os.makedirs(output_path, exist_ok=True)

        json_file = f"{output_path}/results.json"
        with open(json_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nResults saved to volume: {json_file}")

        summary_file = f"{output_path}/summary.txt"
        with open(summary_file, "w") as f:
            f.write(report.get_summary_table())
        print(f"Summary saved to volume: {summary_file}")

        output_volume.commit()
        print(f"To download: modal volume get hamming74-results latency_benchmark_{timestamp} ./results/")

    return report.to_dict()


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/cache": model_cache, "/output": output_volume},
    secrets=[hf_secret],
)
def run_triton_worker(model_name, mode, ber, seed, max_samples):
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
    batch_sizes=None,
    seq_lens=None,
    num_heads=32,
    head_dim=128,
    warmup=10,
    repeat=50,
    save_to_volume=True,
):
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

    results_dict = [r.__dict__ if hasattr(r, "__dict__") else r for r in results]

    if save_to_volume:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/output/kernel_benchmark_{timestamp}"
        os.makedirs(output_path, exist_ok=True)

        with open(f"{output_path}/results.json", "w") as f:
            json.dump({"attention_benchmarks": results_dict}, f, indent=2)

        with open(f"{output_path}/latency_table.md", "w") as f:
            f.write("# Attention Kernel Latency Results\n\n")
            f.write("| Configuration | Latency (us) | Overhead vs Baseline |\n")
            f.write("|---------------|--------------|---------------------|\n")
            for r in results_dict:
                overhead = f"{r.get('overhead_vs_baseline', 'N/A'):.2f}x" if r.get("overhead_vs_baseline") else "N/A"
                f.write(f"| {r['name']} B={r['batch_size']} S={r['seq_len']} | {r['latency_us']:.2f} | {overhead} |\n")

        output_volume.commit()
        print(f"\nResults saved to: {output_path}")

    return {"attention_benchmarks": results_dict}


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=21600,
    volumes={"/cache": model_cache, "/output": output_volume},
    secrets=[hf_secret],
)
def run_benchmark(
    models="gpt2",
    seeds=3,
    include_generation=False,
    generation_ber=0.05,
    output_dir=None,
    save_to_volume=True,
):
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

    if save_to_volume:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/output/{models}_benchmark_{timestamp}"
        os.makedirs(output_path, exist_ok=True)

        json_file = f"{output_path}/results.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to volume: {json_file}")

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
                            f.write(
                                f"BER={ber}: PPL={ppl:.2f}±{ppl_std:.2f}, KL={kl:.4f}\n"
                            )

            if results.get("generation_results"):
                f.write("\n\nGENERATION DEMO RESULTS:\n")
                f.write("-" * 50 + "\n")
                for gen in results["generation_results"]:
                    f.write(f"\nPrompt: {gen.get('prompt', 'N/A')}\n")
                    f.write(f"Mode: {gen.get('mode', 'N/A')}\n")
                    f.write(f"Output: {gen.get('output', 'N/A')}\n")

        print(f"Summary saved to volume: {summary_file}")

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

                    ber_levels = []
                    for mode_data in sweep_data.values():
                        ber_levels = list(mode_data.keys())
                        break

                    header = f"{'Cache Mode':<25}"
                    for ber in ber_levels:
                        header += f" | BER={ber:>8}"
                    f.write(header + "\n")
                    f.write("-" * len(header) + "\n")

                    for mode, ber_results in sweep_data.items():
                        row = f"{mode:<25}"
                        for ber in ber_levels:
                            metrics = ber_results.get(ber, {})
                            ppl = metrics.get("ppl_mean", 0)
                            ppl_std = metrics.get("ppl_std", 0)
                            row += f" | {ppl:>6.2f}±{ppl_std:<4.2f}"
                        f.write(row + "\n")

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
        output_volume.commit()
        print(f"To download: modal volume get hamming74-results {models}_benchmark_{timestamp} ./results/")

    return results


def format_ppl_table(results):
    """Format perplexity results as a markdown table.

    Copied from evaluation/runners/triton_eval.py to avoid GPU imports locally.
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in results:
        key = (r["mode"], r["ber"])
        grouped[key].append(r["ppl"])

    modes = sorted(set(r["mode"] for r in results))
    bers = sorted(set(r["ber"] for r in results))

    lines = []
    lines.append("| Protection | " + " | ".join(f"BER={b:.0e}" for b in bers) + " |")
    lines.append("|" + "----|" * (len(bers) + 1))

    for mode in modes:
        row = f"| {mode} |"
        for ber in bers:
            ppls = grouped.get((mode, ber), [])
            if ppls:
                mean_ppl = sum(ppls) / len(ppls)
                std_ppl = (
                    (sum((p - mean_ppl) ** 2 for p in ppls) / len(ppls)) ** 0.5
                    if len(ppls) > 1
                    else 0
                )
                if mean_ppl > 1000:
                    row += f" >1000 |"
                else:
                    row += f" {mean_ppl:.1f}+/-{std_ppl:.1f} |"
            else:
                row += " - |"
        lines.append(row)

    return "\n".join(lines)


def aggregate_results(results):
    """Aggregate results across seeds.

    Copied from evaluation/runners/triton_eval.py to avoid GPU imports locally.
    """
    from collections import defaultdict
    import statistics

    grouped = defaultdict(list)
    for r in results:
        key = (r["mode"], r["ber"])
        grouped[key].append(r)

    aggregated = {}
    for (mode, ber), trials in grouped.items():
        ppls = [t["ppl"] for t in trials if t["ppl"] != float("inf")]

        if mode not in aggregated:
            aggregated[mode] = {}

        aggregated[mode][str(ber)] = {
            "ppl_mean": statistics.mean(ppls) if ppls else float("inf"),
            "ppl_std": statistics.stdev(ppls) if len(ppls) > 1 else 0.0,
            "ppl_min": min(ppls) if ppls else float("inf"),
            "ppl_max": max(ppls) if ppls else float("inf"),
            "num_trials": len(trials),
            "num_valid": len(ppls),
            "total_injection_count": sum(t.get("injection_count", 0) for t in trials),
            "total_errors_corrected": sum(t.get("errors_corrected", 0) for t in trials),
            "total_errors_detected": sum(t.get("errors_detected", 0) for t in trials),
        }

    return aggregated


def save_triton_results(results, output_path):
    os.makedirs(output_path, exist_ok=True)

    with open(f"{output_path}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    aggregated = aggregate_results(results)
    with open(f"{output_path}/aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    ppl_table = format_ppl_table(results)
    with open(f"{output_path}/ppl_table.md", "w") as f:
        f.write("# Triton ECC Perplexity Results\n\n")
        f.write(ppl_table)

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

        for mode, mode_data in aggregated.items():
            f.write(f"\n{mode.upper()}:\n")
            for ber_str, stats in mode_data.items():
                ppl_mean = stats["ppl_mean"]
                ppl_std = stats["ppl_std"]
                if ppl_mean > 1000:
                    f.write(f"BER={ber_str}: PPL > 1000 (catastrophic)\n")
                else:
                    f.write(f"BER={ber_str}: PPL = {ppl_mean:.2f} +/- {ppl_std:.2f}\n")


@app.local_entrypoint()
def main(
    test_file=None,
    verify=False,
    verify_quantization_flag=False,
    compare_quant_ecc=False,
    llama_tests=False,
    benchmark=False,
    models="gpt2",
    seeds=3,
    include_generation=False,
    ber=0.05,
    output=None,
    latency=False,
    latency_iterations=100,
    eval_triton=False,
    triton_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    triton_seeds=3,
    triton_samples=50,
    benchmark_kernels=False,
    list_results_flag=False,
    pull_results=None,
    pull_latest=False,
    results_dir="./results",
):
    if list_results_flag:
        print("\n" + "=" * 70)
        print("AVAILABLE BENCHMARK RESULTS")
        print("=" * 70)

        results_list = list_results.remote()

        print(f"\n{'Run Name':<45} | {'Files':<30} | Timestamp")
        print("-" * 95)

        for r in results_list:
            ts = datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            files = ", ".join(r["files"])
            print(f"{r['name']:<45} | {files:<30} | {ts}")

        print(f"\nTotal: {len(results_list)} benchmark run(s)")
        print("\nTo pull results: modal run modal_runner.py --pull-results <run_name>")
        return

    if pull_results or pull_latest:
        if pull_latest:
            print("\nFetching latest results...")
            results_list = list_results.remote()
            if not results_list:
                print("No benchmark results found.")
                return
            pull_results = results_list[0]["name"]
            print(f"Latest run: {pull_results}")

        print(f"\nPulling results: {pull_results}")
        result_data = get_results.remote(pull_results)

        if "error" in result_data:
            print(f"Error: {result_data['error']}")
            return

        output_path = Path(results_dir) / pull_results
        output_path.mkdir(parents=True, exist_ok=True)

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

        if "summary.txt" in result_data.get("files", {}):
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(result_data["files"]["summary.txt"])

        return

    if llama_tests:
        test_path = test_file if test_file else "tests/test_llama_integration.py"
        return_code = run_llama_tests.remote(test_file=test_path)
        print(f"\nLLaMA tests completed with return code: {return_code}")

    elif verify:
        result = quick_verify.remote()
        print(f"\nVerification: {'PASSED' if result else 'FAILED'}")

    elif verify_quantization_flag:
        print("\n" + "=" * 70)
        print("VERIFYING QUANTIZATION BACKENDS ON GPU")
        print("=" * 70)
        result = verify_quantization.remote()
        print(f"\nQuantization Verification: {'PASSED' if result else 'FAILED'}")

    elif compare_quant_ecc:
        print("\n" + "=" * 70)
        print("QUANTIZATION × ECC COMPARISON BENCHMARK")
        print("=" * 70)
        result = run_quantization_ecc_comparison.remote()

        print("\n" + result.get("comparison_table", "No results"))
        print("\n" + result.get("quantization_table", ""))

        # Save results if output specified
        if output:
            os.makedirs(output, exist_ok=True)
            with open(f"{output}/quant_ecc_comparison.json", "w") as f:
                # Convert aggregated dict to serializable format
                serializable = {
                    "config": result.get("config", {}),
                    "aggregated": result.get("aggregated", {}),
                }
                json.dump(serializable, f, indent=2)
            with open(f"{output}/comparison_table.txt", "w") as f:
                f.write(result.get("comparison_table", ""))
                f.write("\n\n")
                f.write(result.get("quantization_table", ""))
            print(f"\nResults saved to: {output}")

    elif latency:
        results = run_latency_benchmark.remote(n_iterations=latency_iterations)
        print("\n" + "=" * 70)
        print("LATENCY BENCHMARK COMPLETE")
        print("=" * 70)
        print(f"\nBenchmarked {len(results.get('results', []))} configurations")

    elif benchmark_kernels:
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

            print("\n| Configuration | Latency (us) | Overhead |")
            print("|---------------|--------------|----------|")
            for r in benchmarks[:10]:
                overhead = (
                    f"{r.get('overhead_vs_baseline', 0):.2f}x"
                    if r.get("overhead_vs_baseline")
                    else "N/A"
                )
                print(f"| {r['name'][:25]} | {r['latency_us']:.2f} | {overhead} |")

    elif eval_triton:
        print("\n" + "=" * 70)
        print("TRITON ECC PERPLEXITY SWEEP (PARALLEL EXECUTION)")
        print("=" * 70)

        all_modes = [
            "fp16",
            "int4",
            "int4-hamming",
            "int4-hamming84",
            "int4-hamming84-interp",
            "int12-golay",
            "adaptive",
        ]

        modes = all_modes
        bers = [0.0, 1e-4, 1e-3, 1e-2]
        triton_seeds = int(triton_seeds) if isinstance(triton_seeds, str) else triton_seeds
        triton_samples = int(triton_samples) if isinstance(triton_samples, str) else triton_samples
        seed_list = [42, 101, 997][:triton_seeds]

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

        results = []
        for res in run_triton_worker.starmap(args_list):
            ppl_str = f"{res['ppl']:.2f}" if res["ppl"] < 1000 else ">1000"
            print(f"[DONE] {res['mode']} @ BER={res['ber']:.0e} seed={res['seed']}: PPL={ppl_str}")
            results.append(res)

        print("\n" + "=" * 70)
        print("TRITON PPL SWEEP COMPLETE")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(results_dir) / f"triton_ppl_{timestamp}"
        save_triton_results(results, str(output_path))

        print(f"\nResults saved to: {output_path}")

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

        if results.get("sweep_results"):
            for model_key, sweep_data in results["sweep_results"].items():
                print(f"\n{model_key.upper()} Results:")
                for mode in list(sweep_data.keys())[:3]:
                    mode_data = sweep_data[mode]

                    if "0" in mode_data:
                        ppl = mode_data["0"]["ppl_mean"]
                        print(f"  {mode}: PPL={ppl:.2f} (BER=0)")

        if results.get("architecture_comparison"):
            arch = results["architecture_comparison"]
            print(f"\nArchitecture Comparison:")
            print(f"GPT-2: {arch['gpt2_info'].get('n_kv_projections', 'N/A')} KV projections")
            print(f"LLaMA: {arch['llama_info'].get('n_kv_projections', 'N/A')} KV projections")

        if results.get("generation_results"):
            print(f"\nGeneration Demo: {len(results['generation_results'])} samples generated")
    else:
        return_code = run_tests.remote(test_file=test_file)
        print(f"\nTests completed with return code: {return_code}")
