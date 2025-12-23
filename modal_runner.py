import os
import sys
import json
from datetime import datetime
from pathlib import Path

import modal

app = modal.App("hamming74-experiments")
hf_secret = modal.Secret.from_name("huggingface-secret")

# =============================================================================
# vLLM Image Build with Layer Caching Optimization
# =============================================================================
# Structure:
#   Layer 1 (base_image): Base dependencies - cached until deps change
#   Layer 2 (build_image): CUDA compilation - cached until csrc/ changes
#   Layer 3 (image): Python source - fast to update, no recompilation
#
# Expected build times:
#   - Full rebuild (csrc/ changed): ~20-30 min
#   - Python-only change (vllm/*.py): ~2-3 min (cached build layer)
#   - Evaluation code change: ~1-2 min (cached build layer)
# =============================================================================

# Layer 1: Base image with all dependencies (rarely changes)
base_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel",
    )
    .apt_install("git", "build-essential", "ninja-build", "ccache")
    .pip_install(
        # vLLM build dependencies
        "setuptools>=61",
        "setuptools-scm>=8",
        "wheel",
        "packaging",
        "ninja",
        "jinja2",
        # vLLM runtime dependencies
        "transformers>=4.40.0",
        "datasets>=2.14.0",
        "accelerate",
        "huggingface-hub",
        "torchmetrics>=1.0.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
        "ray>=2.9.0",
        "sentencepiece",
        "tiktoken",
        "triton>=2.1.0",
        "xformers",
        "filelock",
        "fastapi",
        "uvicorn",
        "pydantic>=2.0",
        "prometheus-client",
        "py-cpuinfo",
        "aiohttp",
        "openai",
        "lm-format-enforcer",
        "outlines",
        "compressed-tensors",
        "gguf",
        "mistral_common",
        "pynvml",
        "msgspec",
        "pytest>=7.0.0",
    )
)

# Layer 2: Build vLLM C extensions (cached until csrc/ or build config changes)
# Only copies build-essential files to maximize cache hit rate
build_image = (
    base_image
    # Copy CUDA sources (ECC kernels) - changes trigger recompilation
    .add_local_dir("vllm-main/csrc", remote_path="/app/vllm-main/csrc", copy=True)
    # Copy CMake config
    .add_local_dir("vllm-main/cmake", remote_path="/app/vllm-main/cmake", copy=True)
    # Copy build configuration files (copy=True required for run_commands after)
    .add_local_file("vllm-main/setup.py", "/app/vllm-main/setup.py", copy=True)
    .add_local_file("vllm-main/pyproject.toml", "/app/vllm-main/pyproject.toml", copy=True)
    .add_local_file("vllm-main/CMakeLists.txt", "/app/vllm-main/CMakeLists.txt", copy=True)
    .add_local_file("vllm-main/use_existing_torch.py", "/app/vllm-main/use_existing_torch.py", copy=True)
    # Copy requirements directory (needed by setup.py)
    .add_local_dir("vllm-main/requirements", remote_path="/app/vllm-main/requirements", copy=True)
    # Copy vllm/envs.py (required by setup.py during build)
    .add_local_file("vllm-main/vllm/envs.py", "/app/vllm-main/vllm/envs.py", copy=True)
    # Create minimal vllm/__init__.py placeholder for setuptools package discovery
    .run_commands(
        "echo '__version__ = \"0.6.4.post1\"' > /app/vllm-main/vllm/__init__.py && "
        "echo '' > /app/vllm-main/vllm/py.typed"
    )
    # Build vLLM from source with ECC kernels (CUDA compilation happens here)
    # Optimized for A100 (sm_80) only - significantly reduces compile time
    #
    # Build optimizations:
    # - TORCH_CUDA_ARCH_LIST=8.0: Only build for A100 (sm_80)
    #   - Skips: FlashMLA, QuTLASS, FA3, Machete, scaled_mm_c3x,
    #     sparse kernels, W4A8, NVFP4 (all require sm_90+ or sm_100+)
    # - MAX_JOBS=12: Matches A100 Modal instance vCPU count
    # - NVCC_THREADS=8: Increased parallelism for template instantiation
    # - CMAKE_BUILD_TYPE=Release: Enable compiler optimizations
    .run_commands(
        "cd /app/vllm-main && pip install . --no-build-isolation -v",
        env={
            "TORCH_CUDA_ARCH_LIST": "8.0",
            "CMAKE_ARGS": "-DCMAKE_CUDA_ARCHITECTURES=80",
            "CMAKE_BUILD_TYPE": "Release",
            "MAX_JOBS": "12",
            "NVCC_THREADS": "8",
            "VERBOSE": "1",
            "SETUPTOOLS_SCM_PRETEND_VERSION": "0.6.4.post1",
        },
        gpu="A100-80GB",
    )
)

# Layer 3: Add Python source and evaluation code (changes frequently, no recompilation)
image = (
    build_image
    # Copy the REAL vllm/ Python source (overwrites placeholder)
    .add_local_dir("vllm-main/vllm", remote_path="/app/vllm-main/vllm", copy=True)
    # Reinstall as editable to pick up Python source (fast, no C++ rebuild)
    .run_commands(
        "cd /app/vllm-main && pip install -e . --no-build-isolation --no-deps"
    )
    # Add evaluation code
    .add_local_dir("evaluation", remote_path="/app/evaluation", copy=True)
)

model_cache = modal.Volume.from_name("hamming74-model-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("hamming74-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",  # Must match build target (sm_80)
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
    timeout=14400,
    volumes={"/cache": model_cache, "/output": output_volume},
    secrets=[hf_secret],
)
def run_evaluation(
    model_name: str = "gpt2",
    dataset: str = "wikitext2",
    max_samples: int = 50,
    num_seeds: int = 3,
    save_to_volume: bool = True,
    cache_modes: list = None,
    ber_levels: list = None,
    ecc_sweep: bool = False,
):
    """Run evaluation using vLLM backend with ECC-protected KV cache.

    Args:
        model_name: Model identifier (gpt2, llama-3.1-8b, mistral-7b, etc.)
        dataset: Dataset name (wikitext2, c4, ptb)
        max_samples: Number of evaluation samples
        num_seeds: Number of random seeds for statistical rigor
        save_to_volume: Whether to save results to Modal volume
        cache_modes: List of cache modes to test
        ber_levels: List of BER levels for fault injection
        ecc_sweep: If True, run full ECC comparison sweep

    Returns:
        Dict with sweep results including perplexity and error correction stats
    """
    os.chdir("/app")
    sys.path.insert(0, "/app")
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"

    import torch
    from evaluation.metrics import load_dataset_by_name
    from evaluation.sweep import SweepConfig, run_sweep
    from evaluation.constants import ICML_CONFIG

    # Full ECC sweep mode with all cache modes and BER levels from ICML_CONFIG
    if ecc_sweep:
        cache_modes = ICML_CONFIG["cache_modes"]  # Uses constants.py definition
        ber_levels = ICML_CONFIG["ber_levels"]

    # Defaults
    if cache_modes is None:
        cache_modes = ["fp16"]
    if ber_levels is None:
        ber_levels = [0]

    print("=" * 80)
    print("ECC-PROTECTED KV CACHE EVALUATION")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Cache modes: {cache_modes}")
    print(f"BER levels: {ber_levels}")
    print(f"Max samples: {max_samples}")
    print(f"Seeds: {num_seeds}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # Load dataset
    print(f"\nLoading {dataset} dataset...")
    texts = load_dataset_by_name(dataset, max_samples=max_samples)
    print(f"Loaded {len(texts)} samples")

    # Configure sweep
    seeds = ICML_CONFIG["seeds"][:num_seeds]
    config = SweepConfig(
        cache_modes=cache_modes,
        ber_levels=ber_levels,
        seeds=seeds,
        aggregate_seeds=True,
        max_length=ICML_CONFIG["max_length"],
        stride=ICML_CONFIG["stride"],
        max_samples=max_samples,
        block_size=32,
        device="cuda",
        compute_kl_divergence=False,
        compute_top5=False,
        compute_catastrophic=True,
        catastrophic_threshold=1000.0,
    )

    # Run sweep using vLLM backend
    def progress_callback(msg, current, total):
        print(f"[{current+1}/{total}] {msg}")

    print("\nRunning perplexity sweep (vLLM backend)...")
    results = run_sweep(model_name, texts, config, progress_callback)

    # Format results
    output_lines = []
    output_lines.append("\n" + "=" * 100)
    output_lines.append(f"PERPLEXITY RESULTS (vLLM): {model_name} on {dataset}")
    output_lines.append("=" * 100)

    header = f"{'Cache Mode':<25} |"
    for ber in config.ber_levels:
        header += f" BER={ber:.0e} |"
    output_lines.append(header)
    output_lines.append("-" * len(header))

    for mode in config.cache_modes:
        row = f"{mode:<25} |"
        for ber in config.ber_levels:
            agg = results.get_aggregated(mode, ber)
            if agg:
                row += f" {agg.ppl_mean:>10.2f} |"
            else:
                row += f" {'N/A':>10} |"
        output_lines.append(row)

    output_lines.append("")
    output_lines.append("=" * 100)
    output_lines.append("ERROR CORRECTION STATS (vLLM C++ Kernels)")
    output_lines.append("=" * 100)

    for mode in config.cache_modes:
        if mode == "fp16":
            continue
        output_lines.append(f"\n{mode}:")
        for ber in config.ber_levels:
            agg = results.get_aggregated(mode, ber)
            if agg:
                output_lines.append(
                    f"  BER={ber:.0e}: PPL={agg.ppl_mean:.2f}+/-{agg.ppl_std:.2f}, "
                    f"Corrected={agg.errors_corrected_mean:.0f}, "
                    f"Catastrophic={agg.catastrophic_rate_mean:.2%}"
                )

    output_text = "\n".join(output_lines)
    print(output_text)

    # Build results dict
    results_dict = {
        "backend": "vllm",
        "config": {
            "model_name": model_name,
            "dataset": dataset,
            "max_samples": max_samples,
            "seeds": seeds,
            "cache_modes": config.cache_modes,
            "ber_levels": config.ber_levels,
        },
        "aggregated": {},
        "trials": [],
    }

    # Save raw trial data
    for trial in results.trials:
        results_dict["trials"].append({
            "cache_mode": trial.cache_mode,
            "ber": trial.ber,
            "seed": trial.seed,
            "perplexity": trial.perplexity,
            "errors_corrected": trial.errors_corrected,
            "errors_detected": trial.errors_detected,
            "total_values": trial.total_values,
            "catastrophic_rate": trial.catastrophic_rate,
        })

    # Save aggregated results
    for mode in config.cache_modes:
        results_dict["aggregated"][mode] = {}
        for ber in config.ber_levels:
            agg = results.get_aggregated(mode, ber)
            if agg:
                results_dict["aggregated"][mode][str(ber)] = {
                    "ppl_mean": agg.ppl_mean,
                    "ppl_std": agg.ppl_std,
                    "errors_corrected_mean": agg.errors_corrected_mean,
                    "errors_detected_mean": agg.errors_detected_mean,
                    "total_values": agg.total_values,
                    "catastrophic_rate_mean": agg.catastrophic_rate_mean,
                    "catastrophic_rate_std": agg.catastrophic_rate_std,
                    "n_trials": agg.n_trials,
                }

    # Save results
    if save_to_volume:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/output/eval_{model_name}_{dataset}_{timestamp}"
        os.makedirs(output_path, exist_ok=True)

        with open(f"{output_path}/summary.txt", "w") as f:
            f.write(output_text)

        with open(f"{output_path}/results.json", "w") as f:
            json.dump(results_dict, f, indent=2)

        output_volume.commit()
        print(f"\nResults saved to: {output_path}")

    return results_dict


@app.local_entrypoint()
def main(
    command: str = "eval",
    model: str = "gpt2",
    dataset: str = "wikitext2",
    samples: int = 50,
    seeds: int = 3,
    all_models: bool = False,
    all_datasets: bool = False,
    ecc_sweep: bool = False,
    cache_modes: str = None,  # Comma-separated list: "fp16,int4-hamming84"
    ber_levels: str = None,   # Comma-separated list: "0,1e-6,1e-5"
    test_file: str = None,
    pull_results: str = None,
    list_results_flag: bool = False,
    pull_latest: bool = False,
    results_dir: str = "./results",
):
    """ECC-Protected KV Cache Evaluation CLI.

    Commands:
      eval      - Run perplexity evaluation with ECC protection
      test      - Run test suite

    Eval Flags:
      --model         Model to evaluate (default: gpt2)
      --dataset       Dataset (default: wikitext2)
      --samples       Number of samples (default: 50)
      --seeds         Number of seeds (default: 3)
      --ecc-sweep     Run full ECC comparison (fp16, hamming, golay) with BER sweep
      --cache-modes   Comma-separated cache modes: "fp16,int4-hamming84"
      --ber-levels    Comma-separated BER levels: "0,1e-6,1e-5,1e-4"

    Examples:
      modal run modal_runner.py --ecc-sweep                        # Full ECC comparison
      modal run modal_runner.py --model gpt2 --samples 20          # Quick baseline
      modal run modal_runner.py --cache-modes "fp16,int4-hamming84" --ber-levels "0,1e-5,1e-4"
      modal run modal_runner.py --list-results-flag                # List results
      modal run modal_runner.py --pull-latest                      # Pull latest results
    """
    # Handle result listing/pulling first
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

    # Handle commands
    if command == "eval":
        from evaluation.constants import ICML_CONFIG

        # Determine models and datasets to evaluate
        models = ICML_CONFIG["models"] if all_models else [model]
        datasets = ICML_CONFIG["datasets"] if all_datasets else [dataset]

        total = len(models) * len(datasets)
        is_sweep = total > 1

        if ecc_sweep:
            print("\n" + "=" * 70)
            print("ECC COMPARISON SWEEP")
            print("=" * 70)
            print(f"Model: {model}")
            print(f"Dataset: {dataset}")
            print("Cache modes: fp16, int4-hamming, int4-hamming84, int12-golay")
            print("BER levels: 0, 1e-6, 1e-5, 1e-4, 1e-3")
            print(f"Samples: {samples}, Seeds: {seeds}")
            print("=" * 70)

            result = run_evaluation.remote(
                model_name=model,
                dataset=dataset,
                max_samples=samples,
                num_seeds=seeds,
                save_to_volume=True,
                ecc_sweep=True,
            )
            print("\n" + "=" * 70)
            print("ECC COMPARISON COMPLETE")
            print("=" * 70)

        elif is_sweep:
            print("\n" + "=" * 70)
            print("MULTI-MODEL EVALUATION SWEEP")
            print("=" * 70)
            print(f"Models: {models}")
            print(f"Datasets: {datasets}")
            print(f"Total evaluations: {total}")
            print(f"Samples: {samples}, Seeds: {seeds}")
            print("=" * 70)

            all_results = []
            current = 0

            for m in models:
                for d in datasets:
                    current += 1
                    print(f"\n[{current}/{total}] Evaluating {m} on {d}...")

                    result = run_evaluation.remote(
                        model_name=m,
                        dataset=d,
                        max_samples=samples,
                        num_seeds=seeds,
                        save_to_volume=True,
                    )
                    all_results.append(result)

            print("\n" + "=" * 70)
            print(f"EVALUATION COMPLETE - {len(all_results)} run(s)")
            print("=" * 70)

        else:
            # Parse custom cache modes and BER levels if provided
            custom_cache_modes = None
            custom_ber_levels = None

            if cache_modes:
                custom_cache_modes = [m.strip() for m in cache_modes.split(",")]
            if ber_levels:
                custom_ber_levels = [float(b.strip()) for b in ber_levels.split(",")]

            print("\n" + "=" * 70)
            print(f"EVALUATION: {model} on {dataset}")
            print("=" * 70)
            print(f"Samples: {samples}, Seeds: {seeds}")
            if custom_cache_modes:
                print(f"Cache modes: {custom_cache_modes}")
            if custom_ber_levels:
                print(f"BER levels: {custom_ber_levels}")

            result = run_evaluation.remote(
                model_name=model,
                dataset=dataset,
                max_samples=samples,
                num_seeds=seeds,
                save_to_volume=True,
                cache_modes=custom_cache_modes,
                ber_levels=custom_ber_levels,
            )

            print("\n" + "=" * 70)
            print("EVALUATION COMPLETE")
            print("=" * 70)

    elif command == "test":
        print("=" * 70)
        print("RUNNING TEST SUITE")
        print("=" * 70)
        return_code = run_tests.remote(test_file=test_file)
        print(f"\nTests completed with return code: {return_code}")

    else:
        print(f"Unknown command: {command}")
        print("Available commands: eval, test")
