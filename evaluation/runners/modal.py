"""
Modal GPU Runner for Experiments.

GPU acceleration benefits:
- LLaMA-3.1-8B tests: requires A100-80GB (~16GB VRAM in FP16)
- Verification tests with GPT-2: ~5x faster on T4
- Core Hamming tests: minimal benefit (CPU-bound operations)

This module provides the core Modal functions. The actual Modal app
definition lives in modal_runner.py at the project root.
"""

import os
from typing import List, Dict, Any, Optional

# Note: Modal-specific imports are done at function level to allow
# importing this module without Modal installed (for type hints, etc.)


# Configuration constants
LLAMA_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_SEEDS = [42, 101, 997]  # 3 seeds for faster Modal runs


def setup_modal_environment():
    """Set up environment variables for Modal functions."""
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information for logging."""
    import torch

    if torch.cuda.is_available():
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    return {"available": False}


def run_tests_impl(test_file: str = None, verbose: bool = True) -> int:
    """
    Run pytest on GPU.

    Args:
        test_file: Specific test file to run (e.g., tests/test_verification.py)
        verbose: Enable verbose output

    Returns:
        pytest return code
    """
    import subprocess
    import sys

    os.chdir("/app")
    setup_modal_environment()

    # Build pytest command
    cmd = ["python", "-m", "pytest"]

    if test_file:
        cmd.append(test_file)
    else:
        cmd.append("tests/")

    if verbose:
        cmd.extend(["-v", "--tb=short"])

    # Show GPU info
    print("=" * 60)
    print("GPU Test Runner")
    print("=" * 60)

    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        print(f"GPU: {gpu_info['name']}")
        print(f"CUDA Version: {gpu_info['cuda_version']}")
        print(f"Memory: {gpu_info['memory_gb']:.1f} GB")
    else:
        print("WARNING: No GPU available!")

    print("=" * 60)
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def quick_verify_impl() -> bool:
    """
    Quick verification that everything works on GPU.

    Returns:
        True if all checks pass
    """
    import sys

    os.chdir("/app")
    sys.path.insert(0, "/app")
    setup_modal_environment()

    import torch

    gpu_info = get_gpu_info()
    print(f"GPU available: {gpu_info['available']}")
    if gpu_info["available"]:
        print(f"GPU: {gpu_info['name']}")

    # Test imports
    from hamming74 import Hamming74, INT4Quantizer

    # Test Hamming on CPU (uint8 matmul not supported on CUDA)
    codec = Hamming74(device="cpu")
    quantizer = INT4Quantizer(block_size=32)

    # Test encode/decode on CPU
    data = torch.randint(0, 16, (1000,), dtype=torch.uint8)
    cw = codec.encode(data)
    decoded, errors = codec.decode(cw)

    assert torch.equal(decoded, data), "Hamming roundtrip failed"
    assert not errors.any(), "Unexpected errors"

    print("\nCore Hamming tests: PASSED (CPU - uint8 matmul not supported on CUDA)")

    # Test with model on GPU using Triton ECC shim
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from vllm_kernels.shim import ECCShimConfig, patch_model_with_ecc_attention, get_ecc_stats

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()

    config = ECCShimConfig(codec="hamming84", ber=0.001, inject_errors=True, seed=42)
    input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        with patch_model_with_ecc_attention(model, config, num_blocks=512):
            _ = model(input_ids)

    stats = get_ecc_stats(model)
    print(f"Model patching on {device.upper()}: PASSED "
          f"(processed {stats.get('injection_count', 0)} injections, "
          f"corrected {stats.get('errors_corrected', 0)} errors)")

    return True


def run_benchmark_impl(
    models: str = "gpt2",
    seeds: int = 3,
    include_generation: bool = False,
    generation_ber: float = 0.05,
    output_dir: str = None,
) -> Dict[str, Any]:
    """
    Unified benchmark runner.

    Handles:
    - Single model (gpt2 or llama) -> PPL sweep with Monte Carlo
    - Both models -> Architecture comparison
    - include_generation -> Piggyback text generation demo

    Args:
        models: "gpt2", "llama", or "both"
        seeds: Number of Monte Carlo seeds
        include_generation: Run qualitative generation demo (LLaMA only)
        generation_ber: BER for generation demo
        output_dir: Optional output directory for results

    Returns:
        Dictionary containing benchmark results
    """
    import sys
    import json
    from pathlib import Path

    os.chdir("/app")
    sys.path.insert(0, "/app")
    setup_modal_environment()

    import torch

    gpu_info = get_gpu_info()
    print("=" * 70)
    print("UNIFIED BENCHMARK RUNNER")
    print("=" * 70)
    print(f"GPU: {gpu_info.get('name', 'CPU')}")
    if gpu_info["available"]:
        print(f"GPU Memory: {gpu_info['memory_gb']:.1f} GB")
    print(f"Models: {models}")
    print(f"Seeds: {seeds}")
    print(f"Include Generation: {include_generation}")
    print("=" * 70)

    from ..models import load_model
    from ..metrics import load_wikitext2_test, generate_clean_logits
    from ..constants import get_cache_modes, get_ber_levels, get_seeds
    from ..sweep import SweepConfig, run_sweep
    from ..experiments.monte_carlo import format_results_table
    from ..experiments.architecture import (
        run_architecture_comparison,
        generate_comparison_report,
        plot_comparison,
    )
    from ..experiments.generation import (
        run_generation_demo,
        format_generation_results,
        results_to_dict,
        DEFAULT_GENERATION_PROMPTS,
        DEFAULT_GENERATION_MODES,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.environ.get("HF_TOKEN")

    # Determine which models to run
    if models == "both":
        target_models = ["gpt2", LLAMA_MODEL]
    elif models == "llama":
        target_models = [LLAMA_MODEL]
    else:
        target_models = ["gpt2"]

    # Get seed list
    all_seeds = get_seeds()
    seed_list = all_seeds[:seeds] if seeds <= len(all_seeds) else all_seeds

    results = {
        "config": {
            "models": models,
            "seeds": seed_list,
            "include_generation": include_generation,
            "generation_ber": generation_ber,
        },
        "sweep_results": {},
        "generation_results": None,
        "architecture_comparison": None,
    }

    # Load test data
    texts = load_wikitext2_test(max_samples=20)

    # Handle architecture comparison mode
    if models == "both":
        print("\n" + "=" * 70)
        print("ARCHITECTURE COMPARISON: GPT-2 vs LLaMA")
        print("=" * 70)

        # Load GPT-2
        print("\nLoading GPT-2...")
        gpt2_model, gpt2_tokenizer = load_model("gpt2", device="cpu")

        # Load LLaMA
        print(f"\nLoading {LLAMA_MODEL}...")
        llama_model, llama_tokenizer = load_model(
            LLAMA_MODEL,
            device=device,
            hf_token=hf_token,
        )

        # Run comparison
        comparison = run_architecture_comparison(
            gpt2_model, gpt2_tokenizer,
            llama_model, llama_tokenizer,
            texts, get_ber_levels(),
        )

        # Generate and print report
        report = generate_comparison_report(comparison)
        print("\n" + report)

        results["architecture_comparison"] = {
            "gpt2_info": comparison.gpt2_info.__dict__,
            "llama_info": comparison.llama_info.__dict__,
        }

        # Save if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / "architecture_report.txt", "w") as f:
                f.write(report)
            plot_comparison(comparison, str(output_path / "architecture_comparison.png"))

        # Generation demo on LLaMA if requested
        if include_generation:
            print("\n" + "=" * 70)
            print(f"GENERATION DEMO @ BER = {generation_ber:.2e}")
            print("=" * 70)

            gen_results = run_generation_demo(
                llama_model, llama_tokenizer,
                prompts=DEFAULT_GENERATION_PROMPTS,
                cache_modes=DEFAULT_GENERATION_MODES,
                ber=generation_ber,
                device=device,
            )
            print(format_generation_results(gen_results))
            results["generation_results"] = results_to_dict(gen_results)

        # Clean up GPT-2 to free memory
        del gpt2_model, gpt2_tokenizer

    else:
        # Single model sweep
        for model_name in target_models:
            print("\n" + "=" * 70)
            print(f"BENCHMARK: {model_name}")
            print("=" * 70)

            # Load model
            print(f"\nLoading {model_name}...")
            # Use GPU for all models when available
            model_device = device
            model, tokenizer = load_model(
                model_name,
                device=model_device,
                hf_token=hf_token if "llama" in model_name.lower() else None,
            )

            # Generate clean logits for KL divergence computation
            print("\nGenerating clean baseline logits for KL divergence...")
            with torch.no_grad():
                clean_logits = generate_clean_logits(
                    model, tokenizer, texts,
                    max_length=512, device=model_device
                )
            print(f"  Generated {len(clean_logits)} clean logits")

            # Create sweep config with advanced metrics enabled
            sweep_config = SweepConfig(
                cache_modes=get_cache_modes(),
                ber_levels=get_ber_levels(),
                seeds=seed_list,
                device=model_device,
                aggregate_seeds=True,
                compute_kl_divergence=True,
                compute_top5=True,
                compute_catastrophic=True,
                clean_logits=clean_logits,
                backend="triton",
            )

            # Run sweep with progress callback
            def progress_cb(msg, current, total):
                print(f"  [{current+1}/{total}] {msg}")

            print(f"\nRunning Monte Carlo sweep with {len(seed_list)} seeds...")
            sweep_results = run_sweep(model, tokenizer, texts, sweep_config, progress_callback=progress_cb)
            print("\n" + format_results_table(sweep_results))

            # Store results
            model_key = "gpt2" if model_name == "gpt2" else "llama"
            results["sweep_results"][model_key] = {
                mode: {
                    str(ber): {
                        "ppl_mean": agg.ppl_mean,
                        "ppl_std": agg.ppl_std,
                        "kl_divergence_mean": agg.kl_divergence_mean,
                        "kl_divergence_std": agg.kl_divergence_std,
                        "top5_accuracy_mean": agg.top5_accuracy_mean,
                        "top5_accuracy_std": agg.top5_accuracy_std,
                        "catastrophic_rate_mean": agg.catastrophic_rate_mean,
                        "catastrophic_rate_std": agg.catastrophic_rate_std,
                        "errors_corrected": agg.errors_corrected_mean,
                        "errors_detected": agg.errors_detected_mean,
                    }
                    for ber, agg in ber_results.items()
                }
                for mode, ber_results in sweep_results.aggregated.items()
            }

            # Generation demo if requested and this is LLaMA
            if include_generation and "llama" in model_name.lower():
                print("\n" + "=" * 70)
                print(f"GENERATION DEMO @ BER = {generation_ber:.2e}")
                print("=" * 70)

                gen_results = run_generation_demo(
                    model, tokenizer,
                    prompts=DEFAULT_GENERATION_PROMPTS,
                    cache_modes=DEFAULT_GENERATION_MODES,
                    ber=generation_ber,
                    device=model_device,
                )
                print(format_generation_results(gen_results))
                results["generation_results"] = results_to_dict(gen_results)

            # Clean up
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results
