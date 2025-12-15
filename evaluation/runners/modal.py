import os
from typing import List, Dict, Any, Optional


LLAMA_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_SEEDS = [42, 101, 997]


def setup_modal_environment():
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"


def get_gpu_info() -> Dict[str, Any]:
    import torch

    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }


def run_tests_impl(test_file: str = None, verbose: bool = True) -> int:
    import subprocess
    import sys

    os.chdir("/app")
    setup_modal_environment()

    cmd = ["python", "-m", "pytest"]

    if test_file:
        cmd.append(test_file)
    else:
        cmd.append("tests/")

    if verbose:
        cmd.extend(["-v", "--tb=short"])

    print("=" * 60)
    print("GPU Test Runner")
    print("=" * 60)

    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['name']}")
    print(f"CUDA Version: {gpu_info['cuda_version']}")
    print(f"Memory: {gpu_info['memory_gb']:.1f} GB")

    print("=" * 60)
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def quick_verify_impl() -> bool:
    import sys

    os.chdir("/app")
    sys.path.insert(0, "/app")
    setup_modal_environment()

    import torch

    gpu_info = get_gpu_info()
    print(f"GPU available: {gpu_info['available']}")
    print(f"GPU: {gpu_info['name']}")

    from hamming74 import Hamming74, Hamming84, Golay2412, INT4Quantizer

    # Test Hamming(7,4)
    print("\n[1/4] Testing Hamming(7,4) codec...")
    codec74 = Hamming74(device="cuda")
    data74 = torch.randint(0, 16, (1000,), dtype=torch.uint8, device="cuda")
    cw74 = codec74.encode(data74)
    decoded74, errors74 = codec74.decode(cw74)
    assert torch.equal(decoded74, data74), "Hamming(7,4) roundtrip failed"
    assert not errors74.any(), "Unexpected errors in Hamming(7,4)"
    print("Hamming(7,4): PASSED")

    # Test Hamming(8,4) SECDED
    print("\n[2/4] Testing Hamming(8,4) SECDED codec...")
    codec84 = Hamming84(device="cuda", on_double_error="keep")
    data84 = torch.randint(0, 16, (1000,), dtype=torch.uint8, device="cuda")
    cw84 = codec84.encode(data84)
    result84 = codec84.decode(cw84)
    assert torch.equal(result84.data, data84), "Hamming(8,4) roundtrip failed"
    print("Hamming(8,4) SECDED: PASSED")

    # Test Golay(24,12)
    print("\n[3/4] Testing Golay(24,12) codec...")
    golay = Golay2412(device="cuda")
    # Create triplets for Golay (each triplet -> 12 bits -> 24 bit codeword)
    triplets = torch.randint(0, 16, (333, 3), dtype=torch.uint8, device="cuda")
    cw_golay = golay.encode(triplets)
    result_golay = golay.decode(cw_golay)
    assert torch.equal(result_golay.data, triplets), "Golay(24,12) roundtrip failed"
    print("Golay(24,12): PASSED")

    # Test INT4Quantizer
    print("\n[4/4] Testing INT4Quantizer...")
    quantizer = INT4Quantizer(block_size=32)
    x = torch.randn(128, 256, dtype=torch.float16)
    q, scales = quantizer.quantize(x)
    x_restored = quantizer.dequantize(q, scales)
    # INT4 quantization has some loss, check reasonable error bound
    mse = ((x - x_restored) ** 2).mean().item()
    assert mse < 1.0, f"INT4Quantizer MSE too high: {mse}"
    print(f"INT4Quantizer: PASSED (MSE={mse:.4f})")

    print("\n" + "=" * 50)
    print("ALL GPU CODEC TESTS PASSED")
    print("=" * 50)

    return True


def verify_quantization_backends_impl() -> bool:
    """Verify all quantization backends work correctly."""
    import sys

    os.chdir("/app")
    sys.path.insert(0, "/app")
    setup_modal_environment()

    import torch
    from hamming74.quantization import INT4Quantizer
    from hamming74.quantization_backends import (
        list_backends,
        get_quantizer,
        QuantizationMode,
        quantize_kv_cache,
        dequantize_kv_cache,
    )

    gpu_info = get_gpu_info()
    print("=" * 60)
    print("QUANTIZATION BACKENDS VERIFICATION")
    print("=" * 60)
    print(f"GPU: {gpu_info['name']}")
    print(f"Available backends: {list_backends()}")

    # Test tensor
    x = torch.randn(2, 4, 64, dtype=torch.float16)
    print(f"\nTest tensor shape: {x.shape}")
    print("-" * 60)

    results = {}
    for backend_name in list_backends():
        try:
            quantizer = get_quantizer(backend_name)
            qt = quantizer.quantize(x)
            x_recon = quantizer.dequantize(qt)
            mse = ((x - x_recon) ** 2).mean().item()
            results[backend_name] = mse
            print(f"  {backend_name:<15}: MSE={mse:.6f}")
            assert mse < 2.0, f"{backend_name} MSE too high"
        except Exception as e:
            print(f"  {backend_name:<15}: FAILED - {e}")
            results[backend_name] = None

    # Test KIVI KV quantization
    print("\n" + "-" * 60)
    print("Testing KIVI KV Cache Quantization:")
    print("-" * 60)

    keys = torch.randn(2, 8, 64, 32, dtype=torch.float16)
    values = torch.randn(2, 8, 64, 32, dtype=torch.float16)

    q_keys, q_values = quantize_kv_cache(keys, values, backend="kivi")
    keys_recon, values_recon = dequantize_kv_cache(q_keys, q_values, backend="kivi")

    k_mse = ((keys - keys_recon) ** 2).mean().item()
    v_mse = ((values - values_recon) ** 2).mean().item()

    print(f"  Keys:   {keys.shape} -> scales={q_keys.scales.shape}, MSE={k_mse:.6f}")
    print(f"  Values: {values.shape} -> scales={q_values.scales.shape}, MSE={v_mse:.6f}")

    assert k_mse < 2.0, f"KIVI keys MSE too high: {k_mse}"
    assert v_mse < 2.0, f"KIVI values MSE too high: {v_mse}"

    # Test INT4Quantizer backward compatibility
    print("\n" + "-" * 60)
    print("Testing INT4Quantizer Backward Compatibility:")
    print("-" * 60)

    # Original interface
    quantizer = INT4Quantizer(block_size=32)
    q, scales = quantizer.quantize(x)
    x_recon = quantizer.dequantize(q, scales)
    mse = ((x - x_recon) ** 2).mean().item()
    print(f"  Original interface (block_absmax): MSE={mse:.6f}")
    assert mse < 2.0

    # New KIVI backend interface using quantize_full for asymmetric quantization
    quantizer_kivi = INT4Quantizer(backend="kivi")
    qt_keys = quantizer_kivi.quantize_full(keys, mode="key")
    keys_recon = quantizer_kivi.dequantize_full(qt_keys)
    k_mse = ((keys - keys_recon) ** 2).mean().item()
    print(f"  KIVI quantize_full (keys): MSE={k_mse:.6f}")
    print(f"    - Has zero_points: {qt_keys.zero_points is not None}")
    print(f"    - Group size: {qt_keys.metadata.get('group_size', 'N/A')}")
    assert k_mse < 2.0

    # Also test the simple interface with cached zero_points
    q_k2, k_scales2 = quantizer_kivi.quantize(keys, mode="key")
    keys_recon2 = quantizer_kivi.dequantize(q_k2, k_scales2, mode="key")
    k_mse2 = ((keys - keys_recon2) ** 2).mean().item()
    print(f"  KIVI simple interface (cached zp): MSE={k_mse2:.6f}")
    assert k_mse2 < 2.0

    print("\n" + "=" * 60)
    print("ALL QUANTIZATION BACKEND TESTS PASSED")
    print("=" * 60)

    return True


def run_benchmark_impl(
    models: str = "gpt2",
    seeds: int = 3,
    include_generation: bool = False,
    generation_ber: float = 0.05,
    output_dir: str = None,
) -> Dict[str, Any]:
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
    print(f"GPU: {gpu_info.get('name', 'CUDA')}")
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

    device = "cuda"
    hf_token = os.environ.get("HF_TOKEN")

    if models == "both":
        target_models = ["gpt2", LLAMA_MODEL]
    elif models == "llama":
        target_models = [LLAMA_MODEL]
    else:
        target_models = ["gpt2"]

    all_seeds = get_seeds()
    seeds = int(seeds) if isinstance(seeds, str) else seeds
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

    texts = load_wikitext2_test(max_samples=20)

    if models == "both":
        print("\n" + "=" * 70)
        print("ARCHITECTURE COMPARISON: GPT-2 vs LLaMA")
        print("=" * 70)

        print("\nLoading GPT-2...")
        gpt2_model, gpt2_tokenizer = load_model("gpt2", device=device)

        print(f"\nLoading {LLAMA_MODEL}...")
        llama_model, llama_tokenizer = load_model(
            LLAMA_MODEL,
            device=device,
            hf_token=hf_token,
        )

        comparison = run_architecture_comparison(
            gpt2_model,
            gpt2_tokenizer,
            llama_model,
            llama_tokenizer,
            texts,
            get_ber_levels(),
        )

        report = generate_comparison_report(comparison)
        print("\n" + report)

        results["architecture_comparison"] = {
            "gpt2_info": comparison.gpt2_info.__dict__,
            "llama_info": comparison.llama_info.__dict__,
        }

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / "architecture_report.txt", "w") as f:
                f.write(report)
            plot_comparison(
                comparison, str(output_path / "architecture_comparison.png")
            )

        if include_generation:
            print("\n" + "=" * 70)
            print(f"GENERATION DEMO @ BER = {generation_ber:.2e}")
            print("=" * 70)

            gen_results = run_generation_demo(
                llama_model,
                llama_tokenizer,
                prompts=DEFAULT_GENERATION_PROMPTS,
                cache_modes=DEFAULT_GENERATION_MODES,
                ber=generation_ber,
                device=device,
            )
            print(format_generation_results(gen_results))
            results["generation_results"] = results_to_dict(gen_results)

        del gpt2_model, gpt2_tokenizer

    else:
        for model_name in target_models:
            print("\n" + "=" * 70)
            print(f"BENCHMARK: {model_name}")
            print("=" * 70)

            print(f"\nLoading {model_name}...")

            model_device = device
            model, tokenizer = load_model(
                model_name,
                device=model_device,
                hf_token=hf_token if "llama" in model_name.lower() else None,
            )

            print("\nGenerating clean baseline logits for KL divergence...")
            with torch.no_grad():
                clean_logits = generate_clean_logits(
                    model, tokenizer, texts, max_length=512, device=model_device
                )
            print(f"  Generated {len(clean_logits)} clean logits")

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

            def progress_cb(msg, current, total):
                print(f"  [{current+1}/{total}] {msg}")

            print(f"\nRunning Monte Carlo sweep with {len(seed_list)} seeds...")
            sweep_results = run_sweep(
                model, tokenizer, texts, sweep_config, progress_callback=progress_cb
            )
            print("\n" + format_results_table(sweep_results))

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

            if include_generation and "llama" in model_name.lower():
                print("\n" + "=" * 70)
                print(f"GENERATION DEMO @ BER = {generation_ber:.2e}")
                print("=" * 70)

                gen_results = run_generation_demo(
                    model,
                    tokenizer,
                    prompts=DEFAULT_GENERATION_PROMPTS,
                    cache_modes=DEFAULT_GENERATION_MODES,
                    ber=generation_ber,
                    device=model_device,
                )
                print(format_generation_results(gen_results))
                results["generation_results"] = results_to_dict(gen_results)

            del model, tokenizer
            torch.cuda.empty_cache()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results
