#!/usr/bin/env python3
"""Quick validation test for ECC-protected KV cache modes."""

import sys
import time

def print_header(msg):
    print(f"\n{'='*60}")
    print(f" {msg}")
    print('='*60)

def test_perplexity():
    """Test perplexity computation across cache modes and BER levels."""
    print_header("PERPLEXITY TEST")

    from evaluation.metrics import load_wikitext2_test
    from evaluation.runners.vllm_runner import VLLMEvaluationRunner, CACHE_MODE_TO_VLLM_DTYPE

    # Load small sample of WikiText2
    print("Loading WikiText2 test set (10 samples)...")
    texts = load_wikitext2_test(max_samples=10)
    print(f"Loaded {len(texts)} text samples")

    # Test cache modes (subset for quick validation)
    cache_modes = ["fp16", "int4-hamming84", "int4-golay-hybrid"]
    ber_levels = [0, 1e-4]

    results = {}

    for cache_mode in cache_modes:
        print(f"\n--- Testing cache mode: {cache_mode} ---")
        vllm_dtype = CACHE_MODE_TO_VLLM_DTYPE.get(cache_mode)
        print(f"  vLLM kv_cache_dtype: {vllm_dtype}")

        try:
            runner = VLLMEvaluationRunner(
                model_name="gpt2",
                cache_mode=cache_mode,
                gpu_memory_utilization=0.5,
            )
            print(f"  Runner initialized: {runner}")

            for ber in ber_levels:
                runner.reset_stats()

                # Compute perplexity
                start = time.time()
                ppl = runner.compute_perplexity(texts, max_length=256, stride=128)
                elapsed = time.time() - start

                # Get ECC stats
                stats = runner.get_error_stats()

                key = (cache_mode, ber)
                results[key] = {
                    "perplexity": ppl,
                    "time": elapsed,
                    "corrected": stats.get("total_corrected", 0),
                    "uncorrectable": stats.get("total_uncorrectable", 0),
                }

                print(f"  BER={ber:.0e}: PPL={ppl:.2f}, time={elapsed:.1f}s, "
                      f"corrected={stats.get('total_corrected', 0)}, "
                      f"uncorrectable={stats.get('total_uncorrectable', 0)}")

                # Inject errors if BER > 0
                if ber > 0 and cache_mode != "fp16":
                    print(f"    Injecting errors at BER={ber}...")
                    runner.inject_errors(ber, seed=42)

                    # Recompute with corrupted cache
                    ppl_corrupted = runner.compute_perplexity(texts, max_length=256, stride=128)
                    stats_after = runner.get_error_stats()

                    print(f"    After injection: PPL={ppl_corrupted:.2f}, "
                          f"corrected={stats_after.get('total_corrected', 0)}")

                    results[key]["ppl_after_injection"] = ppl_corrupted
                    results[key]["corrected_after"] = stats_after.get("total_corrected", 0)

            # Clean up
            del runner
            import gc
            gc.collect()
            import torch
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[(cache_mode, "error")] = str(e)

    return results


def test_downstream_tasks():
    """Test downstream task evaluation (MMLU, HellaSwag)."""
    print_header("DOWNSTREAM TASKS TEST")

    from evaluation.metrics import load_mmlu_subset, load_hellaswag_subset
    from evaluation.metrics import evaluate_mmlu, evaluate_hellaswag
    from evaluation.runners.vllm_runner import VLLMEvaluationRunner

    results = {}

    # Test with fp16 baseline first (quick validation)
    cache_mode = "fp16"
    print(f"\nTesting downstream tasks with cache_mode={cache_mode}")

    try:
        runner = VLLMEvaluationRunner(
            model_name="gpt2",
            cache_mode=cache_mode,
            gpu_memory_utilization=0.5,
        )

        # Get the underlying model and tokenizer for evaluation
        # Note: vLLM doesn't expose model directly, need to use generate API
        tokenizer = runner.llm.get_tokenizer()

        # Test MMLU (5 samples from 1 subject)
        print("\n  Testing MMLU...")
        try:
            from evaluation.metrics import MMLU_STEM_SUBJECTS
            mmlu_samples = load_mmlu_subset(
                subjects=MMLU_STEM_SUBJECTS[:1],  # Just 1 subject
                max_samples_per_subject=5,
            )
            print(f"    Loaded {len(mmlu_samples)} MMLU samples")

            # MMLU evaluation needs model.generate or logits - check if compatible
            # For now just verify loading works
            results["mmlu_loaded"] = len(mmlu_samples)
            results["mmlu_sample"] = mmlu_samples[0] if mmlu_samples else None
            print(f"    Sample question: {mmlu_samples[0]['question'][:50]}...")
        except Exception as e:
            print(f"    MMLU error: {e}")
            results["mmlu_error"] = str(e)

        # Test HellaSwag (5 samples)
        print("\n  Testing HellaSwag...")
        try:
            hellaswag_samples = load_hellaswag_subset(max_samples=5)
            print(f"    Loaded {len(hellaswag_samples)} HellaSwag samples")
            results["hellaswag_loaded"] = len(hellaswag_samples)
            results["hellaswag_sample"] = hellaswag_samples[0] if hellaswag_samples else None
            print(f"    Sample context: {hellaswag_samples[0]['context'][:50]}...")
        except Exception as e:
            print(f"    HellaSwag error: {e}")
            results["hellaswag_error"] = str(e)

        # Clean up
        del runner
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ERROR initializing runner: {e}")
        import traceback
        traceback.print_exc()
        results["runner_error"] = str(e)

    return results


def main():
    print_header("ECC-PROTECTED KV CACHE VALIDATION TEST")
    print("Model: gpt2")
    print("Testing: Perplexity + Downstream Tasks")

    # Test 1: Perplexity across cache modes
    print("\n" + "="*60)
    print("TEST 1: PERPLEXITY EVALUATION")
    print("="*60)

    ppl_results = test_perplexity()

    # Test 2: Downstream tasks
    print("\n" + "="*60)
    print("TEST 2: DOWNSTREAM TASKS")
    print("="*60)

    downstream_results = test_downstream_tasks()

    # Summary
    print_header("VALIDATION SUMMARY")

    print("\nPerplexity Results:")
    for (mode, ber), data in sorted(ppl_results.items()):
        if isinstance(data, dict):
            print(f"  {mode} @ BER={ber}: PPL={data['perplexity']:.2f}")
        else:
            print(f"  {mode}: ERROR - {data}")

    print("\nDownstream Tasks:")
    for key, value in downstream_results.items():
        print(f"  {key}: {value}")

    # Check for any errors
    has_errors = any(
        isinstance(v, str) or (isinstance(v, dict) and "error" in str(v).lower())
        for v in list(ppl_results.values()) + list(downstream_results.values())
    )

    if has_errors:
        print("\n[!] Some tests had errors - review output above")
        return 1
    else:
        print("\n[OK] All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
