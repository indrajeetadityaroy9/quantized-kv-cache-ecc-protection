import math
import os
from typing import Dict, Any, List, Tuple, Optional


def load_llama_model(model_name: str) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN")

    cache_dir = os.environ.get("HF_HOME", "/cache/huggingface")

    print(f"[load_llama_model] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        cache_dir=cache_dir,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[load_llama_model] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
        cache_dir=cache_dir,
    )
    model.eval()

    print(
        f"[load_llama_model] Model loaded successfully. Device: {next(model.parameters()).device}"
    )

    return model, tokenizer


def run_single_triton_trial(
    model_name: str,
    mode: str,
    ber: float,
    seed: int,
    max_samples: int,
) -> Dict[str, Any]:
    import torch
    from kv_cache.ecc_shim import (
        ECCShimConfig,
        patch_model_with_ecc_attention,
        reset_ecc_cache,
        get_ecc_stats,
    )
    from evaluation.metrics import load_wikitext2_test

    MODE_CONFIG = {
        "fp16": {"codec": "fp16", "use_interpolation": False},
        "int4": {"codec": "int4", "use_interpolation": False},
        "int4-hamming": {"codec": "hamming74", "use_interpolation": False},
        "int4-hamming84": {"codec": "hamming84", "use_interpolation": False},
        "int4-hamming84-interp": {"codec": "hamming84", "use_interpolation": True},
        "int12-golay": {"codec": "golay", "use_interpolation": False},
        # Aliases for convenience
        "hamming74": {"codec": "hamming74", "use_interpolation": False},
        "hamming84": {"codec": "hamming84", "use_interpolation": False},
        "golay": {"codec": "golay", "use_interpolation": False},
    }

    if mode not in MODE_CONFIG:
        raise ValueError(
            f"Unknown mode: {mode}. Valid modes: {list(MODE_CONFIG.keys())}"
        )

    mode_cfg = MODE_CONFIG[mode]
    codec = mode_cfg["codec"]
    use_interpolation = mode_cfg["use_interpolation"]

    print("=" * 60)
    print(f"[Trial] mode={mode}, ber={ber:.0e}, seed={seed}")
    print(f"        codec={codec}, use_interpolation={use_interpolation}")
    print("=" * 60)

    model, tokenizer = load_llama_model(model_name)

    print(f"[Trial] Loading WikiText-2 test data...")
    texts = load_wikitext2_test(max_samples=max_samples)
    print(f"[Trial] Loaded {len(texts)} documents")

    config = ECCShimConfig(
        codec=codec,
        ber=ber,
        inject_errors=(ber > 0),
        seed=seed,
        num_blocks=2048,
        block_size=16,
        use_interpolation=use_interpolation,
    )

    all_losses = []
    total_tokens = 0

    print(f"[Trial] Starting evaluation with ECC shim...")
    with patch_model_with_ecc_attention(model, config, num_blocks=2048):
        for i, text in enumerate(texts):
            if not text.strip():
                continue

            reset_ecc_cache(model)

            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )
            input_ids = inputs["input_ids"]
            seq_len = input_ids.size(1)

            if seq_len < 2:
                continue

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                if not torch.isnan(loss) and not torch.isinf(loss):
                    all_losses.append(loss.item() * seq_len)
                    total_tokens += seq_len

            if (i + 1) % 10 == 0:
                current_ppl = (
                    math.exp(sum(all_losses) / total_tokens)
                    if total_tokens > 0
                    else float("inf")
                )
                print(
                    f"[Trial] Processed {i + 1}/{len(texts)} documents, running PPL: {current_ppl:.2f}"
                )

        stats = get_ecc_stats(model)

    if total_tokens == 0:
        ppl = float("inf")
    else:
        avg_loss = sum(all_losses) / total_tokens
        ppl = math.exp(avg_loss)

    errors_corrected = stats.get("errors_corrected", 0)
    errors_detected = stats.get("errors_detected", 0)
    injection_count = stats.get("injection_count", 0)

    print("=" * 60)
    print(f"[Trial COMPLETE] mode={mode}, ber={ber:.0e}, seed={seed}")
    print(f"                 PPL={ppl:.2f}, tokens={total_tokens}")
    print(f"                 injection_count={injection_count}")
    print(
        f"                 errors_corrected={errors_corrected}, errors_detected={errors_detected}"
    )
    print("=" * 60)

    return {
        "mode": mode,
        "ber": ber,
        "seed": seed,
        "ppl": ppl,
        "num_samples": len(texts),
        "total_tokens": total_tokens,
        "injection_count": injection_count,
        "errors_corrected": errors_corrected,
        "errors_detected": errors_detected,
    }


def run_triton_ppl_sweep(
    model_name: str,
    modes: List[str],
    ber_levels: List[float],
    seeds: List[int],
    max_samples: int,
) -> List[Dict[str, Any]]:
    results = []

    total_trials = len(modes) * len(ber_levels) * len(seeds)
    trial_num = 0

    for mode in modes:
        for ber in ber_levels:
            for seed in seeds:
                trial_num += 1
                print(
                    f"\n[Sweep {trial_num}/{total_trials}] mode={mode}, ber={ber:.0e}, seed={seed}"
                )

                result = run_single_triton_trial(
                    model_name=model_name,
                    mode=mode,
                    ber=ber,
                    seed=seed,
                    max_samples=max_samples,
                )
                results.append(result)

    return results


def format_ppl_table(results: List[Dict[str, Any]]) -> str:
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


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
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
