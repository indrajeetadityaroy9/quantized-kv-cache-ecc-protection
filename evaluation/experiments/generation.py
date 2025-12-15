import torch
from dataclasses import dataclass

from ..constants import CACHE_MODE_LABELS


@dataclass
class GenerationResult:
    cache_mode: str
    prompt: str
    generated_text: str
    errors_corrected: int
    errors_detected: int
    total_values: int
    ber: float
    analysis: str


MODE_CONFIG = {
    "fp16": {"codec": "fp16", "use_interpolation": False, "sink_blocks": 0},
    "int4": {"codec": "int4", "use_interpolation": False, "sink_blocks": 0},
    "int4-hamming": {
        "codec": "hamming74",
        "use_interpolation": False,
        "sink_blocks": 0,
    },
    "int4-hamming84": {
        "codec": "hamming84",
        "use_interpolation": False,
        "sink_blocks": 0,
    },
    "int4-hamming84-interp": {
        "codec": "hamming84",
        "use_interpolation": True,
        "sink_blocks": 0,
    },
    "int12-golay": {"codec": "golay", "use_interpolation": False, "sink_blocks": 0},
    "adaptive": {"codec": "adaptive", "use_interpolation": False, "sink_blocks": 4},
    "adaptive-uep": {"codec": "adaptive", "use_interpolation": False, "sink_blocks": 4},
}


def run_generation_demo(
    model,
    tokenizer,
    prompts,
    cache_modes,
    ber=0.05,
    seed=42,
    max_new_tokens=30,
    device="cuda",
):
    from vllm_kernels.shim import (
        ECCShimConfig,
        patch_model_with_ecc_attention,
        reset_ecc_cache,
        get_ecc_stats,
    )

    results = []

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        for cache_mode in cache_modes:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            if cache_mode not in MODE_CONFIG:
                print(f"Warning: Unknown cache_mode '{cache_mode}', skipping")
                continue

            mode_cfg = MODE_CONFIG[cache_mode]

            config = ECCShimConfig(
                codec=mode_cfg["codec"],
                ber=ber,
                inject_errors=(ber > 0),
                seed=seed,
                num_blocks=512,
                block_size=16,
                sink_blocks=mode_cfg["sink_blocks"],
                use_interpolation=mode_cfg["use_interpolation"],
            )

            try:
                with torch.no_grad():
                    with patch_model_with_ecc_attention(model, config, num_blocks=512):
                        reset_ecc_cache(model)

                        output_ids = model.generate(
                            input_ids,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                stats = get_ecc_stats(model)
                errors_corrected = stats.get("errors_corrected", 0)
                errors_detected = stats.get("errors_detected", 0)
                total_values = stats.get("total_values", 0)

                generated_text = tokenizer.decode(
                    output_ids[0], skip_special_tokens=True
                )
                analysis = analyze_generation(prompt, generated_text, cache_mode)

                results.append(
                    GenerationResult(
                        cache_mode=cache_mode,
                        prompt=prompt,
                        generated_text=generated_text,
                        errors_corrected=errors_corrected,
                        errors_detected=errors_detected,
                        total_values=total_values,
                        ber=ber,
                        analysis=analysis,
                    )
                )

            except Exception as e:
                import traceback

                traceback.print_exc()
                results.append(
                    GenerationResult(
                        cache_mode=cache_mode,
                        prompt=prompt,
                        generated_text=f"ERROR: {e}",
                        errors_corrected=0,
                        errors_detected=0,
                        total_values=0,
                        ber=ber,
                        analysis="FAILURE",
                    )
                )

    return results


def analyze_generation(prompt, generated, cache_mode):
    if cache_mode == "fp16":
        return "BASELINE"

    words = generated.split()
    if len(words) > 5:
        for i in range(len(words) - 3):
            if words[i] == words[i + 1] == words[i + 2] == words[i + 3]:
                return "ATTENTION SINK - repetition loop"

    if "\ufffd" in generated:
        return "DECODE ERROR - invalid tokens"

    if cache_mode == "int4":
        return "UNPROTECTED"
    elif cache_mode == "int4-hamming":
        return "SEC - may have miscorrections"
    elif cache_mode == "int4-hamming84":
        return "SECDED - zeroed double errors"
    elif cache_mode == "int4-hamming84-interp":
        return "SECDED+INTERP - smoothed errors"
    elif cache_mode == "int12-golay":
        return "ALGEBRAIC RESTORATION"
    elif cache_mode in ("adaptive", "adaptive-uep"):
        return "ADAPTIVE UEP - sink+context protection"

    return ""


def format_generation_results(results):
    lines = []
    lines.append("=" * 80)
    lines.append("TEXT GENERATION COMPARISON")
    lines.append("=" * 80)

    current_prompt = None
    for r in results:
        if r.prompt != current_prompt:
            current_prompt = r.prompt
            lines.append(f'\nPrompt: "{r.prompt}"')
            lines.append("-" * 80)

        label = CACHE_MODE_LABELS.get(r.cache_mode, r.cache_mode)

        if r.cache_mode == "fp16":
            stats_str = "No quantization (baseline)"
        else:
            stats_parts = []
            if r.errors_corrected > 0:
                stats_parts.append(f"Corrected: {r.errors_corrected:,}")
            if r.errors_detected > 0:
                stats_parts.append(f"Detected: {r.errors_detected:,}")
            stats_parts.append(f"Processed: {r.total_values:,} values")
            stats_str = " | ".join(stats_parts)

        lines.append(f"\n[{label}]")
        lines.append(f'Output: "{r.generated_text}"')
        lines.append(f"Stats:  {stats_str}")
        if r.analysis:
            lines.append(f"Analysis: {r.analysis}")

    lines.append("\n" + "=" * 80)
    lines.append("Analysis Key:")
    lines.append("  - BASELINE: Reference output (FP16, no errors)")
    lines.append("  - UNPROTECTED: INT4 with no error correction")
    lines.append(
        "  - ATTENTION SINK: Repetition loop from bit flip causing softmax collapse"
    )
    lines.append(
        "  - SEC: Single-error correction (Hamming 7,4) - may miscorrect 2-bit errors"
    )
    lines.append("  - SECDED: Single-error correct, double-detect (Hamming 8,4)")
    lines.append(
        "  - SECDED+INTERP: SECDED with interpolation for detected double errors"
    )
    lines.append("  - ALGEBRAIC RESTORATION: Golay(24,12) corrects up to 3 errors")
    lines.append("  - ADAPTIVE UEP: Golay for sink tokens, SECDED+interp for context")
    lines.append("=" * 80)

    return "\n".join(lines)


def results_to_dict(results):
    return [
        {
            "cache_mode": r.cache_mode,
            "prompt": r.prompt,
            "generated_text": r.generated_text,
            "errors_corrected": r.errors_corrected,
            "errors_detected": r.errors_detected,
            "total_values": r.total_values,
            "ber": r.ber,
            "analysis": r.analysis,
        }
        for r in results
    ]


DEFAULT_GENERATION_PROMPTS = [
    "The theory of linear algebra states that",
    "In the field of machine learning,",
    "The fundamental theorem of calculus",
]


DEFAULT_GENERATION_MODES = [
    "fp16",
    "int4",
    "int4-hamming",
    "int4-hamming84",
    "int4-hamming84-interp",
    "int12-golay",
    "adaptive",
]
