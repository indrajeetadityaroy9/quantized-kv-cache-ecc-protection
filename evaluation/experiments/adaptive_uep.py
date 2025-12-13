from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import math
import torch
import torch.nn as nn

from vllm_kernels.shim import (
    ECCShimConfig,
    patch_model_with_ecc_attention,
    get_ecc_stats,
)
from ..metrics import compute_perplexity
from ..constants import BER_LEVELS, get_seeds


@dataclass
class UEPExperimentConfig:
    sink_boundaries: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])

    ber_levels: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])

    comparison_modes: List[str] = field(
        default_factory=lambda: [
            "int4-hamming",
            "int4-hamming84",
            "int12-golay",
            "adaptive-uep",
        ]
    )

    seeds: List[int] = field(default_factory=get_seeds)

    hidden_dim: int = 768
    block_size: int = 32

    max_length: int = 256
    stride: int = 128
    max_samples: int = 50


@dataclass
class UEPExperimentResult:
    mode: str
    ber: float
    sink_token_count: Optional[int]
    seed: int

    perplexity: float
    ppl_delta_vs_fp16: float

    errors_corrected: int = 0
    errors_detected: int = 0
    total_values: int = 0

    sink_errors_corrected: Optional[int] = None
    context_errors_corrected: Optional[int] = None
    migrated_errors_avoided: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "mode": self.mode,
            "ber": self.ber,
            "sink_token_count": self.sink_token_count,
            "seed": self.seed,
            "perplexity": self.perplexity,
            "ppl_delta_vs_fp16": self.ppl_delta_vs_fp16,
            "errors_corrected": self.errors_corrected,
            "errors_detected": self.errors_detected,
            "total_values": self.total_values,
        }
        if self.sink_errors_corrected is not None:
            d["sink_errors_corrected"] = self.sink_errors_corrected
            d["context_errors_corrected"] = self.context_errors_corrected
            d["migrated_errors_avoided"] = self.migrated_errors_avoided
        return d


@dataclass
class UEPBoundarySweepResult:
    ber: float
    seed: int
    results_by_boundary: Dict[int, UEPExperimentResult] = field(default_factory=dict)

    def get_optimal_boundary(self) -> int:
        if not self.results_by_boundary:
            return 128
        return min(
            self.results_by_boundary.keys(),
            key=lambda k: self.results_by_boundary[k].perplexity,
        )


@dataclass
class UEPComparisonResult:
    ber: float
    results_by_mode: Dict[str, List[UEPExperimentResult]] = field(default_factory=dict)

    def get_best_mode(self) -> str:
        mode_means = {}
        for mode, results in self.results_by_mode.items():
            if mode != "fp16" and results:
                mode_means[mode] = sum(r.perplexity for r in results) / len(results)
        if not mode_means:
            return "adaptive-uep"
        return min(mode_means.keys(), key=lambda k: mode_means[k])


def run_single_uep_trial(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    mode: str,
    ber: float,
    seed: int,
    sink_token_count: Optional[int] = None,
    config: Optional[UEPExperimentConfig] = None,
    fp16_baseline: Optional[float] = None,
) -> UEPExperimentResult:
    if config is None:
        config = UEPExperimentConfig()

    device = "cuda"
    model_device = next(model.parameters()).device
    if str(model_device) != device:
        model = model.to(device)

    mode_map = {
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
        "adaptive-uep": {"codec": "adaptive", "use_interpolation": False},
    }
    if mode not in mode_map:
        raise ValueError(f"Unsupported mode for UEP (Triton): {mode}")

    sink_blocks = 0
    if mode == "adaptive-uep":
        stc = sink_token_count or 128
        sink_blocks = math.ceil(stc / config.block_size)

    ecc_config = ECCShimConfig(
        codec=mode_map[mode]["codec"],
        ber=ber,
        inject_errors=(ber > 0),
        seed=seed,
        num_blocks=2048,
        block_size=config.block_size,
        sink_blocks=sink_blocks,
        use_interpolation=mode_map[mode].get("use_interpolation", False),
    )

    with torch.no_grad():
        with patch_model_with_ecc_attention(
            model, ecc_config, num_blocks=ecc_config.num_blocks
        ):
            ppl = compute_perplexity(
                model,
                tokenizer,
                texts,
                max_length=config.max_length,
                stride=config.stride,
                device=device,
            )

            stats = get_ecc_stats(model)

    if fp16_baseline is not None and fp16_baseline > 0:
        ppl_delta = (ppl - fp16_baseline) / fp16_baseline * 100
    else:
        ppl_delta = 0.0

    return UEPExperimentResult(
        mode=mode,
        ber=ber,
        sink_token_count=sink_token_count if mode == "adaptive-uep" else None,
        seed=seed,
        perplexity=ppl,
        ppl_delta_vs_fp16=ppl_delta,
        errors_corrected=stats.get("errors_corrected", 0),
        errors_detected=stats.get("errors_detected", 0),
        total_values=stats.get("injection_count", 0),
        sink_errors_corrected=None,
        context_errors_corrected=None,
        migrated_errors_avoided=None,
    )


def run_uep_boundary_sweep(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    ber: float,
    seed: int,
    config: Optional[UEPExperimentConfig] = None,
    progress_callback=None,
) -> UEPBoundarySweepResult:
    if config is None:
        config = UEPExperimentConfig()

    fp16_result = run_single_uep_trial(
        model, tokenizer, texts, "fp16", 0.0, seed, config=config
    )
    fp16_baseline = fp16_result.perplexity

    result = UEPBoundarySweepResult(ber=ber, seed=seed)
    total = len(config.sink_boundaries)

    for i, boundary in enumerate(config.sink_boundaries):
        if progress_callback:
            progress_callback(f"Testing boundary={boundary} tokens", i, total)

        trial_result = run_single_uep_trial(
            model,
            tokenizer,
            texts,
            mode="adaptive-uep",
            ber=ber,
            seed=seed,
            sink_token_count=boundary,
            config=config,
            fp16_baseline=fp16_baseline,
        )
        result.results_by_boundary[boundary] = trial_result

    return result


def run_uep_comparison(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    ber: float,
    config: Optional[UEPExperimentConfig] = None,
    progress_callback=None,
) -> UEPComparisonResult:
    if config is None:
        config = UEPExperimentConfig()

    fp16_result = run_single_uep_trial(
        model, tokenizer, texts, "fp16", 0.0, config.seeds[0], config=config
    )
    fp16_baseline = fp16_result.perplexity

    result = UEPComparisonResult(ber=ber)
    total = len(config.comparison_modes) * len(config.seeds)
    current = 0

    for mode in config.comparison_modes:
        result.results_by_mode[mode] = []

        for seed in config.seeds:
            if progress_callback:
                progress_callback(f"Testing {mode} @ seed={seed}", current, total)

            if mode == "adaptive-uep":
                trial_result = run_single_uep_trial(
                    model,
                    tokenizer,
                    texts,
                    mode=mode,
                    ber=ber,
                    seed=seed,
                    sink_token_count=128,
                    config=config,
                    fp16_baseline=fp16_baseline,
                )
            else:
                trial_result = run_single_uep_trial(
                    model,
                    tokenizer,
                    texts,
                    mode=mode,
                    ber=ber,
                    seed=seed,
                    config=config,
                    fp16_baseline=fp16_baseline,
                )

            result.results_by_mode[mode].append(trial_result)
            current += 1

    return result


def generate_uep_report(comparison: UEPComparisonResult) -> str:
    lines = [
        "=" * 80,
        f"UEP COMPARISON REPORT (BER = {comparison.ber})",
        "=" * 80,
        "",
        f"{'Mode':<20} | {'PPL (mean)':<12} | {'PPL (std)':<10} | {'Delta %':<10} | {'Errors Corrected':<15}",
        "-" * 80,
    ]

    for mode, results in comparison.results_by_mode.items():
        if not results:
            continue

        ppls = [r.perplexity for r in results]
        deltas = [r.ppl_delta_vs_fp16 for r in results]
        errors = [r.errors_corrected for r in results]

        mean_ppl = sum(ppls) / len(ppls)
        std_ppl = (sum((p - mean_ppl) ** 2 for p in ppls) / len(ppls)) ** 0.5
        mean_delta = sum(deltas) / len(deltas)
        mean_errors = sum(errors) / len(errors)

        lines.append(
            f"{mode:<20} | {mean_ppl:<12.2f} | {std_ppl:<10.3f} | {mean_delta:<10.2f} | {mean_errors:<15.0f}"
        )

    lines.extend(
        [
            "-" * 80,
            "",
            f"Best mode: {comparison.get_best_mode()}",
            "",
        ]
    )

    if "adaptive-uep" in comparison.results_by_mode:
        adaptive_results = comparison.results_by_mode["adaptive-uep"]
        if adaptive_results and adaptive_results[0].migrated_errors_avoided is not None:
            total_migrated = sum(
                r.migrated_errors_avoided or 0 for r in adaptive_results
            )
            total_sink = sum(r.sink_errors_corrected or 0 for r in adaptive_results)
            total_context = sum(
                r.context_errors_corrected or 0 for r in adaptive_results
            )

            lines.extend(
                [
                    "Adaptive UEP Analysis:",
                    f"  Sink errors corrected: {total_sink}",
                    f"  Context errors corrected: {total_context}",
                    f"  Migrated errors avoided: {total_migrated}",
                    "  (Migrated = multi-bit errors Golay corrected that Hamming would have failed)",
                    "",
                ]
            )

    return "\n".join(lines)


def run_undervolting_stress_test(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    config: Optional[UEPExperimentConfig] = None,
    progress_callback=None,
) -> Dict[float, UEPComparisonResult]:
    if config is None:
        config = UEPExperimentConfig()

    results = {}

    for ber in config.ber_levels:
        if progress_callback:
            progress_callback(f"Testing BER = {ber}", 0, 0)

        results[ber] = run_uep_comparison(
            model, tokenizer, texts, ber, config, progress_callback
        )

    return results


if __name__ == "__main__":
    print("Adaptive UEP Experiments")
    print("=" * 50)
    print("Use run_uep_comparison() or run_uep_boundary_sweep()")
    print("to validate the adaptive protection hypothesis.")
