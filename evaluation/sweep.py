"""
Monte Carlo Parameter Sweep Harness for ECC-Protected KV Cache Evaluation.

This module implements the experimental framework for sweeping over (codec, BER, seed)
configurations with statistical aggregation. It produces publication-ready results
with confidence intervals for perplexity, KL divergence, top-5 accuracy, and
catastrophic failure rates.

Experimental Design:
    - Independent variable: Cache mode (codec), Bit Error Rate (BER)
    - Random variable: Seed (for Monte Carlo estimation)
    - Dependent variables: Perplexity, KL divergence, Top-5 accuracy, etc.

Statistical Methodology:
    - 10 seeds per (codec, BER) for NeurIPS/ICML statistical rigor
    - Bessel's correction for unbiased sample standard deviation
    - 95% confidence intervals using Student's t-distribution
    - t-critical values from lookup table with linear interpolation

Data Structures:
    - SweepConfig: Configures the sweep (modes, BERs, seeds, metrics to compute)
    - TrialResult: Single (codec, BER, seed) experiment result
    - AggregatedResult: Statistics across seeds for one (codec, BER) pair
    - SweepResults: Full sweep results with trial and aggregated data

Key Functions:
    - run_sweep(): Execute full (codec × BER × seed) sweep
    - run_single_trial(): Execute single configuration
    - AggregatedResult.from_trials(): Compute statistics with 95% CIs

Confidence Interval Computation:
    CI_95 = t_critical(df) × std / sqrt(n)
    where df = n - 1, and t_critical comes from Student's t-distribution.

Usage:
    config = SweepConfig.full()  # Full sweep with 10 seeds
    results = run_sweep(model, tokenizer, texts, config)

    for mode in results.aggregated:
        for ber, agg in results.aggregated[mode].items():
            print(f"{mode} @ BER={ber}: PPL={agg.ppl_mean:.2f} ± {agg.ppl_ci95:.2f}")
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
import torch

from .constants import (
    CACHE_MODE_ORDER,
    BER_LEVELS,
    DEFAULT_CONFIG,
    get_cache_modes,
    get_ber_levels,
    get_seeds,
)
from .metrics import (
    compute_perplexity,
    compute_per_sample_perplexity,
    compute_catastrophic_rate,
    compute_top5_accuracy,
    compute_mean_kl_divergence,
)


@dataclass
class SweepConfig:
    cache_modes: List[str] = field(default_factory=lambda: CACHE_MODE_ORDER.copy())
    ber_levels: List[float] = field(default_factory=lambda: BER_LEVELS.copy())

    seeds: List[int] = field(default_factory=lambda: [42])
    aggregate_seeds: bool = True

    max_length: int = DEFAULT_CONFIG["max_length"]
    stride: int = DEFAULT_CONFIG["stride"]

    block_size: int = DEFAULT_CONFIG["block_size"]

    device: str = "cuda"

    compute_kl_divergence: bool = True
    compute_top5: bool = True
    compute_catastrophic: bool = True
    catastrophic_threshold: float = 1000.0

    clean_logits: Optional[List[torch.Tensor]] = None

    enable_timing: bool = False
    profile_transfers: bool = True
    warmup_iterations: int = 3

    backend: str = "triton"

    @classmethod
    def default(cls) -> "SweepConfig":
        return cls()

    @classmethod
    def full(cls) -> "SweepConfig":
        return cls(
            cache_modes=get_cache_modes(),
            ber_levels=get_ber_levels(),
            seeds=get_seeds(),
        )

    @classmethod
    def with_timing(cls) -> "SweepConfig":
        return cls(
            enable_timing=True,
            profile_transfers=True,
            warmup_iterations=3,
        )


@dataclass
class TrialResult:
    cache_mode: str
    ber: float
    seed: int
    perplexity: float
    errors_corrected: int = 0
    errors_detected: int = 0
    total_values: int = 0

    kl_divergence: float = 0.0
    top5_accuracy: float = 1.0
    catastrophic_rate: float = 0.0

    encode_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    throughput_mvalues_sec: float = 0.0
    transfer_overhead_pct: float = 0.0
    is_cpu_bound: bool = True

    # Error correction statistics
    injection_count: int = 0  # Number of injection operations (for tracking)
    correction_rate: float = 0.0  # corrected / (corrected + detected) - fraction recovered
    detection_rate: float = 0.0  # detected / (corrected + detected) - fraction unrecoverable
    silent_corruption_rate: float = 0.0  # Not measurable without ground truth

    @property
    def computed_correction_rate(self) -> float:
        """Compute correction rate: fraction of errors that were corrected."""
        total = self.errors_corrected + self.errors_detected
        if total == 0:
            return 0.0
        return self.errors_corrected / total

    @property
    def computed_detection_rate(self) -> float:
        """Compute detection rate: fraction of errors that were detected but not corrected."""
        total = self.errors_corrected + self.errors_detected
        if total == 0:
            return 0.0
        return self.errors_detected / total


def _t_critical_95(df: int) -> float:
    """Return t-critical value for 95% CI (two-tailed) given degrees of freedom.

    Uses a lookup table for common values, falls back to approximation for large df.
    """
    # t-critical values for 95% CI (two-tailed, alpha=0.05)
    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 50: 2.009, 100: 1.984,
    }
    if df in t_table:
        return t_table[df]
    # For df > 100, use normal approximation (z=1.96)
    if df > 100:
        return 1.96
    # Linear interpolation for intermediate values
    lower = max(k for k in t_table.keys() if k < df)
    upper = min(k for k in t_table.keys() if k > df)
    frac = (df - lower) / (upper - lower)
    return t_table[lower] + frac * (t_table[upper] - t_table[lower])


@dataclass
class AggregatedResult:
    cache_mode: str
    ber: float
    ppl_mean: float
    ppl_std: float
    errors_corrected_mean: float
    errors_detected_mean: float
    total_values: int
    n_trials: int

    kl_divergence_mean: float = 0.0
    kl_divergence_std: float = 0.0
    top5_accuracy_mean: float = 1.0
    top5_accuracy_std: float = 0.0
    catastrophic_rate_mean: float = 0.0
    catastrophic_rate_std: float = 0.0

    encode_time_ms_mean: float = 0.0
    encode_time_ms_std: float = 0.0
    decode_time_ms_mean: float = 0.0
    decode_time_ms_std: float = 0.0
    throughput_mvalues_sec_mean: float = 0.0
    throughput_mvalues_sec_std: float = 0.0
    transfer_overhead_pct_mean: float = 0.0
    transfer_overhead_pct_std: float = 0.0
    is_cpu_bound: bool = True

    # Error correction statistics
    # correction_rate = corrected / (corrected + detected) - fraction of errors recovered
    # detection_rate = detected / (corrected + detected) - fraction of unrecoverable errors
    injection_count_mean: float = 0.0
    correction_rate_mean: float = 0.0
    correction_rate_std: float = 0.0
    detection_rate_mean: float = 0.0
    detection_rate_std: float = 0.0
    silent_corruption_rate_mean: float = 0.0
    silent_corruption_rate_std: float = 0.0

    # 95% Confidence Intervals (half-width: mean ± ci95)
    ppl_ci95: float = 0.0
    kl_divergence_ci95: float = 0.0
    top5_accuracy_ci95: float = 0.0
    catastrophic_rate_ci95: float = 0.0

    @classmethod
    def from_trials(cls, trials: List[TrialResult]) -> "AggregatedResult":
        if not trials:
            raise ValueError("Cannot aggregate empty trial list")

        cache_mode = trials[0].cache_mode
        ber = trials[0].ber

        n = len(trials)
        df = n - 1 if n > 1 else 1
        t_crit = _t_critical_95(df)

        def mean_std_ci(values):
            m = sum(values) / len(values)
            # Bessel's correction: divide by N-1 for unbiased sample std
            if len(values) > 1:
                s = (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5
                # 95% CI half-width: t_critical * std / sqrt(n)
                ci = t_crit * s / (len(values) ** 0.5)
            else:
                s = 0.0
                ci = 0.0
            return m, s, ci

        def mean_std(values):
            m, s, _ = mean_std_ci(values)
            return m, s

        ppls = [t.perplexity for t in trials]
        ppl_mean, ppl_std, ppl_ci95 = mean_std_ci(ppls)

        kls = [t.kl_divergence for t in trials]
        kl_mean, kl_std, kl_ci95 = mean_std_ci(kls)

        top5s = [t.top5_accuracy for t in trials]
        top5_mean, top5_std, top5_ci95 = mean_std_ci(top5s)

        cats = [t.catastrophic_rate for t in trials]
        cat_mean, cat_std, cat_ci95 = mean_std_ci(cats)

        encode_times = [t.encode_time_ms for t in trials]
        encode_mean, encode_std = mean_std(encode_times)

        decode_times = [t.decode_time_ms for t in trials]
        decode_mean, decode_std = mean_std(decode_times)

        throughputs = [t.throughput_mvalues_sec for t in trials]
        throughput_mean, throughput_std = mean_std(throughputs)

        transfer_overheads = [t.transfer_overhead_pct for t in trials]
        transfer_mean, transfer_std = mean_std(transfer_overheads)

        # Error correction statistics
        correction_rates = [t.correction_rate for t in trials]
        correction_rate_mean, correction_rate_std = mean_std(correction_rates)

        detection_rates = [t.detection_rate for t in trials]
        detection_rate_mean, detection_rate_std = mean_std(detection_rates)

        silent_rates = [t.silent_corruption_rate for t in trials]
        silent_rate_mean, silent_rate_std = mean_std(silent_rates)

        injection_counts = [t.injection_count for t in trials]
        injection_count_mean = sum(injection_counts) / len(trials)

        return cls(
            cache_mode=cache_mode,
            ber=ber,
            ppl_mean=ppl_mean,
            ppl_std=ppl_std,
            errors_corrected_mean=sum(t.errors_corrected for t in trials) / len(trials),
            errors_detected_mean=sum(t.errors_detected for t in trials) / len(trials),
            total_values=trials[0].total_values,
            n_trials=len(trials),
            kl_divergence_mean=kl_mean,
            kl_divergence_std=kl_std,
            top5_accuracy_mean=top5_mean,
            top5_accuracy_std=top5_std,
            catastrophic_rate_mean=cat_mean,
            catastrophic_rate_std=cat_std,
            encode_time_ms_mean=encode_mean,
            encode_time_ms_std=encode_std,
            decode_time_ms_mean=decode_mean,
            decode_time_ms_std=decode_std,
            throughput_mvalues_sec_mean=throughput_mean,
            throughput_mvalues_sec_std=throughput_std,
            transfer_overhead_pct_mean=transfer_mean,
            transfer_overhead_pct_std=transfer_std,
            is_cpu_bound=trials[0].is_cpu_bound,
            injection_count_mean=injection_count_mean,
            correction_rate_mean=correction_rate_mean,
            correction_rate_std=correction_rate_std,
            detection_rate_mean=detection_rate_mean,
            detection_rate_std=detection_rate_std,
            silent_corruption_rate_mean=silent_rate_mean,
            silent_corruption_rate_std=silent_rate_std,
            # 95% Confidence Intervals
            ppl_ci95=ppl_ci95,
            kl_divergence_ci95=kl_ci95,
            top5_accuracy_ci95=top5_ci95,
            catastrophic_rate_ci95=cat_ci95,
        )


@dataclass
class SweepResults:
    config: SweepConfig
    trials: List[TrialResult] = field(default_factory=list)
    aggregated: Dict[str, Dict[float, AggregatedResult]] = field(default_factory=dict)

    def get_aggregated(self, cache_mode: str, ber: float) -> Optional[AggregatedResult]:
        return self.aggregated.get(cache_mode, {}).get(ber)

    def get_trials(
        self, cache_mode: str = None, ber: float = None, seed: int = None
    ) -> List[TrialResult]:
        result = self.trials
        if cache_mode is not None:
            result = [t for t in result if t.cache_mode == cache_mode]
        if ber is not None:
            result = [t for t in result if t.ber == ber]
        if seed is not None:
            result = [t for t in result if t.seed == seed]
        return result


def run_single_trial(
    model,
    tokenizer,
    texts: List[str],
    cache_mode: str,
    ber: float,
    seed: int,
    config: SweepConfig,
) -> TrialResult:
    return _run_single_trial_triton(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        cache_mode=cache_mode,
        ber=ber,
        seed=seed,
        config=config,
    )


def run_sweep(
    model,
    tokenizer,
    texts: List[str],
    config: SweepConfig = None,
    progress_callback: Callable[[str, int, int], None] = None,
) -> SweepResults:
    if config is None:
        config = SweepConfig.default()

    results = SweepResults(config=config)

    total = len(config.cache_modes) * len(config.ber_levels) * len(config.seeds)
    current = 0

    for cache_mode in config.cache_modes:
        for ber in config.ber_levels:
            trials_for_config = []

            for seed in config.seeds:
                if progress_callback:
                    progress_callback(
                        f"{cache_mode} @ BER={ber:.0e} seed={seed}",
                        current,
                        total,
                    )

                trial = run_single_trial(
                    model=model,
                    tokenizer=tokenizer,
                    texts=texts,
                    cache_mode=cache_mode,
                    ber=ber,
                    seed=seed,
                    config=config,
                )

                results.trials.append(trial)
                trials_for_config.append(trial)
                current += 1

            if config.aggregate_seeds and trials_for_config:
                if cache_mode not in results.aggregated:
                    results.aggregated[cache_mode] = {}
                results.aggregated[cache_mode][ber] = AggregatedResult.from_trials(
                    trials_for_config
                )

    return results


def run_sweep_single_seed(
    model,
    tokenizer,
    texts: List[str],
    config: SweepConfig = None,
    seed: int = 42,
    progress_callback: Callable[[str, int, int], None] = None,
) -> Dict[str, Dict[float, TrialResult]]:
    if config is None:
        config = SweepConfig.default()

    config.seeds = [seed]
    config.aggregate_seeds = False

    full_results = run_sweep(model, tokenizer, texts, config, progress_callback)

    results: Dict[str, Dict[float, TrialResult]] = {}
    for trial in full_results.trials:
        if trial.cache_mode not in results:
            results[trial.cache_mode] = {}
        results[trial.cache_mode][trial.ber] = trial

    return results


def _run_single_trial_triton(
    model,
    tokenizer,
    texts: List[str],
    cache_mode: str,
    ber: float,
    seed: int,
    config: SweepConfig,
) -> TrialResult:
    import torch
    from kv_cache.ecc_shim import (
        ECCShimConfig,
        patch_model_with_ecc_attention,
        reset_ecc_cache,
        get_ecc_stats,
    )
    from evaluation.metrics import (
        compute_top5_accuracy,
        compute_per_sample_perplexity,
        compute_catastrophic_rate,
        compute_mean_kl_divergence,
    )

    device = "cuda" if config.device == "auto" else config.device

    if device != "cuda":
        raise RuntimeError("Triton backend requires CUDA.")

    mode_cfg_map = {
        "fp16": {"codec": "fp16", "use_interpolation": False},
        "fp8": {"codec": "fp8", "use_interpolation": False},
        "int4": {"codec": "int4", "use_interpolation": False},
        "int4-hamming": {"codec": "hamming74", "use_interpolation": False},
        "int4-hamming84": {"codec": "hamming84", "use_interpolation": False},
        "int4-hamming84-interp": {"codec": "hamming84", "use_interpolation": True},
        "int12-golay": {"codec": "golay", "use_interpolation": False},
    }
    if cache_mode not in mode_cfg_map:
        raise ValueError(f"Unsupported cache_mode for Triton backend: {cache_mode}")

    mode_cfg = mode_cfg_map[cache_mode]

    ecc_config = ECCShimConfig(
        codec=mode_cfg["codec"],
        ber=ber,
        inject_errors=(ber > 0),
        seed=seed,
        num_blocks=2048,
        block_size=16,
        use_interpolation=mode_cfg["use_interpolation"],
    )

    model_device = next(model.parameters()).device
    if str(model_device) != str(device):
        model = model.to(device)

    total_loss = 0.0
    total_tokens = 0

    kl_div = 0.0
    top5_acc = 1.0
    cat_rate = 0.0

    with torch.no_grad():
        with patch_model_with_ecc_attention(
            model, ecc_config, num_blocks=ecc_config.num_blocks
        ):
            for text in texts:
                if not text.strip():
                    continue

                reset_ecc_cache(model)

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=config.max_length,
                )
                input_ids = inputs["input_ids"].to(device)
                seq_len = input_ids.size(1)
                if seq_len < 2:
                    continue

                pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

                outputs = model(
                    input_ids,
                    labels=input_ids,
                    use_cache=True,
                    pad_token_id=pad_id,
                )
                loss = outputs.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                total_loss += loss.item() * seq_len
                total_tokens += seq_len

            ppl = (
                float("inf")
                if total_tokens == 0
                else float(torch.exp(torch.tensor(total_loss / total_tokens)).item())
            )

            if config.compute_kl_divergence and config.clean_logits is not None:
                kl_div = compute_mean_kl_divergence(
                    model,
                    tokenizer,
                    texts,
                    config.clean_logits,
                    max_length=config.max_length,
                    device=device,
                )

            if config.compute_top5:
                top5_acc = compute_top5_accuracy(
                    model,
                    tokenizer,
                    texts,
                    max_length=config.max_length,
                    device=device,
                )

            if config.compute_catastrophic:
                per_sample_ppls = compute_per_sample_perplexity(
                    model,
                    tokenizer,
                    texts,
                    max_length=config.max_length,
                    stride=config.stride,
                    device=device,
                )
                cat_rate = compute_catastrophic_rate(
                    per_sample_ppls,
                    threshold=config.catastrophic_threshold,
                )

            stats = get_ecc_stats(model)
            errors_corrected = stats.get("errors_corrected", 0)
            errors_detected = stats.get("errors_detected", 0)
            total_values = stats.get("total_values", 0)
            injection_count = stats.get("injection_count", 0)

    # Compute error correction statistics
    # correction_rate = fraction of detected errors that were single-bit and corrected
    # detection_rate = fraction of detected errors that were double-bit (detected but not corrected)
    # Note: silent corruption (3+ bit errors) cannot be measured without ground truth
    total_error_events = errors_corrected + errors_detected
    if total_error_events > 0:
        correction_rate = errors_corrected / total_error_events
        detection_rate = errors_detected / total_error_events
        silent_corruption_rate = 0.0  # Not measurable without ground truth
    else:
        correction_rate = 0.0
        detection_rate = 0.0
        silent_corruption_rate = 0.0

    return TrialResult(
        cache_mode=cache_mode,
        ber=ber,
        seed=seed,
        perplexity=ppl,
        errors_corrected=errors_corrected,
        errors_detected=errors_detected,
        total_values=total_values,
        kl_divergence=kl_div,
        top5_accuracy=top5_acc,
        catastrophic_rate=cat_rate,
        encode_time_ms=0.0,
        decode_time_ms=0.0,
        throughput_mvalues_sec=0.0,
        transfer_overhead_pct=0.0,
        is_cpu_bound=False,
        injection_count=injection_count,
        correction_rate=correction_rate,
        detection_rate=detection_rate,
        silent_corruption_rate=silent_corruption_rate,
    )
