"""Sweep configuration and results for BER evaluation using vLLM backend."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import torch

from .constants import (
    CACHE_MODE_ORDER,
    BER_LEVELS,
    DEFAULT_CONFIG,
    get_cache_modes,
    get_ber_levels,
    get_seeds,
)


@dataclass
class SweepConfig:
    """Configuration for BER sweep evaluation."""
    cache_modes: List[str] = field(default_factory=lambda: CACHE_MODE_ORDER.copy())
    ber_levels: List[float] = field(default_factory=lambda: BER_LEVELS.copy())

    seeds: List[int] = field(default_factory=lambda: [42])
    aggregate_seeds: bool = True

    max_length: int = DEFAULT_CONFIG["max_length"]
    stride: int = DEFAULT_CONFIG["stride"]

    max_samples: int = DEFAULT_CONFIG["max_samples"]

    block_size: int = DEFAULT_CONFIG["block_size"]

    device: str = "cuda"

    compute_kl_divergence: bool = False  # Not supported in vLLM backend
    compute_top5: bool = False  # Not supported in vLLM backend
    compute_catastrophic: bool = True
    catastrophic_threshold: float = 1000.0

    clean_logits: Optional[List[torch.Tensor]] = None

    enable_timing: bool = False
    profile_transfers: bool = True
    warmup_iterations: int = 3

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
    """Result from a single trial (cache_mode + BER + seed)."""
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


@dataclass
class AggregatedResult:
    """Aggregated results across multiple seeds for a (cache_mode, BER) pair."""
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

    @classmethod
    def from_trials(cls, trials: List["TrialResult"]) -> "AggregatedResult":
        if not trials:
            raise ValueError("Cannot aggregate empty trial list")

        cache_mode = trials[0].cache_mode
        ber = trials[0].ber

        def mean_std(values):
            m = sum(values) / len(values)
            s = (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5
            return m, s

        ppls = [t.perplexity for t in trials]
        ppl_mean, ppl_std = mean_std(ppls)

        kls = [t.kl_divergence for t in trials]
        kl_mean, kl_std = mean_std(kls)

        top5s = [t.top5_accuracy for t in trials]
        top5_mean, top5_std = mean_std(top5s)

        cats = [t.catastrophic_rate for t in trials]
        cat_mean, cat_std = mean_std(cats)

        encode_times = [t.encode_time_ms for t in trials]
        encode_mean, encode_std = mean_std(encode_times)

        decode_times = [t.decode_time_ms for t in trials]
        decode_mean, decode_std = mean_std(decode_times)

        throughputs = [t.throughput_mvalues_sec for t in trials]
        throughput_mean, throughput_std = mean_std(throughputs)

        transfer_overheads = [t.transfer_overhead_pct for t in trials]
        transfer_mean, transfer_std = mean_std(transfer_overheads)

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
        )


@dataclass
class SweepResults:
    """Container for all sweep results."""
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


def _run_vllm_trial(
    model_name: str,
    texts: List[str],
    cache_mode: str,
    ber: float,
    seed: int,
    config: SweepConfig,
) -> TrialResult:
    """Run a single trial using vLLM backend.

    Args:
        model_name: HuggingFace model name or path.
        texts: List of text samples.
        cache_mode: Cache mode (fp16, int4, int4-hamming, etc.).
        ber: Bit error rate.
        seed: Random seed.
        config: SweepConfig with parameters.

    Returns:
        TrialResult with perplexity and error stats.
    """
    from evaluation.runners.vllm_runner import VLLMEvaluationRunner

    torch.manual_seed(seed)

    runner = VLLMEvaluationRunner(
        model_name=model_name,
        cache_mode=cache_mode,
        device=config.device,
    )

    # Reset stats before computation
    runner.reset_stats()

    # Compute perplexity (KV cache populated during inference)
    ppl = runner.compute_perplexity(
        texts,
        max_length=config.max_length,
        stride=config.stride,
    )

    # Inject errors if BER > 0 (only for ECC modes, not fp16)
    if ber > 0 and cache_mode != "fp16":
        runner.inject_errors(ber, seed)
        # Recompute with corrupted cache
        ppl = runner.compute_perplexity(
            texts,
            max_length=config.max_length,
            stride=config.stride,
        )

    stats = runner.get_error_stats()

    return TrialResult(
        cache_mode=cache_mode,
        ber=ber,
        seed=seed,
        perplexity=ppl,
        errors_corrected=stats.get("total_corrected", 0),
        errors_detected=stats.get("total_uncorrectable", 0),
        total_values=stats.get("golay_total_triplets", 0) + stats.get("hamming_total_values", 0),
    )


def run_sweep(
    model_name: str,
    texts: List[str],
    config: SweepConfig = None,
    progress_callback: Callable[[str, int, int], None] = None,
) -> SweepResults:
    """Run BER sweep using vLLM C++ backend.

    This function uses vLLM's offline inference API for evaluation,
    which provides better performance through fused CUDA kernels and
    proper paged attention with integrated ECC encode/decode.

    Args:
        model_name: HuggingFace model name or path (e.g., "gpt2", "meta-llama/Llama-3.1-8B").
        texts: List of text samples for perplexity evaluation.
        config: SweepConfig with parameters.
        progress_callback: Optional callback(message, current, total) for progress reporting.

    Returns:
        SweepResults with all trial results and aggregations.

    Example:
        >>> config = SweepConfig(
        ...     cache_modes=["fp16", "int4-hamming84", "int4-golay-hybrid"],
        ...     ber_levels=[0, 1e-6, 1e-5],
        ...     seeds=[42, 101],
        ... )
        >>> results = run_sweep("gpt2", texts, config)
    """
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
                        f"vLLM: {cache_mode} @ BER={ber:.0e} seed={seed}",
                        current,
                        total,
                    )

                trial = _run_vllm_trial(
                    model_name=model_name,
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
    model_name: str,
    texts: List[str],
    config: SweepConfig = None,
    seed: int = 42,
    progress_callback: Callable[[str, int, int], None] = None,
) -> Dict[str, Dict[float, TrialResult]]:
    """Run sweep with a single seed, returning results indexed by mode and BER.

    Args:
        model_name: HuggingFace model name or path.
        texts: List of text samples.
        config: SweepConfig with parameters.
        seed: Random seed to use.
        progress_callback: Optional callback(message, current, total).

    Returns:
        Dict mapping cache_mode -> ber -> TrialResult.
    """
    if config is None:
        config = SweepConfig.default()

    config.seeds = [seed]
    config.aggregate_seeds = False

    full_results = run_sweep(model_name, texts, config, progress_callback)

    results: Dict[str, Dict[float, TrialResult]] = {}
    for trial in full_results.trials:
        if trial.cache_mode not in results:
            results[trial.cache_mode] = {}
        results[trial.cache_mode][trial.ber] = trial

    return results
