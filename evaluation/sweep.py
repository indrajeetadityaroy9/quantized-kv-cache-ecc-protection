"""
Unified BER Sweep Runner.

Single implementation of the BER sweep logic used by all experiments
and runners. Eliminates code duplication across experiment_runner,
architecture_comparison, and modal_test_runner.
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
    """Configuration for BER sweep experiments."""

    # What to sweep
    cache_modes: List[str] = field(default_factory=lambda: CACHE_MODE_ORDER.copy())
    ber_levels: List[float] = field(default_factory=lambda: BER_LEVELS.copy())

    # Monte Carlo settings
    seeds: List[int] = field(default_factory=lambda: [42])
    aggregate_seeds: bool = True  # Whether to compute mean±std across seeds

    # Perplexity computation
    max_length: int = DEFAULT_CONFIG["max_length"]
    stride: int = DEFAULT_CONFIG["stride"]

    # Quantization
    block_size: int = DEFAULT_CONFIG["block_size"]

    # Device
    device: str = "auto"

    # Advanced metrics (all enabled by default for proper evaluation)
    compute_kl_divergence: bool = True
    compute_top5: bool = True
    compute_catastrophic: bool = True
    catastrophic_threshold: float = 1000.0  # PPL threshold for catastrophic failure

    # Clean baselines (set by benchmark runner before sweep)
    clean_logits: Optional[List[torch.Tensor]] = None

    # Timing instrumentation (Phase 1 benchmarking)
    enable_timing: bool = False
    profile_transfers: bool = True  # Include CPU/GPU transfer profiling
    warmup_iterations: int = 3  # Warmup before timing measurements

    # Backend is GPU Triton/vLLM only.
    backend: str = "triton"

    @classmethod
    def default(cls) -> "SweepConfig":
        """Create config with default settings."""
        return cls()

    @classmethod
    def full(cls) -> "SweepConfig":
        """Create config for full Monte Carlo sweep."""
        return cls(
            cache_modes=get_cache_modes(),
            ber_levels=get_ber_levels(),
            seeds=get_seeds(),
        )

    @classmethod
    def with_timing(cls) -> "SweepConfig":
        """Create config with timing instrumentation enabled."""
        return cls(
            enable_timing=True,
            profile_transfers=True,
            warmup_iterations=3,
        )


@dataclass
class TrialResult:
    """Result from a single trial (one seed, one BER, one mode)."""

    cache_mode: str
    ber: float
    seed: int
    perplexity: float
    errors_corrected: int = 0
    errors_detected: int = 0
    total_values: int = 0

    # New metrics (replacing token_divergence)
    kl_divergence: float = 0.0       # KL(clean || corrupted), lower = better
    top5_accuracy: float = 1.0       # Target in top-5 predictions, higher = better
    catastrophic_rate: float = 0.0   # % samples with PPL > threshold, lower = better

    # Timing metrics (Phase 1 - CPU-bound baseline)
    encode_time_ms: float = 0.0              # Mean encode time per operation
    decode_time_ms: float = 0.0              # Mean decode time per operation
    throughput_mvalues_sec: float = 0.0      # Millions of values per second
    transfer_overhead_pct: float = 0.0       # % time spent in CPU/GPU transfers
    is_cpu_bound: bool = True                # Flag: True for Phase 1 (CPU), False for Phase 3 (Triton)


@dataclass
class AggregatedResult:
    """Aggregated result across multiple seeds."""

    cache_mode: str
    ber: float
    ppl_mean: float
    ppl_std: float
    errors_corrected_mean: float
    errors_detected_mean: float
    total_values: int
    n_trials: int

    # New metrics (replacing token_divergence)
    kl_divergence_mean: float = 0.0
    kl_divergence_std: float = 0.0
    top5_accuracy_mean: float = 1.0
    top5_accuracy_std: float = 0.0
    catastrophic_rate_mean: float = 0.0
    catastrophic_rate_std: float = 0.0

    # Timing metrics (Phase 1 - CPU-bound baseline)
    encode_time_ms_mean: float = 0.0
    encode_time_ms_std: float = 0.0
    decode_time_ms_mean: float = 0.0
    decode_time_ms_std: float = 0.0
    throughput_mvalues_sec_mean: float = 0.0
    throughput_mvalues_sec_std: float = 0.0
    transfer_overhead_pct_mean: float = 0.0
    transfer_overhead_pct_std: float = 0.0
    is_cpu_bound: bool = True  # Flag: True for Phase 1 (CPU), False for Phase 3 (Triton)

    @classmethod
    def from_trials(cls, trials: List[TrialResult]) -> "AggregatedResult":
        """Aggregate multiple trial results."""
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

        # Timing metrics aggregation
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
    """Complete results from a BER sweep."""

    config: SweepConfig
    trials: List[TrialResult] = field(default_factory=list)
    aggregated: Dict[str, Dict[float, AggregatedResult]] = field(default_factory=dict)

    def get_aggregated(self, cache_mode: str, ber: float) -> Optional[AggregatedResult]:
        """Get aggregated result for a specific mode and BER."""
        return self.aggregated.get(cache_mode, {}).get(ber)

    def get_trials(
        self, cache_mode: str = None, ber: float = None, seed: int = None
    ) -> List[TrialResult]:
        """Filter trials by criteria."""
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
    """
    Run a single trial with specified parameters.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        texts: List of text samples for perplexity computation
        cache_mode: Protection mode (e.g., "int4-hamming")
        ber: Bit error rate
        seed: Random seed for reproducibility
        config: Sweep configuration

    Returns:
        TrialResult with perplexity, KL divergence, top-5 accuracy, timing, and error statistics
    """
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
    """
    Run a complete BER sweep across all modes, BER levels, and seeds.

    This is the single implementation used by all experiments and runners.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        texts: List of text samples for perplexity computation
        config: Sweep configuration (defaults to SweepConfig.default())
        progress_callback: Optional callback(message, current, total) for progress

    Returns:
        SweepResults with all trials and aggregated statistics

    Example:
        >>> config = SweepConfig.quick()
        >>> results = run_sweep(model, tokenizer, texts, config)
        >>> print(results.aggregated["int4-hamming"][0.01].ppl_mean)
    """
    if config is None:
        config = SweepConfig.default()

    results = SweepResults(config=config)

    # Calculate total iterations for progress
    total = len(config.cache_modes) * len(config.ber_levels) * len(config.seeds)
    current = 0

    # Run all trials
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

            # Aggregate across seeds
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
    """
    Run a BER sweep with a single seed (no aggregation).

    Simpler interface for quick experiments without Monte Carlo.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        texts: List of text samples
        config: Sweep configuration
        seed: Random seed
        progress_callback: Optional progress callback

    Returns:
        Dict mapping cache_mode -> ber -> TrialResult
    """
    if config is None:
        config = SweepConfig.default()

    # Override seeds to single seed
    config.seeds = [seed]
    config.aggregate_seeds = False

    full_results = run_sweep(model, tokenizer, texts, config, progress_callback)

    # Convert to simpler structure
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
    """
    GPU-native trial using the vLLM/Triton ECC shim.

    Uses ECCPagedAttentionShim to quantize, encode, inject faults, and decode
    within attention. Requires CUDA.
    """
    import torch
    from vllm_kernels.shim import (
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

    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    if device == "cpu" or not torch.cuda.is_available():
        raise RuntimeError("Triton backend requires CUDA.")

    # Map cache_mode to shim codec parameters
    mode_cfg_map = {
        "fp16": {"codec": "fp16", "use_interpolation": False, "sink_blocks": 0},
        "int4": {"codec": "int4", "use_interpolation": False, "sink_blocks": 0},
        "int4-hamming": {"codec": "hamming74", "use_interpolation": False, "sink_blocks": 0},
        "int4-hamming84": {"codec": "hamming84", "use_interpolation": False, "sink_blocks": 0},
        "int4-hamming84-interp": {"codec": "hamming84", "use_interpolation": True, "sink_blocks": 0},
        "int12-golay": {"codec": "golay", "use_interpolation": False, "sink_blocks": 0},
        "adaptive": {"codec": "adaptive", "use_interpolation": False, "sink_blocks": 4},
        "adaptive-uep": {"codec": "adaptive", "use_interpolation": False, "sink_blocks": 4},
    }
    if cache_mode not in mode_cfg_map:
        raise ValueError(f"Unsupported cache_mode for Triton backend: {cache_mode}")

    mode_cfg = mode_cfg_map[cache_mode]

    # Shim config
    ecc_config = ECCShimConfig(
        codec=mode_cfg["codec"],
        ber=ber,
        inject_errors=(ber > 0),
        seed=seed,
        num_blocks=2048,
        block_size=16,
        sink_blocks=mode_cfg["sink_blocks"],
        use_interpolation=mode_cfg["use_interpolation"],
    )

    model_device = next(model.parameters()).device
    if str(model_device) != str(device):
        model = model.to(device)

    total_loss = 0.0
    total_tokens = 0

    # Advanced metrics placeholders
    kl_div = 0.0
    top5_acc = 1.0
    cat_rate = 0.0

    with torch.no_grad():
        with patch_model_with_ecc_attention(model, ecc_config, num_blocks=ecc_config.num_blocks):
            # Compute perplexity over documents (reset cache per doc)
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

            ppl = float("inf") if total_tokens == 0 else float(torch.exp(torch.tensor(total_loss / total_tokens)).item())

            # KL divergence if provided clean logits
            if config.compute_kl_divergence and config.clean_logits is not None:
                kl_div = compute_mean_kl_divergence(
                    model,
                    tokenizer,
                    texts,
                    config.clean_logits,
                    max_length=config.max_length,
                    device=device,
                )

            # Top-5 accuracy
            if config.compute_top5:
                top5_acc = compute_top5_accuracy(
                    model,
                    tokenizer,
                    texts,
                    max_length=config.max_length,
                    device=device,
                )

            # Catastrophic rate
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

            # ECC stats from shim
            stats = get_ecc_stats(model)
            errors_corrected = stats.get("errors_corrected", 0)
            errors_detected = stats.get("errors_detected", 0)
            total_values = stats.get("total_values", 0)

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
    )
