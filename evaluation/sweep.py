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

    @classmethod
    def from_trials(cls, trials: List[TrialResult]) -> "AggregatedResult":
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
        "adaptive-uep": {
            "codec": "adaptive",
            "use_interpolation": False,
            "sink_blocks": 4,
        },
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
        sink_blocks=mode_cfg["sink_blocks"],
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
