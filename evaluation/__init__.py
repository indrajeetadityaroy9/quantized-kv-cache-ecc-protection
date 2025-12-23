"""Evaluation framework for ECC-protected KV caches.

Uses vLLM C++ backend for optimized paged attention with integrated
ECC encode/decode.

Core modules:
- metrics: Perplexity, statistics, generation quality, downstream tasks
- sweep: BER sweep configuration and execution (vLLM backend)
- models: Model loading utilities
- constants: Cache modes, BER levels, configuration
"""
from .metrics import (
    # Perplexity metrics
    compute_perplexity,
    compute_perplexity_torchmetrics,
    compute_per_sample_perplexity,
    compute_catastrophic_rate,
    compute_kl_divergence,
    compute_mean_kl_divergence,
    compute_top5_accuracy,
    generate_clean_logits,
    load_wikitext2_test,
    load_c4_validation,
    load_ptb_test,
    load_dataset_by_name,
    # Statistical analysis
    ConfidenceInterval,
    HypothesisTestResult,
    compute_confidence_interval,
    cohens_d,
    interpret_effect_size,
    paired_t_test,
    independent_t_test,
    holm_bonferroni_correction,
    bootstrap_ci,
    # Generation metrics
    compute_bleu,
    compute_sentence_bleu,
    compute_rouge_l,
    compute_generation_metrics,
    # Downstream tasks
    TaskResult,
    MMLU_SUBJECTS,
    MMLU_STEM_SUBJECTS,
    load_mmlu_subset,
    evaluate_mmlu,
    load_hellaswag_subset,
    evaluate_hellaswag,
    run_downstream_evaluation,
)
from .constants import (
    CACHE_MODES,
    CACHE_MODE_ORDER,
    CACHE_MODE_LABELS,
    BER_LEVELS,
    DEFAULT_CONFIG,
    MODELS,
    get_cache_modes,
    get_ber_levels,
    get_seeds,
)
from .models import load_model
from .sweep import (
    SweepConfig,
    TrialResult,
    AggregatedResult,
    SweepResults,
    run_sweep,
    run_sweep_single_seed,
)

__all__ = [
    # Perplexity metrics
    "compute_perplexity",
    "compute_perplexity_torchmetrics",
    "compute_per_sample_perplexity",
    "compute_catastrophic_rate",
    "compute_kl_divergence",
    "compute_mean_kl_divergence",
    "compute_top5_accuracy",
    "generate_clean_logits",
    "load_wikitext2_test",
    "load_c4_validation",
    "load_ptb_test",
    "load_dataset_by_name",
    # Statistical analysis
    "ConfidenceInterval",
    "HypothesisTestResult",
    "compute_confidence_interval",
    "cohens_d",
    "interpret_effect_size",
    "paired_t_test",
    "independent_t_test",
    "holm_bonferroni_correction",
    "bootstrap_ci",
    # Generation metrics
    "compute_bleu",
    "compute_sentence_bleu",
    "compute_rouge_l",
    "compute_generation_metrics",
    # Downstream tasks
    "TaskResult",
    "MMLU_SUBJECTS",
    "MMLU_STEM_SUBJECTS",
    "load_mmlu_subset",
    "evaluate_mmlu",
    "load_hellaswag_subset",
    "evaluate_hellaswag",
    "run_downstream_evaluation",
    # Constants
    "CACHE_MODES",
    "CACHE_MODE_ORDER",
    "CACHE_MODE_LABELS",
    "BER_LEVELS",
    "DEFAULT_CONFIG",
    "MODELS",
    "get_cache_modes",
    "get_ber_levels",
    "get_seeds",
    # Models
    "load_model",
    # Sweep (vLLM backend)
    "SweepConfig",
    "TrialResult",
    "AggregatedResult",
    "SweepResults",
    "run_sweep",
    "run_sweep_single_seed",
]
