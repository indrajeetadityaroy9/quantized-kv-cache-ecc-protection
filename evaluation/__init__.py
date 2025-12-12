"""
Evaluation Package for Hamming-Protected KV Cache Experiments.

This package provides comprehensive evaluation infrastructure for testing
error-correcting codes (Hamming, Golay) applied to INT4 quantized LLM KV caches.

Subpackages:
    experiments: Experiment implementations (monte_carlo, architecture)
    runners: Execution environments (Modal GPU)

Core Modules:
    constants: Shared configuration (BER levels, cache modes, defaults)
    metrics: Perplexity, KL divergence, token divergence computation
    models: Unified model loading
    sweep: BER sweep runner (single implementation used by all experiments)
"""

# Core utilities
from .metrics import (
    compute_perplexity,
    compute_kl_divergence,
    compute_mean_kl_divergence,
    compute_top5_accuracy,
    compute_catastrophic_rate,
    compute_per_sample_perplexity,
    generate_clean_logits,
    load_wikitext2_test,
)
from .constants import (
    CACHE_MODES,
    CACHE_MODE_ORDER,
    CACHE_MODE_LABELS,
    BER_LEVELS,
    BER_LEVELS_EXTENDED,
    DEFAULT_CONFIG,
    DEFAULT_MODEL,
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

# Experiments
from .experiments.monte_carlo import (
    MonteCarloConfig,
    run_monte_carlo_experiment,
    format_results_table,
    format_latex_table,
    save_results,
)
from .experiments.architecture import (
    ArchitectureInfo,
    ComparisonResult,
    analyze_architecture,
    run_architecture_comparison,
    generate_comparison_report,
    plot_comparison,
)

# Linear Algebra Verification
from .verification import (
    NullSpaceResult,
    OrthogonalityResult,
    RankResult,
    ErrorAmplificationResult,
    VerificationReport,
    verify_null_space_condition,
    verify_subspace_orthogonality,
    verify_basis_independence,
    compute_error_amplification_hamming74,
    compute_error_amplification_hamming84,
    verify_hamming74,
    verify_hamming84,
    verify_golay2412,
    run_all_verifications,
    format_verification_report,
)

__all__ = [
    # Constants
    "CACHE_MODES",
    "CACHE_MODE_ORDER",
    "CACHE_MODE_LABELS",
    "BER_LEVELS",
    "BER_LEVELS_EXTENDED",
    "DEFAULT_CONFIG",
    "DEFAULT_MODEL",
    "get_cache_modes",
    "get_ber_levels",
    "get_seeds",
    # Metrics
    "compute_perplexity",
    "compute_kl_divergence",
    "compute_mean_kl_divergence",
    "compute_top5_accuracy",
    "compute_catastrophic_rate",
    "compute_per_sample_perplexity",
    "generate_clean_logits",
    "load_wikitext2_test",
    # Models
    "load_model",
    # Sweep
    "SweepConfig",
    "TrialResult",
    "AggregatedResult",
    "SweepResults",
    "run_sweep",
    "run_sweep_single_seed",
    # Monte Carlo
    "MonteCarloConfig",
    "run_monte_carlo_experiment",
    "format_results_table",
    "format_latex_table",
    "save_results",
    # Architecture
    "ArchitectureInfo",
    "ComparisonResult",
    "analyze_architecture",
    "run_architecture_comparison",
    "generate_comparison_report",
    "plot_comparison",
    # Linear Algebra Verification
    "NullSpaceResult",
    "OrthogonalityResult",
    "RankResult",
    "ErrorAmplificationResult",
    "VerificationReport",
    "verify_null_space_condition",
    "verify_subspace_orthogonality",
    "verify_basis_independence",
    "compute_error_amplification_hamming74",
    "compute_error_amplification_hamming84",
    "verify_hamming74",
    "verify_hamming84",
    "verify_golay2412",
    "run_all_verifications",
    "format_verification_report",
]
