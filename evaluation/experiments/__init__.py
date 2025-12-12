"""
Experiments Subpackage.

Contains specialized experiment implementations:
- monte_carlo: Publication-grade Monte Carlo BER sweep experiments
- architecture: Cross-architecture comparison (GPT-2 vs LLaMA)
- generation: Qualitative text generation demos
- latency: Codec latency benchmarking (Phase 1 - CPU-bound baseline)
"""

from .monte_carlo import (
    run_monte_carlo_experiment,
    format_results_table,
    format_latex_table,
    save_results,
)
from .architecture import (
    analyze_architecture,
    run_architecture_comparison,
    generate_comparison_report,
)
from .generation import (
    run_generation_demo,
    analyze_generation,
    format_generation_results,
    results_to_dict,
    GenerationResult,
    DEFAULT_GENERATION_PROMPTS,
    DEFAULT_GENERATION_MODES,
)
from .latency import (
    CodecBenchmarkConfig,
    CodecBenchmarkResult,
    CodecBenchmarkReport,
    benchmark_codec,
    run_codec_benchmarks,
    run_latency_experiment,
)
from .adaptive_uep import (
    UEPExperimentConfig,
    UEPExperimentResult,
    UEPBoundarySweepResult,
    UEPComparisonResult,
    run_single_uep_trial,
    run_uep_boundary_sweep,
    run_uep_comparison,
    generate_uep_report,
    run_undervolting_stress_test,
)

__all__ = [
    # Monte Carlo
    "run_monte_carlo_experiment",
    "format_results_table",
    "format_latex_table",
    "save_results",
    # Architecture
    "analyze_architecture",
    "run_architecture_comparison",
    "generate_comparison_report",
    # Generation
    "run_generation_demo",
    "analyze_generation",
    "format_generation_results",
    "results_to_dict",
    "GenerationResult",
    "DEFAULT_GENERATION_PROMPTS",
    "DEFAULT_GENERATION_MODES",
    # Latency
    "CodecBenchmarkConfig",
    "CodecBenchmarkResult",
    "CodecBenchmarkReport",
    "benchmark_codec",
    "run_codec_benchmarks",
    "run_latency_experiment",
    # Adaptive UEP
    "UEPExperimentConfig",
    "UEPExperimentResult",
    "UEPBoundarySweepResult",
    "UEPComparisonResult",
    "run_single_uep_trial",
    "run_uep_boundary_sweep",
    "run_uep_comparison",
    "generate_uep_report",
    "run_undervolting_stress_test",
]
