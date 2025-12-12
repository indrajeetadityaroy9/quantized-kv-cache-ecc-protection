"""
Runners Subpackage.

Contains execution environments for experiments:
- modal: Modal GPU orchestration for cloud experiments
- triton_eval: Triton-based ECC evaluation for parallel execution

For local execution, run experiments directly:
    python -m evaluation.experiments.monte_carlo
    python -m evaluation.experiments.architecture
    python -m evaluation.experiments.generation

For Modal execution with Triton ECC:
    modal run modal_runner.py --eval-triton
    modal run modal_runner.py --benchmark-kernels
"""

from .triton_eval import (
    run_single_triton_trial,
    run_triton_ppl_sweep,
    load_llama_model,
    format_ppl_table,
    aggregate_results,
)

__all__ = [
    "run_single_triton_trial",
    "run_triton_ppl_sweep",
    "load_llama_model",
    "format_ppl_table",
    "aggregate_results",
]
