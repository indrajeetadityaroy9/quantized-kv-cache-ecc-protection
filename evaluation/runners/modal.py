"""Modal.com distributed execution runners.

Publication-quality evaluation configurations enforced.
"""
import os
from typing import Dict, Any


LLAMA_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_SEEDS = [42, 101, 997, 1999, 4999]
DEFAULT_SAMPLES = 100


def setup_modal_environment():
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"


def get_gpu_info() -> Dict[str, Any]:
    import torch

    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }


def run_tests_impl(test_file: str = None, verbose: bool = True) -> int:
    import subprocess

    os.chdir("/app")
    setup_modal_environment()

    cmd = ["python", "-m", "pytest"]

    if test_file:
        cmd.append(test_file)
    else:
        cmd.append("tests/")

    if verbose:
        cmd.extend(["-v", "-s", "--tb=short"])

    print("=" * 60)
    print("GPU Test Runner")
    print("=" * 60)

    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['name']}")
    print(f"CUDA Version: {gpu_info['cuda_version']}")
    print(f"Memory: {gpu_info['memory_gb']:.1f} GB")

    print("=" * 60)
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode
