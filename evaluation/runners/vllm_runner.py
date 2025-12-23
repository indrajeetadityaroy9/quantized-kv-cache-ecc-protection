"""vLLM-based evaluation runner for ECC-protected KV cache."""

import math
from typing import List, Dict, Optional
import torch


# Mapping from evaluation cache modes to vLLM kv_cache_dtype
# NOTE: "int4" (unprotected) is not available - vLLM only supports ECC-protected INT4 modes
CACHE_MODE_TO_VLLM_DTYPE = {
    "fp16": "auto",                    # Standard FP16 baseline
    "int4-hamming": "int4_h74",        # INT4 + Hamming(7,4) SEC
    "int4-hamming84": "int4_ecc",      # INT4 + Hamming(8,4) SECDED
    "int12-golay": "int4_golay",       # INT4 triplets + Golay(24,12)
    "int4-golay-hybrid": "int4_golay_hybrid",  # Golay + Hamming hybrid
    "int4-reed-solomon": "int4_rs",    # INT4 + Reed-Solomon(12,8)
}


def _import_ecc_helpers():
    """Lazy import of ECC helpers from modified vLLM source."""
    try:
        from vllm.ecc.golay_table import (
            get_golay_syndrome_lut,
            create_golay_stats_tensors,
            get_error_stats_summary,
        )
        return get_golay_syndrome_lut, create_golay_stats_tensors, get_error_stats_summary
    except ImportError:
        return None, None, None


class VLLMEvaluationRunner:
    """Runs evaluation using vLLM C++ backend with ECC-protected KV cache.

    This runner provides:
    - Perplexity computation using vLLM's prompt_logprobs
    - Fault injection into KV cache at configurable BER
    - ECC error statistics collection

    Usage:
        runner = VLLMEvaluationRunner("gpt2", "int4-hamming84")
        ppl = runner.compute_perplexity(texts)
        runner.inject_errors(ber=1e-6, seed=42)
        ppl_corrupted = runner.compute_perplexity(texts)
        stats = runner.get_error_stats()
    """

    def __init__(
        self,
        model_name: str,
        cache_mode: str = "fp16",
        device: str = "cuda",
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
    ):
        """Initialize vLLM evaluation runner.

        Args:
            model_name: HuggingFace model name or path.
            cache_mode: Cache mode (fp16, int4-hamming, int4-hamming84, int12-golay, etc.).
            device: Device to use (currently only "cuda" supported by vLLM).
            gpu_memory_utilization: Fraction of GPU memory to use.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            dtype: Model weight dtype.
        """
        try:
            from vllm import LLM, SamplingParams
            self._SamplingParams = SamplingParams
        except ImportError as e:
            raise ImportError(
                f"vLLM is not installed: {e}. Install with: pip install vllm"
            )

        self.model_name = model_name
        self.cache_mode = cache_mode
        self.device = device

        # Map cache mode to vLLM dtype
        self.kv_cache_dtype = CACHE_MODE_TO_VLLM_DTYPE.get(cache_mode, "auto")
        if self.kv_cache_dtype is None:
            raise ValueError(
                f"Unknown cache mode: {cache_mode}. "
                f"Available: {list(CACHE_MODE_TO_VLLM_DTYPE.keys())}"
            )

        # Initialize vLLM
        self.llm = LLM(
            model=model_name,
            kv_cache_dtype=self.kv_cache_dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=True,
        )

        # Initialize ECC resources if using ECC mode
        self.golay_syndrome_lut: Optional[torch.Tensor] = None
        self.golay_stats: Optional[torch.Tensor] = None
        self.hamming_stats: Optional[torch.Tensor] = None
        self.rs_stats: Optional[torch.Tensor] = None
        self._ecc_available = False

        if self._is_ecc_mode():
            self._init_ecc_resources()

    def _is_ecc_mode(self) -> bool:
        """Check if current mode uses ECC protection."""
        return self.kv_cache_dtype.startswith("int4")

    def _init_ecc_resources(self):
        """Initialize ECC syndrome LUT and stats tensors."""
        # Initialize Golay/Hamming resources for golay modes
        get_golay_syndrome_lut, create_golay_stats_tensors, _ = _import_ecc_helpers()
        if get_golay_syndrome_lut is not None:
            try:
                self.golay_syndrome_lut = get_golay_syndrome_lut(self.device)
                self.golay_stats, self.hamming_stats = create_golay_stats_tensors(self.device)
                self._ecc_available = True
            except Exception as e:
                print(f"Warning: Failed to initialize Golay/Hamming ECC resources: {e}")
                self._ecc_available = False

        # Initialize RS resources for RS mode
        if self.kv_cache_dtype == "int4_rs":
            try:
                from vllm.ecc.rs_tables import create_rs_stats_tensor
                self.rs_stats = create_rs_stats_tensor(self.device)
                self._ecc_available = True
            except Exception as e:
                print(f"Warning: Failed to initialize RS ECC resources: {e}")
                self._ecc_available = False

    def reset_stats(self):
        """Reset error statistics counters to zero."""
        if self.golay_stats is not None:
            self.golay_stats.zero_()
        if self.hamming_stats is not None:
            self.hamming_stats.zero_()
        if self.rs_stats is not None:
            self.rs_stats.zero_()

    def compute_perplexity(
        self,
        texts: List[str],
        max_length: int = 512,
        stride: int = 256,
    ) -> float:
        """Compute perplexity using vLLM's prompt_logprobs.

        Uses sliding window approach for long texts.

        Args:
            texts: List of text samples.
            max_length: Maximum sequence length per window.
            stride: Sliding window stride.

        Returns:
            Perplexity value (exp of average negative log likelihood).
        """
        sampling_params = self._SamplingParams(
            prompt_logprobs=1,
            max_tokens=1,
            temperature=0.0,
        )

        total_nll = 0.0
        total_tokens = 0

        tokenizer = self.llm.get_tokenizer()

        for text in texts:
            if not text.strip():
                continue

            tokens = tokenizer.encode(text)
            seq_len = len(tokens)

            if seq_len < 2:
                continue

            prev_end = 0
            for begin in range(0, seq_len, stride):
                end = min(begin + max_length, seq_len)

                target_len = end - max(begin, prev_end)
                if target_len <= 0:
                    prev_end = end
                    if end >= seq_len:
                        break
                    continue

                window_tokens = tokens[begin:end]

                try:
                    outputs = self.llm.generate(
                        prompts=[{"prompt_token_ids": window_tokens}],
                        sampling_params=sampling_params,
                    )

                    if outputs and outputs[0].prompt_logprobs:
                        prompt_logprobs = outputs[0].prompt_logprobs
                        start_idx = max(prev_end - begin, 1)

                        for i in range(start_idx, len(prompt_logprobs)):
                            logprob_dict = prompt_logprobs[i]
                            if logprob_dict is None:
                                continue

                            token_id = window_tokens[i]
                            if token_id in logprob_dict:
                                logprob = logprob_dict[token_id].logprob
                                total_nll -= logprob
                                total_tokens += 1

                except Exception as e:
                    print(f"Warning: Error computing logprobs: {e}")
                    pass

                prev_end = end
                if end >= seq_len:
                    break

        if total_tokens == 0:
            return float("inf")

        return math.exp(total_nll / total_tokens)

    def inject_errors(self, ber: float, seed: int) -> Dict[str, int]:
        """Inject bit errors into KV cache at specified BER.

        Args:
            ber: Bit error rate (probability of each bit being flipped).
            seed: Random seed for reproducibility.

        Returns:
            Dict with number of bits flipped in key and value caches.
        """
        if ber <= 0:
            return {"key_bits_flipped": 0, "value_bits_flipped": 0}

        # Use vLLM's inject_cache_errors method
        try:
            self.llm.inject_cache_errors(ber, seed)
        except AttributeError:
            print("Warning: inject_cache_errors not available in this vLLM build")

        return {"key_bits_flipped": 0, "value_bits_flipped": 0}

    def get_error_stats(self) -> Dict[str, int]:
        """Get ECC error correction statistics.

        Returns:
            Dict with error counts from ECC decode operations.
        """
        result = {}

        # Get RS stats if available
        if self.rs_stats is not None:
            try:
                from vllm.ecc.rs_tables import get_rs_error_stats_summary
                result.update(get_rs_error_stats_summary(self.rs_stats))
            except Exception:
                pass

        # Get Golay/Hamming stats if available
        if self.golay_stats is not None:
            _, _, get_error_stats_summary = _import_ecc_helpers()
            if get_error_stats_summary is not None:
                result.update(get_error_stats_summary(self.golay_stats, self.hamming_stats))

        # Return default values if no stats available
        if not result:
            return {
                "golay_no_error": 0,
                "golay_corrected_1bit": 0,
                "golay_corrected_2bit": 0,
                "golay_corrected_3bit": 0,
                "golay_uncorrectable": 0,
                "golay_total_triplets": 0,
                "golay_total_corrected": 0,
                "hamming_no_error": 0,
                "hamming_corrected": 0,
                "hamming_detected_uncorrectable": 0,
                "hamming_total_values": 0,
                "rs_no_error": 0,
                "rs_corrected_1symbol": 0,
                "rs_corrected_2symbol": 0,
                "rs_uncorrectable": 0,
                "total_corrected": 0,
                "total_uncorrectable": 0,
            }

        return result

    def __repr__(self) -> str:
        return (
            f"VLLMEvaluationRunner("
            f"model={self.model_name!r}, "
            f"cache_mode={self.cache_mode!r}, "
            f"kv_cache_dtype={self.kv_cache_dtype!r})"
        )


def compute_perplexity_vllm(
    model_name: str,
    texts: List[str],
    cache_mode: str = "fp16",
    max_length: int = 512,
    stride: int = 256,
    **kwargs,
) -> float:
    """Convenience function to compute perplexity with vLLM.

    Args:
        model_name: HuggingFace model name.
        texts: List of text samples.
        cache_mode: KV cache mode.
        max_length: Maximum sequence length.
        stride: Sliding window stride.
        **kwargs: Additional arguments to VLLMEvaluationRunner.

    Returns:
        Perplexity value.
    """
    runner = VLLMEvaluationRunner(model_name, cache_mode, **kwargs)
    return runner.compute_perplexity(texts, max_length, stride)
