"""Evaluation metrics for ECC-protected KV caches.

This module provides:
- Perplexity computation (sliding window and torchmetrics-based)
- Statistical analysis (CI, hypothesis tests, effect sizes)
- Generation quality (BLEU, ROUGE-L)
- Downstream tasks (MMLU, HellaSwag)
"""
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from torchmetrics.functional import kl_divergence
from torchmetrics.functional.text import perplexity as tm_perplexity


def compute_perplexity(model, tokenizer, texts, max_length=512, stride=256, device=None):
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue

            encodings = tokenizer(text, return_tensors="pt", truncation=False)
            input_ids = encodings.input_ids.to(device)
            seq_len = input_ids.size(1)

            if seq_len == 0:
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

                input_slice = input_ids[:, begin:end]

                labels = input_slice.clone()
                if begin > 0:
                    labels[:, : min(prev_end - begin, end - begin)] = -100

                try:
                    outputs = model(input_slice, labels=labels, use_cache=False)
                    loss = outputs.loss

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    total_loss += loss.item() * target_len
                    total_tokens += target_len
                except Exception:
                    continue

                prev_end = end
                if end >= seq_len:
                    break

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def compute_kl_divergence(logits_a, logits_b, temperature=1.0):
    """Compute KL divergence D_KL(P||Q) using torchmetrics.

    Args:
        logits_a: Logits from distribution P (reference)
        logits_b: Logits from distribution Q (approximation)
        temperature: Temperature for softmax scaling

    Returns:
        float: KL divergence value
    """
    p = F.softmax(logits_a / temperature, dim=-1)
    q = F.softmax(logits_b / temperature, dim=-1)

    p_flat = p.reshape(-1, p.shape[-1])
    q_flat = q.reshape(-1, q.shape[-1])

    kl = kl_divergence(p_flat, q_flat, log_prob=False, reduction="mean")
    return kl.item()


def load_wikitext2_test(max_samples=100):
    try:
        from datasets import load_dataset

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in dataset["text"] if len(t.strip()) > 50]
        return texts[:max_samples]
    except ImportError:
        return [
            "The quick brown fox jumps over the lazy dog. " * 10,
            "In the beginning, there was nothing. Then there was light. " * 10,
            "Science is the systematic study of the natural world. " * 10,
        ][:max_samples]


def load_c4_validation(max_samples: int = 100) -> List[str]:
    """Load C4 validation set for web text evaluation.

    Args:
        max_samples: Maximum number of samples to load.

    Returns:
        List of text samples from C4 validation set.
    """
    from datasets import load_dataset

    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for i, item in enumerate(dataset):
        if len(texts) >= max_samples:
            break
        if len(item["text"].split()) > 50:  # Filter short texts
            texts.append(item["text"])
    return texts


def load_ptb_test(max_samples: int = 100) -> List[str]:
    """Load Penn Treebank test set.

    Args:
        max_samples: Maximum number of samples to load.

    Returns:
        List of text samples from Penn Treebank.
    """
    from datasets import load_dataset

    try:
        # Try the new namespaced version first
        dataset = load_dataset("ptb-text-only/ptb_text_only", split="test", trust_remote_code=True)
    except Exception:
        try:
            # Fallback to legacy name with trust_remote_code
            dataset = load_dataset("ptb_text_only", split="test", trust_remote_code=True)
        except Exception:
            # Final fallback: use WikiText-103 as PTB alternative
            print("Warning: PTB dataset unavailable, using WikiText-103 as fallback")
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
            texts = [t for t in dataset["text"] if len(t.strip()) > 50]
            return texts[:max_samples]

    texts = [item["sentence"] for item in dataset]
    # PTB sentences are short, so concatenate for meaningful perplexity
    combined = []
    current = ""
    for text in texts:
        current += " " + text
        if len(current.split()) > 100:
            combined.append(current.strip())
            current = ""
            if len(combined) >= max_samples:
                break
    if current and len(combined) < max_samples:
        combined.append(current.strip())
    return combined[:max_samples]


def load_dataset_by_name(name: str, max_samples: int = 100) -> List[str]:
    """Unified dataset loader.

    Args:
        name: Dataset name ('wikitext2', 'c4', 'ptb').
        max_samples: Maximum number of samples to load.

    Returns:
        List of text samples from the specified dataset.

    Raises:
        ValueError: If dataset name is not recognized.
    """
    loaders = {
        "wikitext2": load_wikitext2_test,
        "c4": load_c4_validation,
        "ptb": load_ptb_test,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")
    return loaders[name](max_samples=max_samples)


def compute_catastrophic_rate(perplexities, threshold=1000.0):
    if not perplexities:
        return 0.0

    catastrophic_count = sum(
        1 for ppl in perplexities if ppl > threshold or math.isinf(ppl)
    )
    return catastrophic_count / len(perplexities)


def compute_top5_accuracy(model, tokenizer, texts, clean_logits_cache=None, max_length=256, device=None):
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    total_positions = 0
    top5_hits = 0

    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue

            encodings = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            )
            input_ids = encodings.input_ids.to(device)

            if input_ids.size(1) < 2:
                continue

            try:
                outputs = model(input_ids, use_cache=False)
                logits = outputs.logits

                top5_preds = torch.topk(logits[0, :-1], k=5, dim=-1).indices

                targets = input_ids[0, 1:]

                for pos in range(targets.size(0)):
                    total_positions += 1
                    if targets[pos] in top5_preds[pos]:
                        top5_hits += 1

            except Exception:
                continue

    accuracy = top5_hits / total_positions if total_positions > 0 else 0.0
    return accuracy


def compute_mean_kl_divergence(model, tokenizer, texts, clean_logits_list, max_length=256, device=None):
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    kl_values = []

    with torch.no_grad():
        for text, clean_logits in zip(texts, clean_logits_list):
            if not text.strip() or clean_logits is None:
                continue

            encodings = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            )
            input_ids = encodings.input_ids.to(device)

            if input_ids.size(1) < 2:
                continue

            try:
                outputs = model(input_ids, use_cache=False)
                corrupted_logits = outputs.logits[0]

                min_len = min(corrupted_logits.size(0), clean_logits.size(0))
                if min_len < 1:
                    continue

                corrupted_logits = corrupted_logits[:min_len]
                clean_logits_slice = clean_logits[:min_len].to(device)

                kl = compute_kl_divergence(clean_logits_slice, corrupted_logits)
                if not math.isnan(kl) and not math.isinf(kl):
                    kl_values.append(kl)

            except Exception:
                continue

    return sum(kl_values) / len(kl_values) if kl_values else 0.0


def generate_clean_logits(model, tokenizer, texts, max_length=256, device=None):
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    logits_list = []

    with torch.no_grad():
        for text in texts:
            if not text.strip():
                logits_list.append(None)
                continue

            encodings = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            )
            input_ids = encodings.input_ids.to(device)

            try:
                outputs = model(input_ids, use_cache=False)

                logits_list.append(outputs.logits[0].cpu())
            except Exception:
                logits_list.append(None)

    return logits_list


def compute_per_sample_perplexity(model, tokenizer, texts, max_length=512, stride=256, device=None):
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    perplexities = []

    with torch.no_grad():
        for text in texts:
            if not text.strip():
                perplexities.append(float("inf"))
                continue

            encodings = tokenizer(text, return_tensors="pt", truncation=False)
            input_ids = encodings.input_ids.to(device)
            seq_len = input_ids.size(1)

            if seq_len == 0:
                perplexities.append(float("inf"))
                continue

            total_loss = 0.0
            total_tokens = 0
            prev_end = 0

            for begin in range(0, seq_len, stride):
                end = min(begin + max_length, seq_len)
                target_len = end - max(begin, prev_end)

                if target_len <= 0:
                    prev_end = end
                    if end >= seq_len:
                        break
                    continue

                input_slice = input_ids[:, begin:end]
                labels = input_slice.clone()
                if begin > 0:
                    labels[:, : min(prev_end - begin, end - begin)] = -100

                try:
                    outputs = model(input_slice, labels=labels, use_cache=False)
                    loss = outputs.loss

                    if not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss.item() * target_len
                        total_tokens += target_len
                except Exception:
                    pass

                prev_end = end
                if end >= seq_len:
                    break

            if total_tokens > 0:
                perplexities.append(math.exp(total_loss / total_tokens))
            else:
                perplexities.append(float("inf"))

    return perplexities


def compute_perplexity_torchmetrics(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    device=None,
    ignore_index: int = -100,
) -> float:
    """Compute perplexity using torchmetrics.

    This is a cleaner implementation using torchmetrics.functional.text.perplexity.
    For sequences longer than max_length, use compute_perplexity() with sliding window.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        texts: List of text samples.
        max_length: Maximum sequence length (truncates if longer).
        device: Device to run on.
        ignore_index: Index to ignore in loss computation.

    Returns:
        float: Perplexity value.
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue

            encodings = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            )
            input_ids = encodings.input_ids.to(device)

            if input_ids.size(1) < 2:
                continue

            try:
                outputs = model(input_ids, use_cache=False)
                logits = outputs.logits  # (1, seq_len, vocab_size)

                # Shift for next-token prediction: logits[:-1] predicts targets[1:]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = input_ids[:, 1:].contiguous()

                all_logits.append(shift_logits)
                all_targets.append(shift_targets)

            except Exception:
                continue

    if not all_logits:
        return float("inf")

    # Concatenate all samples
    combined_logits = torch.cat(all_logits, dim=1)  # (1, total_tokens, vocab_size)
    combined_targets = torch.cat(all_targets, dim=1)  # (1, total_tokens)

    # Use torchmetrics perplexity
    ppl = tm_perplexity(combined_logits, combined_targets, ignore_index=ignore_index)

    return ppl.item()


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

@dataclass
class ConfidenceInterval:
    """Confidence interval result."""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    confidence: float = 0.95
    n: int = 0


@dataclass
class HypothesisTestResult:
    """Result of a hypothesis test."""
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    effect_size_interpretation: str
    test_name: str


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Compute confidence interval using t-distribution.

    Args:
        values: List of sample values.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        ConfidenceInterval with mean, std, and bounds.
    """
    n = len(values)
    if n < 2:
        mean = values[0] if values else 0.0
        return ConfidenceInterval(
            mean=mean, std=0.0,
            ci_lower=mean, ci_upper=mean,
            confidence=confidence, n=n
        )

    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))

    # t-distribution critical value
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_critical * std / math.sqrt(n)

    return ConfidenceInterval(
        mean=mean, std=std,
        ci_lower=mean - margin, ci_upper=mean + margin,
        confidence=confidence, n=n
    )


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size.

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    arr1, arr2 = np.array(group1), np.array(group2)
    n1, n2 = len(arr1), len(arr2)

    if n1 < 2 or n2 < 2:
        return 0.0

    mean1, mean2 = np.mean(arr1), np.mean(arr2)
    var1, var2 = np.var(arr1, ddof=1), np.var(arr2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((mean1 - mean2) / pooled_std)


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def paired_t_test(
    group1: List[float],
    group2: List[float],
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Perform paired t-test for dependent samples.

    Args:
        group1: First group of measurements.
        group2: Second group (same subjects).
        alpha: Significance level.

    Returns:
        HypothesisTestResult with test statistics.
    """
    if len(group1) != len(group2):
        raise ValueError("Groups must have same length for paired test")

    t_stat, p_value = stats.ttest_rel(group1, group2)
    d = cohens_d(group1, group2)

    return HypothesisTestResult(
        statistic=float(t_stat),
        p_value=float(p_value),
        significant=p_value < alpha,
        effect_size=d,
        effect_size_interpretation=interpret_effect_size(d),
        test_name="paired_t_test"
    )


def independent_t_test(
    group1: List[float],
    group2: List[float],
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Perform independent samples t-test (Welch's).

    Args:
        group1: First group of measurements.
        group2: Second group (independent).
        alpha: Significance level.

    Returns:
        HypothesisTestResult with test statistics.
    """
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    d = cohens_d(group1, group2)

    return HypothesisTestResult(
        statistic=float(t_stat),
        p_value=float(p_value),
        significant=p_value < alpha,
        effect_size=d,
        effect_size_interpretation=interpret_effect_size(d),
        test_name="welch_t_test"
    )


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Apply Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values from multiple tests.
        alpha: Family-wise error rate.

    Returns:
        List of booleans indicating significance after correction.
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values with original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    significant = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        if p <= adjusted_alpha:
            significant[orig_idx] = True
        else:
            # Once we fail to reject, all remaining are not significant
            break

    return significant


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> ConfidenceInterval:
    """Compute bootstrap confidence interval.

    Args:
        values: Sample values.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        seed: Random seed for reproducibility.

    Returns:
        ConfidenceInterval from bootstrap distribution.
    """
    arr = np.array(values)
    n = len(arr)

    if n < 2:
        mean = float(arr[0]) if n > 0 else 0.0
        return ConfidenceInterval(mean=mean, std=0.0, ci_lower=mean, ci_upper=mean, n=n)

    rng = np.random.default_rng(seed)

    # Generate bootstrap samples efficiently
    bootstrap_indices = rng.integers(0, n, size=(n_bootstrap, n))
    bootstrap_samples = arr[bootstrap_indices]
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    bootstrap_means.sort()

    alpha = 1 - confidence
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))

    return ConfidenceInterval(
        mean=mean,
        std=std,
        ci_lower=float(bootstrap_means[lower_idx]),
        ci_upper=float(bootstrap_means[min(upper_idx, n_bootstrap - 1)]),
        confidence=confidence,
        n=n
    )


# =============================================================================
# GENERATION QUALITY METRICS (BLEU, ROUGE)
# =============================================================================

def compute_bleu(
    references: List[str],
    hypotheses: List[str],
    max_n: int = 4,
) -> float:
    """Compute corpus-level BLEU score.

    Args:
        references: List of reference texts.
        hypotheses: List of generated texts.
        max_n: Maximum n-gram order (default 4).

    Returns:
        BLEU score (0-100).
    """
    def get_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            ngrams[gram] = ngrams.get(gram, 0) + 1
        return ngrams

    def clip_count(hyp_ngrams: Dict, ref_ngrams: Dict) -> int:
        clipped = 0
        for gram, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(gram, 0))
        return clipped

    # Tokenize
    ref_tokens = [r.lower().split() for r in references]
    hyp_tokens = [h.lower().split() for h in hypotheses]

    # Compute n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        total_clipped = 0
        total_count = 0

        for ref, hyp in zip(ref_tokens, hyp_tokens):
            ref_ngrams = get_ngrams(ref, n)
            hyp_ngrams = get_ngrams(hyp, n)

            total_clipped += clip_count(hyp_ngrams, ref_ngrams)
            total_count += max(len(hyp) - n + 1, 0)

        if total_count > 0:
            precisions.append(total_clipped / total_count)
        else:
            precisions.append(0.0)

    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        log_avg = sum(math.log(p) for p in precisions) / len(precisions)
        geo_mean = math.exp(log_avg)
    else:
        geo_mean = 0.0

    # Brevity penalty
    ref_len = sum(len(r) for r in ref_tokens)
    hyp_len = sum(len(h) for h in hyp_tokens)

    if hyp_len == 0:
        return 0.0

    if hyp_len < ref_len:
        bp = math.exp(1 - ref_len / hyp_len)
    else:
        bp = 1.0

    return bp * geo_mean * 100


def compute_sentence_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Compute sentence-level BLEU score."""
    return compute_bleu([reference], [hypothesis], max_n)


def compute_rouge_l(reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute ROUGE-L score based on longest common subsequence.

    Args:
        reference: Reference text.
        hypothesis: Generated text.

    Returns:
        Dict with precision, recall, and f1.
    """
    def lcs_length(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = lcs_length(ref_tokens, hyp_tokens)

    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_generation_metrics(
    model,
    tokenizer,
    prompts: List[str],
    references: List[str],
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Compute generation quality metrics for a model.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        prompts: List of prompts to complete.
        references: List of reference completions.
        max_new_tokens: Maximum tokens to generate.
        device: Device for inference.

    Returns:
        Dict with BLEU, ROUGE-L, and individual scores.
    """
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hypotheses = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the prompt from generated text
                if generated.startswith(prompt):
                    generated = generated[len(prompt):].strip()
                hypotheses.append(generated)
            except Exception as e:
                hypotheses.append("")

    # Compute corpus BLEU
    bleu = compute_bleu(references, hypotheses)

    # Compute ROUGE-L for each pair
    rouge_scores = [compute_rouge_l(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    avg_rouge_f1 = sum(s["f1"] for s in rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    return {
        "bleu": bleu,
        "rouge_l_f1": avg_rouge_f1,
        "hypotheses": hypotheses,
        "rouge_details": rouge_scores,
    }


# =============================================================================
# DOWNSTREAM TASK EVALUATION (MMLU, HellaSwag)
# =============================================================================

# MMLU subject categories
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

MMLU_STEM_SUBJECTS = [
    "abstract_algebra", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_physics",
    "computer_security", "electrical_engineering", "machine_learning",
    "high_school_biology", "high_school_chemistry", "high_school_physics",
    "high_school_mathematics", "high_school_computer_science",
]


@dataclass
class TaskResult:
    """Result from downstream task evaluation."""
    task: str
    accuracy: float
    n_correct: int
    n_total: int
    per_sample: Optional[List[bool]] = None


def load_mmlu_subset(
    subjects: List[str] = None,
    split: str = "test",
    max_samples_per_subject: int = 50,
) -> List[Dict]:
    """Load MMLU dataset subset.

    Args:
        subjects: List of subjects to load (default: STEM subjects).
        split: Dataset split ("test", "validation", "dev").
        max_samples_per_subject: Max samples per subject.

    Returns:
        List of question dicts with question, choices, answer.
    """
    if subjects is None:
        subjects = MMLU_STEM_SUBJECTS[:5]  # Default to first 5 STEM

    try:
        from datasets import load_dataset
    except ImportError:
        # Return dummy data for testing
        return [
            {
                "question": "What is 2 + 2?",
                "choices": ["3", "4", "5", "6"],
                "answer": 1,
                "subject": "elementary_math",
            }
        ] * 10

    all_samples = []

    for subject in subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split=split)
            samples = list(dataset)[:max_samples_per_subject]

            for s in samples:
                all_samples.append({
                    "question": s["question"],
                    "choices": s["choices"],
                    "answer": s["answer"],
                    "subject": subject,
                })
        except Exception:
            continue

    return all_samples


def evaluate_mmlu(
    model,
    tokenizer,
    samples: List[Dict],
    device: str = "cuda",
) -> TaskResult:
    """Evaluate model on MMLU multiple choice questions.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        samples: List of MMLU question dicts.
        device: Device for inference.

    Returns:
        TaskResult with accuracy metrics.
    """
    model.eval()

    correct = 0
    total = 0
    per_sample = []

    choice_tokens = ["A", "B", "C", "D"]

    with torch.no_grad():
        for sample in samples:
            question = sample["question"]
            choices = sample["choices"]
            answer_idx = sample["answer"]

            # Format prompt
            prompt = f"Question: {question}\n"
            for i, choice in enumerate(choices):
                prompt += f"{choice_tokens[i]}. {choice}\n"
            prompt += "Answer:"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            try:
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]  # Last token logits

                # Get logits for A, B, C, D tokens
                choice_logits = []
                for token in choice_tokens[:len(choices)]:
                    token_id = tokenizer.encode(token, add_special_tokens=False)
                    if token_id:
                        choice_logits.append(logits[token_id[0]].item())
                    else:
                        choice_logits.append(float("-inf"))

                predicted = choice_logits.index(max(choice_logits))
                is_correct = predicted == answer_idx

                correct += int(is_correct)
                per_sample.append(is_correct)
            except Exception:
                per_sample.append(False)

            total += 1

    return TaskResult(
        task="mmlu",
        accuracy=correct / total if total > 0 else 0.0,
        n_correct=correct,
        n_total=total,
        per_sample=per_sample,
    )


def load_hellaswag_subset(
    split: str = "validation",
    max_samples: int = 200,
) -> List[Dict]:
    """Load HellaSwag dataset subset.

    Args:
        split: Dataset split.
        max_samples: Maximum number of samples.

    Returns:
        List of completion dicts.
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("hellaswag", split=split)
        samples = list(dataset)[:max_samples]

        return [
            {
                "context": s["ctx"],
                "endings": s["endings"],
                "answer": int(s["label"]),
            }
            for s in samples
        ]
    except Exception:
        # Return dummy data
        return [
            {
                "context": "A person is walking down the street.",
                "endings": [
                    "They stop to tie their shoe.",
                    "They fly into the sky.",
                    "They turn into a tree.",
                    "They explode.",
                ],
                "answer": 0,
            }
        ] * 10


def evaluate_hellaswag(
    model,
    tokenizer,
    samples: List[Dict],
    device: str = "cuda",
) -> TaskResult:
    """Evaluate model on HellaSwag completion task.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        samples: List of HellaSwag samples.
        device: Device for inference.

    Returns:
        TaskResult with accuracy metrics.
    """
    model.eval()

    correct = 0
    total = 0
    per_sample = []

    with torch.no_grad():
        for sample in samples:
            context = sample["context"]
            endings = sample["endings"]
            answer_idx = sample["answer"]

            # Score each ending by computing loss
            ending_scores = []

            for ending in endings:
                full_text = context + " " + ending
                inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                try:
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    # Lower loss = better completion
                    ending_scores.append(-outputs.loss.item())
                except Exception:
                    ending_scores.append(float("-inf"))

            predicted = ending_scores.index(max(ending_scores))
            is_correct = predicted == answer_idx

            correct += int(is_correct)
            per_sample.append(is_correct)
            total += 1

    return TaskResult(
        task="hellaswag",
        accuracy=correct / total if total > 0 else 0.0,
        n_correct=correct,
        n_total=total,
        per_sample=per_sample,
    )


def run_downstream_evaluation(
    model,
    tokenizer,
    tasks: List[str] = None,
    device: str = "cuda",
    max_samples: int = 100,
) -> Dict[str, TaskResult]:
    """Run downstream task evaluation suite.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        tasks: List of tasks ("mmlu", "hellaswag").
        device: Device for inference.
        max_samples: Maximum samples per task.

    Returns:
        Dict mapping task name to TaskResult.
    """
    if tasks is None:
        tasks = ["mmlu", "hellaswag"]

    results = {}

    if "mmlu" in tasks:
        samples = load_mmlu_subset(max_samples_per_subject=max_samples // 5)
        results["mmlu"] = evaluate_mmlu(model, tokenizer, samples, device)

    if "hellaswag" in tasks:
        samples = load_hellaswag_subset(max_samples=max_samples)
        results["hellaswag"] = evaluate_hellaswag(model, tokenizer, samples, device)

    return results
