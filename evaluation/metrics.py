"""
Metrics computation for Hamming(7,4) robustness experiments.

Provides:
- Perplexity computation (model-agnostic)
- KL divergence measurement
- Token divergence measurement
- Catastrophic failure detection
"""

import math
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F


def compute_perplexity(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    max_length: int = 512,
    stride: int = 256,
    device: Optional[str] = None,
) -> float:
    """
    Compute perplexity on a list of texts using sliding window.

    Uses the standard sliding window approach where overlapping regions
    are used for context but only non-overlapping regions contribute to loss.

    Args:
        model: Causal LM model (GPT-2, LLaMA, etc.)
        tokenizer: Corresponding tokenizer
        texts: List of text strings
        max_length: Maximum sequence length per window
        stride: Sliding window stride
        device: Device to use (defaults to model device)

    Returns:
        Perplexity (exp of average cross-entropy loss)
    """
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

                # Compute target length (non-overlapping region)
                target_len = end - max(begin, prev_end)
                if target_len <= 0:
                    prev_end = end
                    if end >= seq_len:
                        break
                    continue

                input_slice = input_ids[:, begin:end]

                # Create labels with masked overlapping region
                labels = input_slice.clone()
                if begin > 0:
                    labels[:, :min(prev_end - begin, end - begin)] = -100

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
        return float('inf')

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def compute_kl_divergence(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    temperature: float = 1.0,
) -> float:
    """
    Compute KL divergence between two sets of logits.

    KL(P || Q) where P is derived from logits_a (target) and Q from logits_b.

    Args:
        logits_a: Reference logits (target distribution)
        logits_b: Test logits (approximating distribution)
        temperature: Softmax temperature

    Returns:
        KL divergence in nats
    """
    # Apply temperature and compute log probabilities
    log_p = F.log_softmax(logits_a / temperature, dim=-1)
    log_q = F.log_softmax(logits_b / temperature, dim=-1)

    # KL(P || Q) = sum(P * (log P - log Q))
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1).mean()

    return kl.item()


def load_wikitext2_test(max_samples: int = 100) -> List[str]:
    """
    Load WikiText-2 test set.

    Args:
        max_samples: Maximum number of samples to return

    Returns:
        List of text strings
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in dataset["text"] if len(t.strip()) > 50]
        return texts[:max_samples]
    except ImportError:
        # Fallback test texts
        return [
            "The quick brown fox jumps over the lazy dog. " * 10,
            "In the beginning, there was nothing. Then there was light. " * 10,
            "Science is the systematic study of the natural world. " * 10,
        ][:max_samples]


def compute_catastrophic_rate(
    perplexities: List[float],
    threshold: float = 1000.0,
) -> float:
    """
    Compute the rate of catastrophic failures.

    A catastrophic failure is when perplexity exceeds a threshold,
    indicating the model has become essentially non-functional.

    Args:
        perplexities: List of perplexity values from individual samples
        threshold: PPL threshold above which is considered catastrophic

    Returns:
        Rate of catastrophic failures (0.0 to 1.0)
    """
    if not perplexities:
        return 0.0

    catastrophic_count = sum(1 for ppl in perplexities if ppl > threshold or math.isinf(ppl))
    return catastrophic_count / len(perplexities)


def compute_top5_accuracy(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    clean_logits_cache: Optional[Dict[str, torch.Tensor]] = None,
    max_length: int = 256,
    device: Optional[str] = None,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """
    Compute top-5 accuracy: does the clean model's top-1 token appear in corrupted model's top-5?

    This is more forgiving than exact match - shows if model is "close" even when not exact.

    Args:
        model: Model (potentially with corrupted cache)
        tokenizer: Tokenizer
        texts: List of text strings
        clean_logits_cache: Pre-computed clean logits (speeds up repeated calls)
        max_length: Maximum sequence length
        device: Device to use

    Returns:
        Tuple of (top5_accuracy, logits_cache for reuse)
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    total_positions = 0
    top5_hits = 0

    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue

            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encodings.input_ids.to(device)

            if input_ids.size(1) < 2:
                continue

            try:
                outputs = model(input_ids, use_cache=False)
                logits = outputs.logits  # Shape: (1, seq_len, vocab_size)

                # Get top-5 predictions for each position
                top5_preds = torch.topk(logits[0, :-1], k=5, dim=-1).indices  # (seq_len-1, 5)

                # Target tokens (shifted by 1)
                targets = input_ids[0, 1:]  # (seq_len-1,)

                # Check if target appears in top-5
                for pos in range(targets.size(0)):
                    total_positions += 1
                    if targets[pos] in top5_preds[pos]:
                        top5_hits += 1

            except Exception:
                continue

    accuracy = top5_hits / total_positions if total_positions > 0 else 0.0
    return accuracy


def compute_mean_kl_divergence(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    clean_logits_list: List[torch.Tensor],
    max_length: int = 256,
    device: Optional[str] = None,
) -> float:
    """
    Compute mean KL divergence between clean and corrupted model outputs.

    Measures how much the probability distributions have shifted due to errors.
    More robust than token divergence as it considers full distribution.

    Args:
        model: Model (potentially with corrupted cache)
        tokenizer: Tokenizer
        texts: List of text strings
        clean_logits_list: Pre-computed clean logits for each text
        max_length: Maximum sequence length
        device: Device to use

    Returns:
        Mean KL divergence in nats (lower = more similar to clean)
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    kl_values = []

    with torch.no_grad():
        for text, clean_logits in zip(texts, clean_logits_list):
            if not text.strip() or clean_logits is None:
                continue

            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encodings.input_ids.to(device)

            if input_ids.size(1) < 2:
                continue

            try:
                outputs = model(input_ids, use_cache=False)
                corrupted_logits = outputs.logits[0]  # (seq_len, vocab_size)

                # Ensure shapes match
                min_len = min(corrupted_logits.size(0), clean_logits.size(0))
                if min_len < 1:
                    continue

                corrupted_logits = corrupted_logits[:min_len]
                clean_logits_slice = clean_logits[:min_len].to(device)

                # Compute KL divergence
                kl = compute_kl_divergence(clean_logits_slice, corrupted_logits)
                if not math.isnan(kl) and not math.isinf(kl):
                    kl_values.append(kl)

            except Exception:
                continue

    return sum(kl_values) / len(kl_values) if kl_values else 0.0


def generate_clean_logits(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    max_length: int = 256,
    device: Optional[str] = None,
) -> List[torch.Tensor]:
    """
    Generate clean baseline logits for KL divergence comparison.

    Args:
        model: Clean model (no fault injection)
        tokenizer: Tokenizer
        texts: List of text strings
        max_length: Maximum sequence length
        device: Device to use

    Returns:
        List of logit tensors (moved to CPU to save GPU memory)
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    logits_list = []

    with torch.no_grad():
        for text in texts:
            if not text.strip():
                logits_list.append(None)
                continue

            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encodings.input_ids.to(device)

            try:
                outputs = model(input_ids, use_cache=False)
                # Store on CPU to save GPU memory
                logits_list.append(outputs.logits[0].cpu())
            except Exception:
                logits_list.append(None)

    return logits_list


def compute_per_sample_perplexity(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    max_length: int = 512,
    stride: int = 256,
    device: Optional[str] = None,
) -> List[float]:
    """
    Compute perplexity for each text individually.

    Used for catastrophic failure detection.

    Args:
        model: Causal LM model
        tokenizer: Tokenizer
        texts: List of text strings
        max_length: Maximum sequence length per window
        stride: Sliding window stride
        device: Device to use

    Returns:
        List of perplexity values, one per text
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    perplexities = []

    with torch.no_grad():
        for text in texts:
            if not text.strip():
                perplexities.append(float('inf'))
                continue

            encodings = tokenizer(text, return_tensors="pt", truncation=False)
            input_ids = encodings.input_ids.to(device)
            seq_len = input_ids.size(1)

            if seq_len == 0:
                perplexities.append(float('inf'))
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
                    labels[:, :min(prev_end - begin, end - begin)] = -100

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
                perplexities.append(float('inf'))

    return perplexities
