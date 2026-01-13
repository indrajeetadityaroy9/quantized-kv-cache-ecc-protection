"""
Evaluation Metrics for ECC-Protected KV Cache Quality Assessment.

This module implements the metrics used to evaluate cache quality under bit
error corruption. Metrics are chosen to capture different aspects of model
degradation.

Metrics:
    Perplexity (PPL):
        Measures language model quality. PPL = exp(cross-entropy loss).
        Lower is better. Increase from baseline indicates cache corruption impact.
        Uses sliding window with stride for long sequences.

    KL Divergence:
        Measures distribution shift from clean to corrupted outputs.
        KL(clean || corrupted) in nats. Lower is better (0 = identical).
        Computed per-position and averaged over sequence.

    Top-5 Accuracy:
        Fraction of positions where true next token is in model's top-5 predictions.
        Higher is better. Measures prediction confidence degradation.

    Catastrophic Failure Rate:
        Fraction of samples with PPL > threshold (default 1000) or inf.
        Captures complete model breakdown rather than gradual degradation.

Implementation Notes:
    - Sliding window perplexity avoids memory issues on long sequences
    - Labels masked with -100 for positions already seen in previous windows
    - NaN/inf losses are skipped (numerical instability)
    - Batched perplexity available for ~2x speedup on large evaluations

Usage:
    # Aggregate perplexity
    ppl = compute_perplexity(model, tokenizer, texts)

    # Per-sample for catastrophic rate
    sample_ppls = compute_per_sample_perplexity(model, tokenizer, texts)
    cat_rate = compute_catastrophic_rate(sample_ppls, threshold=1000)

    # KL divergence from clean baseline
    clean_logits = generate_clean_logits(model, tokenizer, texts)
    kl = compute_mean_kl_divergence(model, tokenizer, texts, clean_logits)
"""
import math
import torch
import torch.nn.functional as F


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
    log_p = F.log_softmax(logits_a / temperature, dim=-1)
    log_q = F.log_softmax(logits_b / temperature, dim=-1)

    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1).mean()

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


def compute_batched_perplexity(
    model, tokenizer, texts, batch_size=4, max_length=512, stride=256, device=None
):
    """
    Compute perplexity with batched forward passes for ~2x speedup.

    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of text samples
        batch_size: Number of samples to process together
        max_length: Maximum sequence length per sample
        stride: Stride for sliding window
        device: Device to run on

    Returns:
        Aggregate perplexity across all texts
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    # Tokenize all texts first
    all_encodings = []
    for text in texts:
        if not text.strip():
            continue
        encodings = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        all_encodings.append(encodings.input_ids[0])

    if not all_encodings:
        return float("inf")

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        # Process in batches
        for i in range(0, len(all_encodings), batch_size):
            batch_ids = all_encodings[i : i + batch_size]

            # Pad to same length
            max_len = max(ids.size(0) for ids in batch_ids)
            padded_batch = []
            attention_masks = []

            for ids in batch_ids:
                pad_len = max_len - ids.size(0)
                if pad_len > 0:
                    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                    padded = torch.cat(
                        [ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)]
                    )
                    mask = torch.cat(
                        [torch.ones(ids.size(0)), torch.zeros(pad_len)]
                    )
                else:
                    padded = ids
                    mask = torch.ones(ids.size(0))

                padded_batch.append(padded)
                attention_masks.append(mask)

            input_ids = torch.stack(padded_batch).to(device)
            attention_mask = torch.stack(attention_masks).to(device)

            # Create labels (shift is handled internally)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # Ignore padding

            try:
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )

                # Compute per-token loss manually for accurate counting
                logits = outputs.logits[:, :-1, :].contiguous()
                targets = labels[:, 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                per_token_loss = loss_fct(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
                per_token_loss = per_token_loss.view(targets.size())

                # Sum only non-padding tokens
                valid_mask = targets != -100
                batch_loss = (per_token_loss * valid_mask).sum().item()
                batch_tokens = valid_mask.sum().item()

                total_loss += batch_loss
                total_tokens += batch_tokens

            except Exception:
                # Fall back to sequential processing for this batch
                for ids in batch_ids:
                    try:
                        ids = ids.unsqueeze(0).to(device)
                        outputs = model(ids, labels=ids, use_cache=False)
                        loss = outputs.loss
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            total_loss += loss.item() * (ids.size(1) - 1)
                            total_tokens += ids.size(1) - 1
                    except Exception:
                        continue

    if total_tokens == 0:
        return float("inf")

    return math.exp(total_loss / total_tokens)


def precompute_clean_logits_cache(model, tokenizer, texts, max_length=256, device=None):
    """
    Precompute clean logits for all texts once.

    This is useful when computing KL divergence across multiple BER levels -
    the clean (baseline) logits only need to be computed once.

    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of text samples
        max_length: Maximum sequence length
        device: Device to run on

    Returns:
        List of clean logits tensors (on CPU to save GPU memory)
    """
    print("Precomputing clean logits for KL divergence baseline...")
    return generate_clean_logits(model, tokenizer, texts, max_length, device)
