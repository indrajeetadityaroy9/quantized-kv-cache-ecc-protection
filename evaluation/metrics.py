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
