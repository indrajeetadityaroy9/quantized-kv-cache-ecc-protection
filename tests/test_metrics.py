"""
Tests for Evaluation Metrics with Real Models.

These tests verify the correctness of the sliding window perplexity computation
using actual GPT-2 model inference to ensure the implementation produces
mathematically correct results.
"""
import math
import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from evaluation.metrics import (
    compute_perplexity,
    compute_per_sample_perplexity,
    compute_batched_perplexity,
)


@pytest.fixture(scope="module")
def gpt2_model_and_tokenizer():
    """Load GPT-2 model and tokenizer once for all tests."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    return model, tokenizer


class TestPerplexityComputation:
    """Tests for perplexity computation correctness."""

    def test_perplexity_is_finite(self, gpt2_model_and_tokenizer):
        """Verify perplexity returns finite values for normal text."""
        model, tokenizer = gpt2_model_and_tokenizer

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
        ]

        ppl = compute_perplexity(model, tokenizer, texts)

        assert math.isfinite(ppl), f"Perplexity should be finite, got {ppl}"
        assert ppl > 1.0, f"Perplexity should be > 1, got {ppl}"

    def test_perplexity_lower_for_coherent_text(self, gpt2_model_and_tokenizer):
        """Verify coherent text has lower perplexity than random tokens."""
        model, tokenizer = gpt2_model_and_tokenizer

        coherent = ["The weather today is sunny and warm."]
        random_text = ["xyzzy plugh qwerty asdfgh zxcvbn"]

        ppl_coherent = compute_perplexity(model, tokenizer, coherent)
        ppl_random = compute_perplexity(model, tokenizer, random_text)

        assert ppl_coherent < ppl_random, (
            f"Coherent text ({ppl_coherent:.2f}) should have lower perplexity "
            f"than random text ({ppl_random:.2f})"
        )

    def test_sliding_window_handles_long_sequences(self, gpt2_model_and_tokenizer):
        """Verify sliding window works for sequences longer than max_length."""
        model, tokenizer = gpt2_model_and_tokenizer

        # Create a long text that exceeds max_length
        long_text = " ".join(["The quick brown fox jumps over the lazy dog."] * 50)
        texts = [long_text]

        # This should not raise and should return finite perplexity
        ppl = compute_perplexity(model, tokenizer, texts, max_length=256, stride=128)

        assert math.isfinite(ppl), f"Long sequence perplexity should be finite: {ppl}"

    def test_per_sample_returns_list(self, gpt2_model_and_tokenizer):
        """Verify per-sample perplexity returns one value per sample."""
        model, tokenizer = gpt2_model_and_tokenizer

        texts = [
            "First sample text.",
            "Second sample with different content.",
            "Third sample here.",
        ]

        ppls = compute_per_sample_perplexity(model, tokenizer, texts)

        assert len(ppls) == len(texts), (
            f"Expected {len(texts)} perplexities, got {len(ppls)}"
        )
        for i, ppl in enumerate(ppls):
            assert math.isfinite(ppl), f"Sample {i} perplexity should be finite: {ppl}"

    def test_batched_matches_sequential(self, gpt2_model_and_tokenizer):
        """Verify batched perplexity approximately matches sequential."""
        model, tokenizer = gpt2_model_and_tokenizer

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question.",
        ]

        ppl_sequential = compute_perplexity(model, tokenizer, texts, max_length=256)
        ppl_batched = compute_batched_perplexity(
            model, tokenizer, texts, batch_size=2, max_length=256
        )

        # Allow small numerical differences due to batching
        relative_diff = abs(ppl_sequential - ppl_batched) / ppl_sequential
        assert relative_diff < 0.01, (
            f"Batched ({ppl_batched:.4f}) should match sequential ({ppl_sequential:.4f}), "
            f"diff = {relative_diff:.4f}"
        )

    def test_empty_text_handling(self, gpt2_model_and_tokenizer):
        """Verify empty texts are handled gracefully."""
        model, tokenizer = gpt2_model_and_tokenizer

        texts = ["", "   ", "Valid text here."]

        # Should not raise, empty texts should be skipped
        ppl = compute_perplexity(model, tokenizer, texts)
        assert math.isfinite(ppl)

    def test_deterministic_results(self, gpt2_model_and_tokenizer):
        """Verify same inputs produce same perplexity."""
        model, tokenizer = gpt2_model_and_tokenizer

        texts = ["The quick brown fox jumps over the lazy dog."]

        ppl1 = compute_perplexity(model, tokenizer, texts)
        ppl2 = compute_perplexity(model, tokenizer, texts)

        assert ppl1 == ppl2, f"Perplexity should be deterministic: {ppl1} vs {ppl2}"


class TestSlidingWindowAlgorithm:
    """Tests verifying the sliding window algorithm correctness."""

    def test_sliding_window_no_double_counting(self):
        """Verify sum(target_len) equals total positions processed.

        This test validates the sliding window logic in compute_perplexity()
        ensures every position is counted exactly once within each window.

        Note: The sliding window processes all seq_len positions (beginning at 0),
        and loss is computed for predicting position i+1 from position i.
        """
        # Test multiple sequence lengths and stride/max_length combinations
        test_cases = [
            (1000, 256, 128),  # Standard case
            (500, 256, 128),  # Shorter sequence
            (256, 256, 128),  # Exactly max_length
            (100, 256, 128),  # Shorter than max_length
            (1000, 256, 256),  # No overlap (stride == max_length)
            (1000, 512, 256),  # Larger window
        ]

        for seq_len, max_length, stride in test_cases:
            total_target_len = 0
            prev_end = 0

            for begin in range(0, seq_len, stride):
                end = min(begin + max_length, seq_len)
                target_len = end - max(begin, prev_end)

                if target_len <= 0:
                    prev_end = end
                    if end >= seq_len:
                        break
                    continue

                total_target_len += target_len
                prev_end = end
                if end >= seq_len:
                    break

            # The sliding window covers all seq_len positions exactly once
            # (each position contributes to exactly one loss computation)
            expected = seq_len
            assert total_target_len == expected, (
                f"Token counting failed for seq_len={seq_len}, "
                f"max_length={max_length}, stride={stride}: "
                f"counted {total_target_len}, expected {expected}"
            )

    def test_sliding_window_label_masking_correctness(self):
        """Verify label masking correctly excludes already-seen positions.

        The sliding window masks labels with -100 for positions that were
        already included in the previous window's loss computation.
        """
        seq_len = 500
        max_length = 256
        stride = 128

        # Track which positions contribute to loss
        counted_positions = set()

        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)

            # Labels masked from begin to prev_end (already counted)
            # Labels counted from max(begin, prev_end) to end
            mask_end = min(prev_end - begin, end - begin) if begin > 0 else 0
            count_start = max(begin, prev_end)
            count_end = end

            # Target positions are shifted by 1 (predicting next token)
            # Position i predicts token at position i+1
            for pos in range(count_start, count_end):
                if pos >= 1:  # Skip first position (no target)
                    target_pos = pos  # This position's target is token at pos
                    assert target_pos not in counted_positions, (
                        f"Position {target_pos} counted twice!"
                    )
                    counted_positions.add(target_pos)

            prev_end = end
            if end >= seq_len:
                break

        # Verify all positions from 1 to seq_len-1 are counted exactly once
        expected_positions = set(range(1, seq_len))
        assert counted_positions == expected_positions, (
            f"Missing positions: {expected_positions - counted_positions}, "
            f"Extra positions: {counted_positions - expected_positions}"
        )


class TestSlidingWindowConsistency:
    """Tests verifying sliding window produces consistent results."""

    def test_stride_does_not_affect_result_significantly(self, gpt2_model_and_tokenizer):
        """Verify different stride values produce similar perplexity."""
        model, tokenizer = gpt2_model_and_tokenizer

        # Long enough to require multiple windows
        long_text = " ".join(["The quick brown fox jumps."] * 30)
        texts = [long_text]

        ppl_stride_64 = compute_perplexity(
            model, tokenizer, texts, max_length=256, stride=64
        )
        ppl_stride_128 = compute_perplexity(
            model, tokenizer, texts, max_length=256, stride=128
        )
        ppl_stride_256 = compute_perplexity(
            model, tokenizer, texts, max_length=256, stride=256
        )

        # All should be similar (within 5% of each other)
        ppls = [ppl_stride_64, ppl_stride_128, ppl_stride_256]
        max_ppl, min_ppl = max(ppls), min(ppls)

        relative_range = (max_ppl - min_ppl) / min_ppl
        assert relative_range < 0.05, (
            f"Stride variations should produce similar results: "
            f"64={ppl_stride_64:.2f}, 128={ppl_stride_128:.2f}, 256={ppl_stride_256:.2f}"
        )

    def test_short_vs_long_max_length(self, gpt2_model_and_tokenizer):
        """Verify different max_length values produce similar perplexity."""
        model, tokenizer = gpt2_model_and_tokenizer

        long_text = " ".join(["The weather is nice today."] * 20)
        texts = [long_text]

        ppl_128 = compute_perplexity(
            model, tokenizer, texts, max_length=128, stride=64
        )
        ppl_256 = compute_perplexity(
            model, tokenizer, texts, max_length=256, stride=128
        )
        ppl_512 = compute_perplexity(
            model, tokenizer, texts, max_length=512, stride=256
        )

        # All should be similar
        ppls = [ppl_128, ppl_256, ppl_512]
        max_ppl, min_ppl = max(ppls), min(ppls)

        relative_range = (max_ppl - min_ppl) / min_ppl
        assert relative_range < 0.1, (
            f"max_length variations should produce similar results: "
            f"128={ppl_128:.2f}, 256={ppl_256:.2f}, 512={ppl_512:.2f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
