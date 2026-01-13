# Stabilizing Quantized KV Caches via Subspace Embeddings and Manifold Interpolation

This repository contains the implementation and evaluation framework for protecting INT4-quantized KV caches in transformer models against bit-flip errors using error-correcting codes (ECC).

## Abstract

Large language model inference is bottlenecked by the memory footprint of the key-value (KV) cache, which grows linearly with sequence length. INT4 quantization reduces this footprint by 4× but introduces vulnerability to bit-level storage errors—a single bit flip in a 4-bit representation can cause up to 94% value corruption, leading to catastrophic attention score distortion.

We address this through **algebraic subspace embeddings**: encoding each INT4 value (or triplet) into a higher-dimensional binary vector constrained to lie in a linear code's subspace. This enables syndrome-based error correction during cache retrieval. For detected-but-uncorrectable errors (SECDED double-bit events), we apply **manifold-aware interpolation**, exploiting the temporal smoothness of cached states to reconstruct corrupted values from neighbors.

## Key Results

At bit error rate (BER) p = 10⁻², our methods maintain near-baseline perplexity:

| Protection Mode | LLaMA-3.1-8B PPL | GPT-2 PPL | Errors Corrected |
|-----------------|------------------|-----------|------------------|
| FP16 (Oracle)   | 1.42             | 1.78      | —                |
| Hamming(7,4)    | 1.51 ± 0.05      | 1.86 ± 0.03 | 5.6M           |
| Hamming(8,4)    | 1.50 ± 0.02      | 1.92 ± 0.18 | 5.4M           |
| **H(8,4)+Interp** | **1.44 ± 0.02** | **1.77 ± 0.03** | 5.4M       |
| **Golay(24,12)** | **1.44 ± 0.01** | **1.77 ± 0.02** | 6.6M        |

## Features

- **Three ECC Codecs**:
  - **Hamming(7,4)**: Single-error-correcting (SEC), 75% storage overhead
  - **Hamming(8,4)**: Single-error-correcting, double-error-detecting (SECDED), 100% overhead
  - **Golay(24,12)**: 3-error-correcting, encodes INT4 triplets, 100% effective overhead

- **Manifold Interpolation**: Temporal neighbor averaging for SECDED-detected double errors

- **GPU-Accelerated Triton Kernels**: Fused encode/decode operations with O(1) syndrome lookup

- **Monte Carlo Evaluation Framework**: Statistical fault injection with configurable BER levels

- **Model Support**: GPT-2, LLaMA-3.1-8B, Mistral-7B (extensible to other HuggingFace models)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/quantized-kv-cache-ecc-protection.git
cd quantized-kv-cache-ecc-protection

# Create conda environment
conda create -n ecc-kv python=3.10
conda activate ecc-kv

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- Triton 2.1+
- NVIDIA GPU (tested on A100, T4)

## Quick Start

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_cache import patch_model_with_ecc_attention, ECCShimConfig, reset_ecc_cache

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Configure ECC protection
config = ECCShimConfig(
    codec="hamming84",           # Options: hamming74, hamming84, golay
    use_interpolation=True,      # Enable manifold interpolation for SECDED
    ber=0.01,                    # Bit error rate for fault injection (0 for no injection)
    backend="triton"             # GPU-accelerated backend
)

# Patch model with ECC-protected attention
patch_model_with_ecc_attention(model, config)

# Run inference
inputs = tokenizer("The quick brown fox", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))

# Check error correction statistics
from kv_cache import get_ecc_stats
stats = get_ecc_stats()
print(f"Errors corrected: {stats['corrected']}, Errors detected: {stats['detected']}")

# Reset cache for next sequence
reset_ecc_cache()
```

### Running Experiments

```bash
# Run Monte Carlo evaluation on GPT-2
python -m evaluation.sweep \
    --model gpt2 \
    --cache-modes int4-hamming int4-hamming84 int4-hamming84-interp int12-golay \
    --ber-levels 0 1e-4 1e-3 1e-2 \
    --seeds 42 101 997 \
    --output-dir results/gpt2_experiment

# Run on LLaMA-3.1-8B (requires HuggingFace authentication)
python -m evaluation.sweep \
    --model llama-3.1-8b \
    --cache-modes fp16 int4-hamming84-interp int12-golay \
    --output-dir results/llama_experiment
```

## ECC Codec Details

### Hamming(7,4) - Single Error Correction

- **Parameters**: [7, 4, 3] code
- **Capability**: Corrects 1-bit errors
- **Storage**: 7 bits per INT4 value (75% overhead)
- **Decoding**: 8-entry syndrome lookup table

### Hamming(8,4) - SECDED

- **Parameters**: [8, 4, 4] extended code
- **Capability**: Corrects 1-bit errors, detects 2-bit errors
- **Storage**: 8 bits per INT4 value (100% overhead)
- **With Interpolation**: Detected double errors reconstructed via `v̂ = (v_{t-1} + v_{t+1}) / 2`

### Golay(24,12) - Triple Error Correction

- **Parameters**: [24, 12, 8] perfect code
- **Capability**: Corrects up to 3-bit errors
- **Storage**: 24 bits per 3 INT4 values (100% effective overhead)
- **Decoding**: 4096-entry syndrome lookup table

## Evaluation Metrics

- **Perplexity (PPL)**: `exp(cross-entropy)` - lower is better
- **KL Divergence**: Distribution shift from clean outputs (nats) - lower is better
- **Top-5 Accuracy**: Fraction where true token is in top-5 predictions - higher is better
- **Error Statistics**: Corrected/detected counts per codec

## Configuration Options

### Cache Modes

| Mode | Codec | Interpolation | Description |
|------|-------|---------------|-------------|
| `fp16` | None | No | FP16 oracle baseline |
| `int4` | None | No | Unprotected INT4 |
| `int4-hamming` | Hamming(7,4) | No | SEC protection |
| `int4-hamming84` | Hamming(8,4) | No | SECDED, keeps corrupted on detection |
| `int4-hamming84-interp` | Hamming(8,4) | Yes | SECDED + manifold interpolation |
| `int12-golay` | Golay(24,12) | No | 3-error correction |

### BER Levels

Standard evaluation uses: `[0, 1e-4, 1e-3, 1e-2]`

- `0`: No fault injection (quantization-only baseline)
- `1e-4`: Low error rate (typical DRAM soft error regime)
- `1e-3`: Moderate error rate
- `1e-2`: High error rate (stress test)

## Reproducing Paper Results

```bash
# GPT-2 experiments (runs on T4 GPU)
python -m evaluation.experiments.monte_carlo \
    --model gpt2 \
    --output-dir results/gpt2_publication \
    --max-samples 20 \
    --seeds 42 101 997

# LLaMA-3.1-8B experiments (requires A100 80GB)
python -m evaluation.experiments.monte_carlo \
    --model llama-3.1-8b \
    --output-dir results/llama_publication \
    --max-samples 20 \
    --seeds 42 101 997
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_hamming.py -v
pytest tests/test_golay.py -v
pytest tests/test_ecc_shim.py -v

# Run with coverage
pytest tests/ --cov=kv_cache --cov=ecc_codecs
```

## Limitations

- **Triton Implementation**: Current kernels are in Triton for portability; production deployment would benefit from CUDA C++ reimplementation
- **No FlashAttention Integration**: Does not integrate with FlashAttention's fused kernels
- **No PagedAttention Integration**: Not integrated with vLLM's memory management
- **BSC Error Model**: Assumes independent bit flips; real errors may exhibit correlation
