# FP8 vs INT4+ECC Comparison Under Noise

**Model:** gpt2
**Baseline PPL:** 49.89
**Samples:** 30

## Results

| BER | FP8 PPL | FP8 Δ% | INT4+ECC PPL | ECC Δ% | ECC Advantage |
|-----|---------|--------|--------------|--------|---------------|
| 0 | 49.89 | +0.0% | 49.89 | +0.0% | **+0.0%** |
| 1e-05 | 50.68 | +1.6% | 49.89 | +0.0% | **+1.6%** |
| 1e-04 | 58.34 | +16.9% | 49.92 | +0.1% | **+16.9%** |
| 1e-03 | 238.37 | +377.8% | 50.83 | +1.9% | **+375.9%** |
| 5e-03 | 124193.80 | +248823.9% | 80.45 | +61.2% | **+248762.7%** |
| 1e-02 | 309148044.45 | +619630993.7% | 319.62 | +540.6% | **+619630453.0%** |

## Key Findings

- **Maximum ECC advantage:** 619630453.0% at BER=1e-02
- **FP8 >50% degradation:** BER ≥ 1e-03
- **ECC <10% degradation:** BER ≤ 1e-03

## Conclusion

INT4+ECC provides significant fault tolerance compared to FP8:
- Single-bit errors (most common at low BER) are **fully corrected**
- Double-bit errors are **detected** (can trigger recomputation)
- At typical memory BER (1e-5 to 1e-4), ECC maintains near-baseline quality
