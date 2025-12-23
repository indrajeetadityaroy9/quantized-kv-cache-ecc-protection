/**
 * Golay(24,12) Triple-Error-Correcting CUDA Implementation
 *
 * Ported from: ecc_codecs/triton_kernels/golay_triton.py
 *
 * Encoding: 12-bit data (3 INT4 values) -> 24-bit codeword (stored in int32)
 *   - Bits 0-11: Data bits (d0-d11)
 *   - Bits 12-23: Parity bits (p0-p11)
 *
 * Decoding: 24-bit codeword -> 12-bit data
 *   - Computes 12-bit syndrome
 *   - Looks up error pattern in 4096-entry table
 *   - Corrects up to 3-bit errors
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace vllm {
namespace ecc {

// B matrix columns for parity computation
// Each column defines which data bits contribute to each parity bit
__constant__ int32_t GOLAY_B_COLS[12] = {
    0b101000111011,  // B_COL_0
    0b110100011101,  // B_COL_1
    0b111010001110,  // B_COL_2
    0b101101000111,  // B_COL_3
    0b110110100011,  // B_COL_4
    0b111011010001,  // B_COL_5
    0b111101101000,  // B_COL_6
    0b101110110100,  // B_COL_7
    0b100111011010,  // B_COL_8
    0b100011101101,  // B_COL_9
    0b110001110110,  // B_COL_10
    0b011111111111   // B_COL_11
};

// H matrix row masks for syndrome computation
// Pre-computed from H = [B^T | I_12]
__constant__ int32_t GOLAY_H_MASKS[12] = {
    0x801D6B,  // Row 0
    0x402EB5,  // Row 1
    0x2015DA,  // Row 2
    0x100BED,  // Row 3
    0x0815F6,  // Row 4
    0x040AFB,  // Row 5
    0x02157D,  // Row 6
    0x012ABE,  // Row 7
    0x00955F,  // Row 8
    0x004EAF,  // Row 9
    0x002757,  // Row 10
    0x001FAB   // Row 11
};

/**
 * Error type classification for Golay(24,12) decode
 */
enum class GolayErrorType : uint8_t {
    NO_ERROR = 0,
    CORRECTED_1 = 1,
    CORRECTED_2 = 2,
    CORRECTED_3 = 3,
    UNCORRECTABLE = 4,
};

/**
 * Compute popcount mod 2 for 12-bit value (parity)
 */
__device__ __forceinline__ int popcount_mod2_12bit(int32_t x) {
    x = x ^ (x >> 8);
    x = x ^ (x >> 4);
    x = x ^ (x >> 2);
    x = x ^ (x >> 1);
    return x & 1;
}

/**
 * Compute popcount mod 2 for 24-bit value (parity)
 */
__device__ __forceinline__ int popcount_mod2_24bit(int32_t x) {
    x = x ^ (x >> 16);
    x = x ^ (x >> 8);
    x = x ^ (x >> 4);
    x = x ^ (x >> 2);
    x = x ^ (x >> 1);
    return x & 1;
}

/**
 * Compute actual popcount (number of 1 bits) for error counting
 */
__device__ __forceinline__ int popcount_24bit(int32_t x) {
    x = (x & 0x555555) + ((x >> 1) & 0x555555);
    x = (x & 0x333333) + ((x >> 2) & 0x333333);
    x = (x & 0x0F0F0F) + ((x >> 4) & 0x0F0F0F);
    x = (x & 0x00FF00FF) + ((x >> 8) & 0x00FF00FF);
    x = (x & 0x0000FFFF) + ((x >> 16) & 0x0000FFFF);
    return x;
}

/**
 * Pack 3 INT4 values into 12-bit data
 *
 * @param n0, n1, n2: Three 4-bit values (0-15)
 * @return: 12-bit packed data
 */
__device__ __forceinline__ int32_t pack_triplet(uint8_t n0, uint8_t n1, uint8_t n2) {
    return (n0 & 0xF) | ((n1 & 0xF) << 4) | ((n2 & 0xF) << 8);
}

/**
 * Unpack 12-bit data into 3 INT4 values
 *
 * @param data_12bit: 12-bit packed data
 * @param n0, n1, n2: Output 4-bit values
 */
__device__ __forceinline__ void unpack_triplet(int32_t data_12bit,
                                                uint8_t& n0, uint8_t& n1, uint8_t& n2) {
    n0 = (data_12bit >> 0) & 0xF;
    n1 = (data_12bit >> 4) & 0xF;
    n2 = (data_12bit >> 8) & 0xF;
}

/**
 * Encode 12-bit data to 24-bit Golay codeword
 *
 * Algorithm:
 * 1. For each parity bit, compute XOR of data bits selected by B matrix column
 * 2. Combine data and parity into 24-bit codeword
 *
 * @param data_12bit: 12-bit input data
 * @return: 24-bit Golay codeword
 */
__device__ __forceinline__ int32_t golay_encode(int32_t data_12bit) {
    // Compute 12 parity bits using B matrix columns
    int32_t p0  = popcount_mod2_12bit(data_12bit & GOLAY_B_COLS[0]);
    int32_t p1  = popcount_mod2_12bit(data_12bit & GOLAY_B_COLS[1]);
    int32_t p2  = popcount_mod2_12bit(data_12bit & GOLAY_B_COLS[2]);
    int32_t p3  = popcount_mod2_12bit(data_12bit & GOLAY_B_COLS[3]);
    int32_t p4  = popcount_mod2_12bit(data_12bit & GOLAY_B_COLS[4]);
    int32_t p5  = popcount_mod2_12bit(data_12bit & GOLAY_B_COLS[5]);
    int32_t p6  = popcount_mod2_12bit(data_12bit & GOLAY_B_COLS[6]);
    int32_t p7  = popcount_mod2_12bit(data_12bit & GOLAY_B_COLS[7]);
    int32_t p8  = popcount_mod2_12bit(data_12bit & GOLAY_B_COLS[8]);
    int32_t p9  = popcount_mod2_12bit(data_12bit & GOLAY_B_COLS[9]);
    int32_t p10 = popcount_mod2_12bit(data_12bit & GOLAY_B_COLS[10]);
    int32_t p11 = popcount_mod2_12bit(data_12bit & GOLAY_B_COLS[11]);

    // Assemble parity bits
    int32_t parity_12bit = (p0 << 0)  | (p1 << 1)  | (p2 << 2)  | (p3 << 3)  |
                           (p4 << 4)  | (p5 << 5)  | (p6 << 6)  | (p7 << 7)  |
                           (p8 << 8)  | (p9 << 9)  | (p10 << 10) | (p11 << 11);

    // Combine data (low 12 bits) and parity (high 12 bits)
    return (data_12bit & 0xFFF) | (parity_12bit << 12);
}

/**
 * Compute 12-bit syndrome for Golay codeword
 *
 * @param codeword: 24-bit Golay codeword
 * @return: 12-bit syndrome (0 = no error)
 */
__device__ __forceinline__ int32_t golay_syndrome(int32_t codeword) {
    int32_t s0  = popcount_mod2_24bit(codeword & GOLAY_H_MASKS[0]);
    int32_t s1  = popcount_mod2_24bit(codeword & GOLAY_H_MASKS[1]);
    int32_t s2  = popcount_mod2_24bit(codeword & GOLAY_H_MASKS[2]);
    int32_t s3  = popcount_mod2_24bit(codeword & GOLAY_H_MASKS[3]);
    int32_t s4  = popcount_mod2_24bit(codeword & GOLAY_H_MASKS[4]);
    int32_t s5  = popcount_mod2_24bit(codeword & GOLAY_H_MASKS[5]);
    int32_t s6  = popcount_mod2_24bit(codeword & GOLAY_H_MASKS[6]);
    int32_t s7  = popcount_mod2_24bit(codeword & GOLAY_H_MASKS[7]);
    int32_t s8  = popcount_mod2_24bit(codeword & GOLAY_H_MASKS[8]);
    int32_t s9  = popcount_mod2_24bit(codeword & GOLAY_H_MASKS[9]);
    int32_t s10 = popcount_mod2_24bit(codeword & GOLAY_H_MASKS[10]);
    int32_t s11 = popcount_mod2_24bit(codeword & GOLAY_H_MASKS[11]);

    return (s0 << 0)  | (s1 << 1)  | (s2 << 2)  | (s3 << 3)  |
           (s4 << 4)  | (s5 << 5)  | (s6 << 6)  | (s7 << 7)  |
           (s8 << 8)  | (s9 << 9)  | (s10 << 10) | (s11 << 11);
}

/**
 * Decode 24-bit Golay codeword to 12-bit data with error correction
 *
 * @param codeword: 24-bit Golay codeword
 * @param syndrome_lut: Pointer to 4096-entry syndrome lookup table
 * @param error_type: Optional pointer to store error classification
 * @return: Decoded 12-bit data
 */
__device__ __forceinline__ int32_t golay_decode(
    int32_t codeword,
    const int32_t* syndrome_lut,
    GolayErrorType* error_type = nullptr
) {
    // Compute syndrome
    int32_t syndrome = golay_syndrome(codeword);

    // Look up error pattern
    int32_t error_pattern = syndrome_lut[syndrome];

    // Determine error type
    GolayErrorType etype;
    int32_t corrected;

    if (error_pattern >= 0) {
        // Correctable - apply correction
        corrected = codeword ^ error_pattern;

        // Count number of errors corrected
        int num_errors = popcount_24bit(error_pattern);
        if (num_errors == 0) {
            etype = GolayErrorType::NO_ERROR;
        } else if (num_errors == 1) {
            etype = GolayErrorType::CORRECTED_1;
        } else if (num_errors == 2) {
            etype = GolayErrorType::CORRECTED_2;
        } else {
            etype = GolayErrorType::CORRECTED_3;
        }
    } else {
        // Uncorrectable (4+ bit errors)
        corrected = codeword;  // Return uncorrected
        etype = GolayErrorType::UNCORRECTABLE;
    }

    if (error_type) {
        *error_type = etype;
    }

    // Extract 12 data bits
    return corrected & 0xFFF;
}

/**
 * Combined INT4 symmetric quantization + Golay encode for triplet
 *
 * Quantizes 3 floating-point values to INT4, packs them, and encodes with Golay
 *
 * @param v0, v1, v2: Three input floating-point values
 * @param scale: Quantization scale (typically absmax/7.0)
 * @return: 24-bit Golay codeword
 */
template <typename scalar_t>
__device__ __forceinline__ int32_t int4_golay_encode_triplet(
    scalar_t v0, scalar_t v1, scalar_t v2, float scale
) {
    // Handle zero scale
    if (scale <= 0.0f) {
        return golay_encode(pack_triplet(8, 8, 8));  // Zero values
    }

    // Quantize each value to signed INT4 [-8, 7]
    auto quantize = [scale](scalar_t v) -> uint8_t {
        float fval = static_cast<float>(v);
        float scaled = fval / scale;
        int ival = __float2int_rn(scaled);
        ival = max(-8, min(7, ival));
        return static_cast<uint8_t>(ival + 8) & 0x0F;  // Map to [0,15]
    };

    uint8_t n0 = quantize(v0);
    uint8_t n1 = quantize(v1);
    uint8_t n2 = quantize(v2);

    // Pack and encode
    int32_t data_12bit = pack_triplet(n0, n1, n2);
    return golay_encode(data_12bit);
}

/**
 * Combined Golay decode + INT4 symmetric dequantization for triplet
 *
 * @param codeword: 24-bit Golay codeword
 * @param scale: Quantization scale
 * @param syndrome_lut: Pointer to syndrome lookup table
 * @param out0, out1, out2: Output dequantized values
 * @param error_type: Optional pointer to store error classification
 */
template <typename scalar_t>
__device__ __forceinline__ void int4_golay_decode_triplet(
    int32_t codeword,
    float scale,
    const int32_t* syndrome_lut,
    scalar_t& out0, scalar_t& out1, scalar_t& out2,
    GolayErrorType* error_type = nullptr
) {
    // Decode Golay
    int32_t data_12bit = golay_decode(codeword, syndrome_lut, error_type);

    // Unpack triplet
    uint8_t n0, n1, n2;
    unpack_triplet(data_12bit, n0, n1, n2);

    // Dequantize: unsigned [0,15] -> signed [-8,7] -> float
    auto dequantize = [scale](uint8_t n) -> float {
        int ival = static_cast<int>(n) - 8;
        return static_cast<float>(ival) * scale;
    };

    out0 = static_cast<scalar_t>(dequantize(n0));
    out1 = static_cast<scalar_t>(dequantize(n1));
    out2 = static_cast<scalar_t>(dequantize(n2));
}

// Specialization for half precision
template <>
__device__ __forceinline__ int32_t int4_golay_encode_triplet<__half>(
    __half v0, __half v1, __half v2, float scale
) {
    if (scale <= 0.0f) {
        return golay_encode(pack_triplet(8, 8, 8));
    }

    auto quantize = [scale](__half v) -> uint8_t {
        float fval = __half2float(v);
        float scaled = fval / scale;
        int ival = __float2int_rn(scaled);
        ival = max(-8, min(7, ival));
        return static_cast<uint8_t>(ival + 8) & 0x0F;
    };

    uint8_t n0 = quantize(v0);
    uint8_t n1 = quantize(v1);
    uint8_t n2 = quantize(v2);

    return golay_encode(pack_triplet(n0, n1, n2));
}

// Specialization for uint16_t (PyTorch Half tensor storage type)
template <>
__device__ __forceinline__ int32_t int4_golay_encode_triplet<uint16_t>(
    uint16_t v0, uint16_t v1, uint16_t v2, float scale
) {
    return int4_golay_encode_triplet<__half>(
        __ushort_as_half(v0), __ushort_as_half(v1), __ushort_as_half(v2), scale);
}

template <>
__device__ __forceinline__ void int4_golay_decode_triplet<__half>(
    int32_t codeword,
    float scale,
    const int32_t* syndrome_lut,
    __half& out0, __half& out1, __half& out2,
    GolayErrorType* error_type
) {
    int32_t data_12bit = golay_decode(codeword, syndrome_lut, error_type);

    uint8_t n0, n1, n2;
    unpack_triplet(data_12bit, n0, n1, n2);

    auto dequantize = [scale](uint8_t n) -> __half {
        int ival = static_cast<int>(n) - 8;
        return __float2half(static_cast<float>(ival) * scale);
    };

    out0 = dequantize(n0);
    out1 = dequantize(n1);
    out2 = dequantize(n2);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// Specialization for bfloat16
template <>
__device__ __forceinline__ int32_t int4_golay_encode_triplet<__nv_bfloat16>(
    __nv_bfloat16 v0, __nv_bfloat16 v1, __nv_bfloat16 v2, float scale
) {
    if (scale <= 0.0f) {
        return golay_encode(pack_triplet(8, 8, 8));
    }

    auto quantize = [scale](__nv_bfloat16 v) -> uint8_t {
        float fval = __bfloat162float(v);
        float scaled = fval / scale;
        int ival = __float2int_rn(scaled);
        ival = max(-8, min(7, ival));
        return static_cast<uint8_t>(ival + 8) & 0x0F;
    };

    uint8_t n0 = quantize(v0);
    uint8_t n1 = quantize(v1);
    uint8_t n2 = quantize(v2);

    return golay_encode(pack_triplet(n0, n1, n2));
}

template <>
__device__ __forceinline__ void int4_golay_decode_triplet<__nv_bfloat16>(
    int32_t codeword,
    float scale,
    const int32_t* syndrome_lut,
    __nv_bfloat16& out0, __nv_bfloat16& out1, __nv_bfloat16& out2,
    GolayErrorType* error_type
) {
    int32_t data_12bit = golay_decode(codeword, syndrome_lut, error_type);

    uint8_t n0, n1, n2;
    unpack_triplet(data_12bit, n0, n1, n2);

    auto dequantize = [scale](uint8_t n) -> __nv_bfloat16 {
        int ival = static_cast<int>(n) - 8;
        return __float2bfloat16(static_cast<float>(ival) * scale);
    };

    out0 = dequantize(n0);
    out1 = dequantize(n1);
    out2 = dequantize(n2);
}
#endif

/**
 * Compute per-block absmax scale for 3 values
 */
template <typename scalar_t>
__device__ __forceinline__ float compute_absmax_scale_triplet(
    scalar_t v0, scalar_t v1, scalar_t v2
) {
    float f0 = fabsf(static_cast<float>(v0));
    float f1 = fabsf(static_cast<float>(v1));
    float f2 = fabsf(static_cast<float>(v2));
    float absmax = fmaxf(f0, fmaxf(f1, f2));
    return (absmax > 0.0f) ? (absmax / 7.0f) : 1.0f;
}

template <>
__device__ __forceinline__ float compute_absmax_scale_triplet<__half>(
    __half v0, __half v1, __half v2
) {
    float f0 = fabsf(__half2float(v0));
    float f1 = fabsf(__half2float(v1));
    float f2 = fabsf(__half2float(v2));
    float absmax = fmaxf(f0, fmaxf(f1, f2));
    return (absmax > 0.0f) ? (absmax / 7.0f) : 1.0f;
}

}  // namespace ecc
}  // namespace vllm
