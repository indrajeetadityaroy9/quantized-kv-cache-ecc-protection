/**
 * Hamming(7,4) SEC (Single Error Correct) CUDA Implementation
 *
 * Ported from: ecc_codecs/triton_kernels/hamming74_triton.py
 *
 * Encoding: 4-bit data -> 7-bit codeword (stored in uint8)
 *   - Bits 0-3: Data bits (d0, d1, d2, d3)
 *   - Bits 4-6: Parity bits (p0, p1, p2)
 *   - Bit 7: Unused (always 0)
 *
 * Decoding: 7-bit codeword -> 4-bit data
 *   - Computes 3-bit syndrome to locate single-bit errors
 *   - Corrects single-bit errors
 *   - Note: Double-bit errors are NOT detected (will cause silent corruption)
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace vllm {
namespace ecc {

// Syndrome lookup table: syndrome value -> error bit position
// -1 means no error (syndrome 0)
// Values 0-3 are data bits, 4-6 are parity bits
__constant__ int8_t SYNDROME_LUT_74[8] = {-1, 4, 5, 0, 6, 1, 2, 3};

/**
 * Error type classification for Hamming(7,4) SEC decode
 */
enum class ErrorType74 : uint8_t {
    NO_ERROR = 0,           // No error detected
    SINGLE_CORRECTED = 1,   // Single-bit error found and corrected
};

/**
 * Encode a 4-bit INT4 value to a 7-bit Hamming(7,4) codeword.
 *
 * Algorithm:
 * 1. Extract 4 data bits from input
 * 2. Compute 3 Hamming parity bits using XOR
 * 3. Assemble 7-bit codeword (stored in uint8 with MSB=0)
 *
 * @param int4_val: 4-bit unsigned value (0-15)
 * @return: 7-bit Hamming codeword in uint8 (bit 7 unused)
 */
__device__ __forceinline__ uint8_t hamming74_encode(uint8_t int4_val) {
    // Extract data bits (d0, d1, d2, d3)
    uint8_t d0 = (int4_val >> 0) & 1;
    uint8_t d1 = (int4_val >> 1) & 1;
    uint8_t d2 = (int4_val >> 2) & 1;
    uint8_t d3 = (int4_val >> 3) & 1;

    // Compute Hamming(7,4) parity bits
    // p0 covers positions 1,3,5,7 -> d0, d1, d3
    // p1 covers positions 2,3,6,7 -> d0, d2, d3
    // p2 covers positions 4,5,6,7 -> d1, d2, d3
    uint8_t p0 = d0 ^ d1 ^ d3;
    uint8_t p1 = d0 ^ d2 ^ d3;
    uint8_t p2 = d1 ^ d2 ^ d3;

    // Assemble 7-bit codeword
    // Layout: [d0 d1 d2 d3 p0 p1 p2 0]
    uint8_t codeword = (d0 << 0) | (d1 << 1) | (d2 << 2) | (d3 << 3) |
                       (p0 << 4) | (p1 << 5) | (p2 << 6);

    return codeword;
}

/**
 * Decode a 7-bit Hamming(7,4) codeword to a 4-bit INT4 value.
 *
 * Algorithm:
 * 1. Extract 7 code bits
 * 2. Compute 3-bit syndrome
 * 3. If syndrome != 0, flip the indicated bit
 * 4. Extract and return 4 data bits
 *
 * Note: This is SEC-only. Double-bit errors will NOT be detected
 * and will cause silent data corruption.
 *
 * @param codeword: 7-bit Hamming codeword in uint8
 * @param error_type: Optional pointer to store error classification
 * @return: Decoded 4-bit value
 */
__device__ __forceinline__ uint8_t hamming74_decode(
    uint8_t codeword,
    ErrorType74* error_type = nullptr
) {
    // Extract individual code bits (only use lower 7 bits)
    uint8_t c0 = (codeword >> 0) & 1;
    uint8_t c1 = (codeword >> 1) & 1;
    uint8_t c2 = (codeword >> 2) & 1;
    uint8_t c3 = (codeword >> 3) & 1;
    uint8_t c4 = (codeword >> 4) & 1;
    uint8_t c5 = (codeword >> 5) & 1;
    uint8_t c6 = (codeword >> 6) & 1;

    // Compute syndrome bits
    // s0 checks positions 1,3,5,7 -> c0, c1, c3, c4 (data d0,d1,d3 and parity p0)
    // s1 checks positions 2,3,6,7 -> c0, c2, c3, c5 (data d0,d2,d3 and parity p1)
    // s2 checks positions 4,5,6,7 -> c1, c2, c3, c6 (data d1,d2,d3 and parity p2)
    uint8_t s0 = c0 ^ c1 ^ c3 ^ c4;
    uint8_t s1 = c0 ^ c2 ^ c3 ^ c5;
    uint8_t s2 = c1 ^ c2 ^ c3 ^ c6;

    // Combine syndrome bits
    int syndrome = s0 | (s1 << 1) | (s2 << 2);

    // Determine error type and apply correction
    uint8_t corrected = codeword & 0x7F;  // Mask to 7 bits
    ErrorType74 etype = ErrorType74::NO_ERROR;

    if (syndrome != 0) {
        // Single-bit error - correct it using syndrome LUT
        int8_t error_pos = SYNDROME_LUT_74[syndrome];
        if (error_pos >= 0) {
            corrected ^= (1 << error_pos);
        }
        etype = ErrorType74::SINGLE_CORRECTED;
    }

    // Store error type if requested
    if (error_type) {
        *error_type = etype;
    }

    // Extract and return 4 data bits
    return corrected & 0x0F;
}

/**
 * Combined INT4 symmetric quantization + Hamming(7,4) encode.
 *
 * Quantization: FP16/BF16/FP32 -> INT4 symmetric
 *   q = clamp(round(x / scale), -8, 7)
 *   unsigned_q = q + 8  (maps [-8,7] to [0,15])
 *
 * @param value: Input floating-point value
 * @param scale: Quantization scale (typically absmax/7.0)
 * @return: 7-bit Hamming codeword in uint8
 */
template <typename scalar_t>
__device__ __forceinline__ uint8_t int4_h74_encode(scalar_t value, float scale) {
    // Handle zero scale (avoid division by zero)
    if (scale <= 0.0f) {
        return hamming74_encode(8);  // Encode zero (8 in unsigned representation)
    }

    // Quantize to signed INT4 range [-8, 7]
    float fval = static_cast<float>(value);
    float scaled = fval / scale;
    int ival = __float2int_rn(scaled);  // Round to nearest integer
    ival = max(-8, min(7, ival));       // Clamp to INT4 range

    // Convert signed [-8,7] to unsigned [0,15]
    uint8_t int4_val = static_cast<uint8_t>(ival + 8) & 0x0F;

    // Encode with Hamming(7,4)
    return hamming74_encode(int4_val);
}

/**
 * Combined Hamming(7,4) decode + INT4 symmetric dequantization.
 *
 * Dequantization: INT4 unsigned -> signed -> FP
 *   signed_q = unsigned_q - 8  (maps [0,15] to [-8,7])
 *   x = signed_q * scale
 *
 * @param codeword: 7-bit Hamming codeword in uint8
 * @param scale: Quantization scale
 * @param error_type: Optional pointer to store error classification
 * @return: Dequantized floating-point value
 */
template <typename scalar_t>
__device__ __forceinline__ scalar_t int4_h74_decode(
    uint8_t codeword,
    float scale,
    ErrorType74* error_type = nullptr
) {
    // Decode Hamming(7,4) to INT4
    uint8_t int4_val = hamming74_decode(codeword, error_type);

    // Convert unsigned [0,15] to signed [-8,7]
    int ival = static_cast<int>(int4_val) - 8;

    // Dequantize
    float fval = static_cast<float>(ival) * scale;

    return static_cast<scalar_t>(fval);
}

// Specialization for half precision
template <>
__device__ __forceinline__ __half int4_h74_decode<__half>(
    uint8_t codeword,
    float scale,
    ErrorType74* error_type
) {
    uint8_t int4_val = hamming74_decode(codeword, error_type);
    int ival = static_cast<int>(int4_val) - 8;
    float fval = static_cast<float>(ival) * scale;
    return __float2half(fval);
}

template <>
__device__ __forceinline__ uint8_t int4_h74_encode<__half>(__half value, float scale) {
    if (scale <= 0.0f) {
        return hamming74_encode(8);
    }
    float fval = __half2float(value);
    float scaled = fval / scale;
    int ival = __float2int_rn(scaled);
    ival = max(-8, min(7, ival));
    uint8_t int4_val = static_cast<uint8_t>(ival + 8) & 0x0F;
    return hamming74_encode(int4_val);
}

// Specialization for uint16_t (PyTorch Half tensor storage type)
template <>
__device__ __forceinline__ uint8_t int4_h74_encode<uint16_t>(uint16_t val, float scale) {
    return int4_h74_encode<__half>(__ushort_as_half(val), scale);
}

template <>
__device__ __forceinline__ uint16_t int4_h74_decode<uint16_t>(
    uint8_t codeword,
    float scale,
    ErrorType74* error_type
) {
    return __half_as_ushort(int4_h74_decode<__half>(codeword, scale, error_type));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// Specialization for bfloat16 (requires compute capability 8.0+)
template <>
__device__ __forceinline__ __nv_bfloat16 int4_h74_decode<__nv_bfloat16>(
    uint8_t codeword,
    float scale,
    ErrorType74* error_type
) {
    uint8_t int4_val = hamming74_decode(codeword, error_type);
    int ival = static_cast<int>(int4_val) - 8;
    float fval = static_cast<float>(ival) * scale;
    return __float2bfloat16(fval);
}

template <>
__device__ __forceinline__ uint8_t int4_h74_encode<__nv_bfloat16>(__nv_bfloat16 value, float scale) {
    if (scale <= 0.0f) {
        return hamming74_encode(8);
    }
    float fval = __bfloat162float(value);
    float scaled = fval / scale;
    int ival = __float2int_rn(scaled);
    ival = max(-8, min(7, ival));
    uint8_t int4_val = static_cast<uint8_t>(ival + 8) & 0x0F;
    return hamming74_encode(int4_val);
}
#endif

/**
 * Compute per-block absmax scale for INT4 quantization.
 *
 * For a block of values, finds max(|x|) and computes scale = absmax / 7.0
 * This ensures the range [-8,7] maps to [-absmax, absmax] with some headroom.
 *
 * @param values: Pointer to input values
 * @param count: Number of values in block
 * @return: Quantization scale
 */
template <typename scalar_t>
__device__ __forceinline__ float compute_absmax_scale_74(
    const scalar_t* values,
    int count
) {
    float absmax = 0.0f;

    #pragma unroll 4
    for (int i = 0; i < count; i++) {
        float fval = static_cast<float>(values[i]);
        absmax = fmaxf(absmax, fabsf(fval));
    }

    // Scale maps [-7, 7] to [-absmax, absmax]
    // Using 7.0 instead of 8.0 leaves headroom for outliers
    return (absmax > 0.0f) ? (absmax / 7.0f) : 1.0f;
}

// Specialization for half precision
template <>
__device__ __forceinline__ float compute_absmax_scale_74<__half>(
    const __half* values,
    int count
) {
    float absmax = 0.0f;

    #pragma unroll 4
    for (int i = 0; i < count; i++) {
        float fval = __half2float(values[i]);
        absmax = fmaxf(absmax, fabsf(fval));
    }

    return (absmax > 0.0f) ? (absmax / 7.0f) : 1.0f;
}

}  // namespace ecc
}  // namespace vllm
