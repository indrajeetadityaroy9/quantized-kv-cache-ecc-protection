/**
 * Hamming(8,4) SECDED (Single Error Correct, Double Error Detect) CUDA Implementation
 *
 * Ported from: ecc_codecs/triton_kernels/hamming84_triton.py
 *
 * Encoding: 4-bit data -> 8-bit codeword
 *   - Bits 0-3: Data bits (d0, d1, d2, d3)
 *   - Bits 4-6: Parity bits (p0, p1, p2) for Hamming(7,4)
 *   - Bit 7: Overall parity for SECDED
 *
 * Decoding: 8-bit codeword -> 4-bit data
 *   - Computes syndrome to locate single-bit errors
 *   - Uses overall parity to distinguish single vs double errors
 *   - Corrects single-bit errors, detects double-bit errors
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace vllm {
namespace ecc {

// Syndrome lookup table: syndrome value -> error bit position
// -1 means no error in data/parity bits (syndrome 0 case)
// Values 0-3 are data bits, 4-6 are parity bits
__constant__ int8_t SYNDROME_LUT[8] = {-1, 4, 5, 0, 6, 1, 2, 3};

/**
 * Error type classification for Hamming(8,4) SECDED decode
 */
enum class ErrorType : uint8_t {
    NO_ERROR = 0,           // No error detected
    SINGLE_CORRECTED = 1,   // Single-bit error found and corrected
    DOUBLE_DETECTED = 2,    // Double-bit error detected (uncorrectable)
    PARITY_ONLY = 3         // Only overall parity bit is wrong
};

/**
 * Encode a 4-bit INT4 value to an 8-bit Hamming(8,4) SECDED codeword.
 *
 * Algorithm:
 * 1. Extract 4 data bits from input
 * 2. Compute 3 Hamming parity bits using XOR
 * 3. Assemble 7-bit Hamming codeword
 * 4. Compute overall parity (XOR of all 7 bits)
 * 5. Return 8-bit codeword with overall parity in MSB
 *
 * @param int4_val: 4-bit unsigned value (0-15)
 * @return: 8-bit Hamming SECDED codeword
 */
__device__ __forceinline__ uint8_t hamming84_encode(uint8_t int4_val) {
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

    // Assemble Hamming(7,4) codeword
    // Layout: [d0 d1 d2 d3 p0 p1 p2]
    uint8_t hamming7 = (d0 << 0) | (d1 << 1) | (d2 << 2) | (d3 << 3) |
                       (p0 << 4) | (p1 << 5) | (p2 << 6);

    // Compute overall parity using bit-parallel XOR
    uint8_t parity = hamming7;
    parity ^= (parity >> 4);
    parity ^= (parity >> 2);
    parity ^= (parity >> 1);
    uint8_t overall_parity = parity & 1;

    // Return 8-bit SECDED codeword
    return hamming7 | (overall_parity << 7);
}

/**
 * Decode an 8-bit Hamming(8,4) SECDED codeword to a 4-bit INT4 value.
 *
 * Algorithm:
 * 1. Extract Hamming(7,4) portion and stored overall parity
 * 2. Compute syndrome from received bits
 * 3. Compute actual overall parity and compare with stored
 * 4. Determine error type based on syndrome and parity check
 * 5. Correct single-bit error if detected
 * 6. Extract and return 4 data bits
 *
 * @param codeword: 8-bit Hamming SECDED codeword
 * @param error_type: Optional pointer to store error classification
 * @return: Decoded 4-bit value
 */
__device__ __forceinline__ uint8_t hamming84_decode(
    uint8_t codeword,
    ErrorType* error_type = nullptr
) {
    // Extract Hamming(7,4) portion and stored overall parity
    uint8_t hamming7 = codeword & 0x7F;
    uint8_t stored_parity = (codeword >> 7) & 1;

    // Extract individual code bits
    uint8_t c0 = (hamming7 >> 0) & 1;
    uint8_t c1 = (hamming7 >> 1) & 1;
    uint8_t c2 = (hamming7 >> 2) & 1;
    uint8_t c3 = (hamming7 >> 3) & 1;
    uint8_t c4 = (hamming7 >> 4) & 1;
    uint8_t c5 = (hamming7 >> 5) & 1;
    uint8_t c6 = (hamming7 >> 6) & 1;

    // Compute syndrome bits
    // s0 checks positions 1,3,5,7 -> c0, c1, c3, c4 (data d0,d1,d3 and parity p0)
    // s1 checks positions 2,3,6,7 -> c0, c2, c3, c5 (data d0,d2,d3 and parity p1)
    // s2 checks positions 4,5,6,7 -> c1, c2, c3, c6 (data d1,d2,d3 and parity p2)
    uint8_t s0 = c0 ^ c1 ^ c3 ^ c4;
    uint8_t s1 = c0 ^ c2 ^ c3 ^ c5;
    uint8_t s2 = c1 ^ c2 ^ c3 ^ c6;

    // Combine syndrome bits
    int syndrome = s0 | (s1 << 1) | (s2 << 2);

    // Compute actual overall parity
    uint8_t actual_parity = hamming7;
    actual_parity ^= (actual_parity >> 4);
    actual_parity ^= (actual_parity >> 2);
    actual_parity ^= (actual_parity >> 1);
    actual_parity &= 1;

    // Check for parity error
    bool parity_error = (stored_parity != actual_parity);

    // Determine error type and apply correction
    uint8_t corrected = hamming7;
    ErrorType etype = ErrorType::NO_ERROR;

    if (syndrome == 0) {
        if (parity_error) {
            // Only overall parity bit is wrong
            etype = ErrorType::PARITY_ONLY;
        }
        // else: no error
    } else {
        if (parity_error) {
            // Single-bit error - correct it using syndrome LUT
            int8_t error_pos = SYNDROME_LUT[syndrome];
            if (error_pos >= 0) {
                corrected ^= (1 << error_pos);
            }
            etype = ErrorType::SINGLE_CORRECTED;
        } else {
            // Double-bit error - detected but uncorrectable
            // Zero out the data to indicate unreliable value
            corrected = 0;
            etype = ErrorType::DOUBLE_DETECTED;
        }
    }

    // Store error type if requested
    if (error_type) {
        *error_type = etype;
    }

    // Extract and return 4 data bits
    return corrected & 0x0F;
}

/**
 * Combined INT4 symmetric quantization + Hamming(8,4) encode.
 *
 * Quantization: FP16/BF16/FP32 -> INT4 symmetric
 *   q = clamp(round(x / scale), -8, 7)
 *   unsigned_q = q + 8  (maps [-8,7] to [0,15])
 *
 * @param value: Input floating-point value
 * @param scale: Quantization scale (typically absmax/7.0)
 * @return: 8-bit Hamming codeword
 */
template <typename scalar_t>
__device__ __forceinline__ uint8_t int4_ecc_encode(scalar_t value, float scale) {
    // Handle zero scale (avoid division by zero)
    if (scale <= 0.0f) {
        return hamming84_encode(8);  // Encode zero (8 in unsigned representation)
    }

    // Quantize to signed INT4 range [-8, 7]
    float fval = static_cast<float>(value);
    float scaled = fval / scale;
    int ival = __float2int_rn(scaled);  // Round to nearest integer
    ival = max(-8, min(7, ival));       // Clamp to INT4 range

    // Convert signed [-8,7] to unsigned [0,15]
    uint8_t int4_val = static_cast<uint8_t>(ival + 8) & 0x0F;

    // Encode with Hamming(8,4)
    return hamming84_encode(int4_val);
}

/**
 * Combined Hamming(8,4) decode + INT4 symmetric dequantization.
 *
 * Dequantization: INT4 unsigned -> signed -> FP
 *   signed_q = unsigned_q - 8  (maps [0,15] to [-8,7])
 *   x = signed_q * scale
 *
 * @param codeword: 8-bit Hamming codeword
 * @param scale: Quantization scale
 * @param error_type: Optional pointer to store error classification
 * @return: Dequantized floating-point value
 */
template <typename scalar_t>
__device__ __forceinline__ scalar_t int4_ecc_decode(
    uint8_t codeword,
    float scale,
    ErrorType* error_type = nullptr
) {
    // Decode Hamming(8,4) to INT4
    uint8_t int4_val = hamming84_decode(codeword, error_type);

    // Convert unsigned [0,15] to signed [-8,7]
    int ival = static_cast<int>(int4_val) - 8;

    // Dequantize
    float fval = static_cast<float>(ival) * scale;

    return static_cast<scalar_t>(fval);
}

// Specialization for half precision
template <>
__device__ __forceinline__ __half int4_ecc_decode<__half>(
    uint8_t codeword,
    float scale,
    ErrorType* error_type
) {
    uint8_t int4_val = hamming84_decode(codeword, error_type);
    int ival = static_cast<int>(int4_val) - 8;
    float fval = static_cast<float>(ival) * scale;
    return __float2half(fval);
}

template <>
__device__ __forceinline__ uint8_t int4_ecc_encode<__half>(__half value, float scale) {
    if (scale <= 0.0f) {
        return hamming84_encode(8);
    }
    float fval = __half2float(value);
    float scaled = fval / scale;
    int ival = __float2int_rn(scaled);
    ival = max(-8, min(7, ival));
    uint8_t int4_val = static_cast<uint8_t>(ival + 8) & 0x0F;
    return hamming84_encode(int4_val);
}

// Specialization for uint16_t (PyTorch Half tensor storage type)
template <>
__device__ __forceinline__ uint8_t int4_ecc_encode<uint16_t>(uint16_t val, float scale) {
    return int4_ecc_encode<__half>(__ushort_as_half(val), scale);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// Specialization for bfloat16 (requires compute capability 8.0+)
template <>
__device__ __forceinline__ __nv_bfloat16 int4_ecc_decode<__nv_bfloat16>(
    uint8_t codeword,
    float scale,
    ErrorType* error_type
) {
    uint8_t int4_val = hamming84_decode(codeword, error_type);
    int ival = static_cast<int>(int4_val) - 8;
    float fval = static_cast<float>(ival) * scale;
    return __float2bfloat16(fval);
}

template <>
__device__ __forceinline__ uint8_t int4_ecc_encode<__nv_bfloat16>(__nv_bfloat16 value, float scale) {
    if (scale <= 0.0f) {
        return hamming84_encode(8);
    }
    float fval = __bfloat162float(value);
    float scaled = fval / scale;
    int ival = __float2int_rn(scaled);
    ival = max(-8, min(7, ival));
    uint8_t int4_val = static_cast<uint8_t>(ival + 8) & 0x0F;
    return hamming84_encode(int4_val);
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
__device__ __forceinline__ float compute_absmax_scale(
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
__device__ __forceinline__ float compute_absmax_scale<__half>(
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
