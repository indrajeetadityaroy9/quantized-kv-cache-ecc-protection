/**
 * Reed-Solomon RS(12,8) over GF(2^4) CUDA Implementation
 *
 * Encodes 8 INT4 values (32 bits) into 12 GF(16) symbols (48 bits).
 * Corrects up to 2 symbol errors with 50% storage overhead.
 *
 * Storage: 6 bytes per block (48 bits, NO padding for true density)
 *
 * GF(2^4) uses primitive polynomial x^4 + x + 1.
 * Primitive element alpha = 2.
 *
 * Decoding pipeline:
 *   1. Syndrome computation (4 syndromes)
 *   2. Berlekamp-Massey for error locator polynomial
 *   3. Chien search for error positions
 *   4. Forney algorithm for error values
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace vllm {
namespace ecc {

// ============================================================================
// GF(2^4) Lookup Tables in Constant Memory (~70 bytes total)
// ============================================================================

// Powers of primitive element alpha=2: alpha^i for i=0..31 (cyclic, extended)
__constant__ uint8_t GF16_EXP[32] = {
    1, 2, 4, 8, 3, 6, 12, 11, 5, 10, 7, 14, 15, 13, 9,  // alpha^0 to alpha^14
    1, 2, 4, 8, 3, 6, 12, 11, 5, 10, 7, 14, 15, 13, 9,  // Extended copy
    1, 2  // Extra for safe access
};

// Discrete logarithms: log_alpha(x) for x=0..15 (log[0] unused)
__constant__ uint8_t GF16_LOG[16] = {
    0,   // log(0) undefined
    0,   // log(1) = 0
    1,   // log(2) = 1
    4,   // log(3) = 4
    2,   // log(4) = 2
    8,   // log(5) = 8
    5,   // log(6) = 5
    10,  // log(7) = 10
    3,   // log(8) = 3
    14,  // log(9) = 14
    9,   // log(10) = 9
    7,   // log(11) = 7
    6,   // log(12) = 6
    13,  // log(13) = 13
    11,  // log(14) = 11
    12,  // log(15) = 12
};

// Multiplicative inverses: x^(-1) for x=0..15 (inv[0] = 0 placeholder)
__constant__ uint8_t GF16_INV[16] = {
    0,   // inv(0) undefined
    1,   // inv(1) = 1
    9,   // inv(2) = 9
    14,  // inv(3) = 14
    13,  // inv(4) = 13
    11,  // inv(5) = 11
    7,   // inv(6) = 7
    6,   // inv(7) = 6
    15,  // inv(8) = 15
    2,   // inv(9) = 2
    12,  // inv(10) = 12
    5,   // inv(11) = 5
    10,  // inv(12) = 10
    4,   // inv(13) = 4
    3,   // inv(14) = 3
    8,   // inv(15) = 8
};

// RS(12,8) generator polynomial: g(x) = x^4 + 13x^3 + 12x^2 + 8x + 7
// g(x) = (x + α)(x + α²)(x + α³)(x + α⁴) where α = 2
// Roots: α¹=2, α²=4, α³=8, α⁴=3
// Coefficients: [g0, g1, g2, g3, g4] = [7, 8, 12, 13, 1]
__constant__ uint8_t RS_GENERATOR[5] = {7, 8, 12, 13, 1};

// ============================================================================
// Error Type Classification
// ============================================================================

enum class RSErrorType : uint8_t {
    NO_ERROR = 0,
    CORRECTED_1 = 1,     // 1 symbol error corrected
    CORRECTED_2 = 2,     // 2 symbol errors corrected
    UNCORRECTABLE = 3,   // 3+ symbol errors detected
};

// ============================================================================
// GF(2^4) Arithmetic Functions
// ============================================================================

/**
 * Addition in GF(2^4) - XOR operation
 */
__device__ __forceinline__ uint8_t gf16_add(uint8_t a, uint8_t b) {
    return a ^ b;
}

/**
 * Subtraction in GF(2^4) - same as addition (XOR) in characteristic 2
 */
__device__ __forceinline__ uint8_t gf16_sub(uint8_t a, uint8_t b) {
    return a ^ b;
}

/**
 * Multiplication in GF(2^4) via log/antilog tables
 */
__device__ __forceinline__ uint8_t gf16_mul(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) return 0;
    int log_sum = GF16_LOG[a] + GF16_LOG[b];
    if (log_sum >= 15) log_sum -= 15;  // mod 15
    return GF16_EXP[log_sum];
}

/**
 * Division in GF(2^4): a / b = a * b^(-1)
 */
__device__ __forceinline__ uint8_t gf16_div(uint8_t a, uint8_t b) {
    if (a == 0) return 0;
    // a / b = a * b^(-1) = alpha^(log(a) - log(b)) mod 15
    int log_diff = GF16_LOG[a] - GF16_LOG[b];
    if (log_diff < 0) log_diff += 15;
    return GF16_EXP[log_diff];
}

/**
 * Power in GF(2^4): a^n
 */
__device__ __forceinline__ uint8_t gf16_pow(uint8_t a, int n) {
    if (a == 0) return (n == 0) ? 1 : 0;
    int log_result = (GF16_LOG[a] * n) % 15;
    if (log_result < 0) log_result += 15;
    return GF16_EXP[log_result];
}

// ============================================================================
// RS(12,8) Encoding
// ============================================================================

/**
 * Encode 8 data symbols (nibbles) to 12-symbol RS codeword.
 *
 * Uses systematic encoding: codeword = [data | parity]
 *
 * @param data: Array of 8 data symbols (uint8_t, only lower 4 bits used)
 * @param codeword: Output array of 12 symbols (packed as nibbles)
 */
__device__ __forceinline__ void rs128_encode(
    const uint8_t* data,      // 8 data symbols
    uint8_t* codeword         // 12 symbols output
) {
    // Generator polynomial: g(x) = x^4 + 15x^3 + 3x^2 + 6x + 12
    // Systematic encoding via polynomial division

    // Shift register for division (4 parity symbols)
    uint8_t reg[4] = {0, 0, 0, 0};

    // Process each data symbol from high to low degree
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        uint8_t d = data[i] & 0x0F;
        uint8_t feedback = gf16_add(d, reg[3]);

        // Shift and multiply by generator coefficients
        reg[3] = gf16_add(reg[2], gf16_mul(feedback, RS_GENERATOR[3]));  // 15
        reg[2] = gf16_add(reg[1], gf16_mul(feedback, RS_GENERATOR[2]));  // 3
        reg[1] = gf16_add(reg[0], gf16_mul(feedback, RS_GENERATOR[1]));  // 6
        reg[0] = gf16_mul(feedback, RS_GENERATOR[0]);                     // 12
    }

    // Output: [parity_0 ... parity_3 | data_0 ... data_7]
    // This format matches LFSR systematic encoding: c(x) = m(x)*x^t - r(x)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        codeword[i] = reg[i];  // Parity symbols first (positions 0-3)
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        codeword[4 + i] = data[i] & 0x0F;  // Data symbols after (positions 4-11)
    }
}

/**
 * Encode with packed I/O for true 48-bit density.
 *
 * @param data_in: 8 INT4s packed in uint32_t (bits 0-31)
 * @param out_6bytes: Output 6 bytes (48 bits, 12 nibbles)
 */
__device__ __forceinline__ void rs128_encode_packed(
    uint32_t data_in,
    uint8_t* out_6bytes
) {
    // Unpack 8 nibbles from uint32_t
    uint8_t data[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        data[i] = (data_in >> (i * 4)) & 0x0F;
    }

    // Encode
    uint8_t codeword[12];
    rs128_encode(data, codeword);

    // Pack 12 nibbles into 6 bytes
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        out_6bytes[i] = (codeword[2*i] & 0x0F) | ((codeword[2*i + 1] & 0x0F) << 4);
    }
}

// ============================================================================
// RS(12,8) Syndrome Computation
// ============================================================================

/**
 * Compute 4 syndromes for RS(12,8) codeword.
 *
 * S_j = sum(r_i * alpha^(i*j)) for i=0..11, j=1..4
 * If all syndromes are 0, there are no errors.
 *
 * @param received: 12-symbol received codeword (nibbles)
 * @param syndromes: Output 4 syndrome values
 */
__device__ __forceinline__ void rs128_syndromes(
    const uint8_t* received,  // 12 symbols
    uint8_t* syndromes        // 4 syndromes output
) {
    syndromes[0] = syndromes[1] = syndromes[2] = syndromes[3] = 0;

    #pragma unroll
    for (int i = 0; i < 12; i++) {
        uint8_t r_i = received[i] & 0x0F;
        if (r_i == 0) continue;

        // S_j = sum(r_i * alpha^(i*j)) for j=1,2,3,4
        // alpha^1=2, alpha^2=4, alpha^3=8, alpha^4=3
        uint8_t alpha_i = GF16_EXP[i % 15];              // alpha^i
        uint8_t alpha_2i = GF16_EXP[(2 * i) % 15];       // alpha^(2i)
        uint8_t alpha_3i = GF16_EXP[(3 * i) % 15];       // alpha^(3i)
        uint8_t alpha_4i = GF16_EXP[(4 * i) % 15];       // alpha^(4i)

        syndromes[0] ^= gf16_mul(r_i, alpha_i);
        syndromes[1] ^= gf16_mul(r_i, alpha_2i);
        syndromes[2] ^= gf16_mul(r_i, alpha_3i);
        syndromes[3] ^= gf16_mul(r_i, alpha_4i);
    }
}

// ============================================================================
// Berlekamp-Massey Algorithm (for t=2)
// ============================================================================

/**
 * Find error locator polynomial using Berlekamp-Massey.
 *
 * For t=2, finds sigma(x) = 1 + sigma_1*x + sigma_2*x^2
 * such that sigma(X_j^-1) = 0 for each error position X_j.
 *
 * @param syndromes: 4 syndrome values [S1, S2, S3, S4]
 * @param sigma: Output error locator [sigma_0=1, sigma_1, sigma_2]
 * @param num_errors: Output number of errors detected (0, 1, 2, or 3+ for uncorrectable)
 */
__device__ __forceinline__ void rs128_berlekamp_massey(
    const uint8_t* syndromes,
    uint8_t* sigma,
    int* num_errors
) {
    uint8_t S1 = syndromes[0], S2 = syndromes[1];
    uint8_t S3 = syndromes[2], S4 = syndromes[3];

    sigma[0] = 1;  // sigma_0 is always 1
    sigma[1] = 0;
    sigma[2] = 0;
    *num_errors = 0;

    // Check for no errors (all syndromes zero)
    if ((S1 | S2 | S3 | S4) == 0) {
        return;
    }

    // Check for 1-error case
    // For single error at position i with value e:
    // S_j = e * alpha^(i*j), so S_j = S_1^j
    // Thus S2 = S1^2, S3 = S1^3, S4 = S1^4 (all in GF)
    if (S1 != 0) {
        uint8_t S1_sq = gf16_mul(S1, S1);
        uint8_t S1_cu = gf16_mul(S1_sq, S1);
        uint8_t S1_4 = gf16_mul(S1_sq, S1_sq);

        if (S2 == S1_sq && S3 == S1_cu && S4 == S1_4) {
            // Single error: sigma(x) = 1 + S1*x
            sigma[1] = S1;
            *num_errors = 1;
            return;
        }
    }

    // 2-error case: solve linear system using Newton's identities
    // sigma_1 = S1
    // sigma_2 = (S1*S2 - S3) / (S1^2 - S2)
    // But need to verify consistency

    // For 2 errors, using the key equation:
    // | S1  1  | | sigma_2 |   | S2 |
    // | S2 S1  | | sigma_1 | = | S3 |
    //
    // Solving: sigma_1 = S1
    //          S1*sigma_2 + sigma_1 = S2 => sigma_2 = (S2 + S1) / S1 = (S2 ^ S1) / S1

    // Actually, use standard form:
    // sigma_1 + S1 = 0 => sigma_1 = S1 (in char 2)
    // sigma_2 + sigma_1*S1 + S2 = 0 => sigma_2 = S2 + S1^2

    // Verify with S3, S4:
    // sigma_2*S1 + sigma_1*S2 + S3 = 0
    // sigma_2*S2 + sigma_1*S3 + S4 = 0

    uint8_t s1 = S1;
    uint8_t s2 = gf16_add(S2, gf16_mul(S1, S1));  // S2 + S1^2

    // Verify: sigma_2*S1 + sigma_1*S2 should equal S3
    uint8_t check1 = gf16_add(gf16_mul(s2, S1), gf16_mul(s1, S2));
    // Verify: sigma_2*S2 + sigma_1*S3 should equal S4
    uint8_t check2 = gf16_add(gf16_mul(s2, S2), gf16_mul(s1, S3));

    if (check1 == S3 && check2 == S4) {
        sigma[1] = s1;
        sigma[2] = s2;
        *num_errors = 2;
        return;
    }

    // If we get here, there are 3+ errors (uncorrectable)
    *num_errors = 3;
}

// ============================================================================
// Chien Search (find error positions)
// ============================================================================

/**
 * Chien search to find error positions.
 *
 * Evaluates sigma(alpha^-i) for i=0..11 to find roots.
 * Error at position i if sigma(alpha^-i) = 0.
 *
 * @param sigma: Error locator polynomial [1, sigma_1, sigma_2]
 * @param num_errors: Expected number of errors (1 or 2)
 * @param positions: Output error positions (indices 0-11)
 * @return: Number of roots found (should equal num_errors)
 */
__device__ __forceinline__ int rs128_chien_search(
    const uint8_t* sigma,
    int num_errors,
    uint8_t* positions
) {
    int found = 0;

    #pragma unroll
    for (int i = 0; i < 12 && found < num_errors; i++) {
        // Evaluate sigma(alpha^-i) = sigma(alpha^(15-i))
        // alpha^-i = alpha^(15-i) mod 15
        int neg_i = (15 - i) % 15;
        uint8_t x = GF16_EXP[neg_i];  // alpha^-i
        uint8_t x_sq = gf16_mul(x, x);

        // sigma(x) = 1 + sigma_1*x + sigma_2*x^2
        uint8_t eval = 1;
        eval ^= gf16_mul(sigma[1], x);
        if (num_errors >= 2) {
            eval ^= gf16_mul(sigma[2], x_sq);
        }

        if (eval == 0) {
            positions[found++] = i;
        }
    }

    return found;
}

// ============================================================================
// Forney Algorithm (compute error values)
// ============================================================================

/**
 * Forney algorithm to compute error magnitudes.
 *
 * For RS codes: e_j = Omega(X_j^-1) / sigma'(X_j^-1)
 * where Omega(x) = S(x) * sigma(x) mod x^(2t)
 *       sigma'(x) = derivative of sigma(x)
 *
 * @param syndromes: 4 syndrome values
 * @param sigma: Error locator polynomial
 * @param positions: Error positions from Chien search
 * @param num_errors: Number of errors
 * @param error_values: Output error magnitudes
 */
__device__ __forceinline__ void rs128_forney(
    const uint8_t* syndromes,
    const uint8_t* sigma,
    const uint8_t* positions,
    int num_errors,
    uint8_t* error_values
) {
    uint8_t S1 = syndromes[0], S2 = syndromes[1];

    // Omega(x) = S(x) * sigma(x) mod x^4
    // S(x) = S1 + S2*x + S3*x^2 + S4*x^3
    // For t=2, we only need Omega_0 and Omega_1:
    // Omega_0 = S1
    // Omega_1 = S2 + sigma_1*S1

    uint8_t omega_0 = S1;
    uint8_t omega_1 = gf16_add(S2, gf16_mul(sigma[1], S1));

    // sigma'(x) = sigma_1 (derivative in char 2: only odd powers survive)
    // sigma(x) = 1 + sigma_1*x + sigma_2*x^2
    // sigma'(x) = sigma_1 + 0 = sigma_1
    uint8_t sigma_prime = sigma[1];

    for (int j = 0; j < num_errors; j++) {
        int pos = positions[j];
        uint8_t X_j_inv = GF16_EXP[(15 - pos) % 15];  // alpha^-pos = X_j^-1

        // Omega(X_j^-1) = omega_0 + omega_1 * X_j^-1
        uint8_t omega_eval = gf16_add(omega_0, gf16_mul(omega_1, X_j_inv));

        // sigma'(X_j^-1) = sigma_1 (constant for degree-2 sigma)
        if (sigma_prime == 0) {
            // Degenerate case - should not happen for valid 2-error case
            error_values[j] = omega_eval;
        } else {
            // e_j = Omega(X_j^-1) / sigma'(X_j^-1)
            error_values[j] = gf16_div(omega_eval, sigma_prime);
        }
    }
}

// ============================================================================
// Full RS(12,8) Decode
// ============================================================================

/**
 * Full RS(12,8) decode with error correction.
 *
 * @param received: 12-symbol received codeword
 * @param decoded: Output 8 data symbols
 * @param error_type: Output error classification
 */
__device__ __forceinline__ void rs128_decode(
    const uint8_t* received,
    uint8_t* decoded,
    RSErrorType* error_type = nullptr
) {
    // Step 1: Compute syndromes
    uint8_t syndromes[4];
    rs128_syndromes(received, syndromes);

    // Check for no errors
    if ((syndromes[0] | syndromes[1] | syndromes[2] | syndromes[3]) == 0) {
        // Data is at positions 4-11 in [parity | data] format
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            decoded[i] = received[4 + i] & 0x0F;
        }
        if (error_type) *error_type = RSErrorType::NO_ERROR;
        return;
    }

    // Step 2: Find error locator polynomial
    uint8_t sigma[3];
    int num_errors;
    rs128_berlekamp_massey(syndromes, sigma, &num_errors);

    if (num_errors > 2) {
        // Uncorrectable - return received data as-is (positions 4-11)
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            decoded[i] = received[4 + i] & 0x0F;
        }
        if (error_type) *error_type = RSErrorType::UNCORRECTABLE;
        return;
    }

    // Step 3: Find error positions (Chien search)
    uint8_t positions[2] = {0, 0};
    int found = rs128_chien_search(sigma, num_errors, positions);

    if (found != num_errors) {
        // Inconsistent - likely more than 2 errors (positions 4-11)
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            decoded[i] = received[4 + i] & 0x0F;
        }
        if (error_type) *error_type = RSErrorType::UNCORRECTABLE;
        return;
    }

    // Step 4: Compute error values (Forney)
    uint8_t error_values[2] = {0, 0};
    rs128_forney(syndromes, sigma, positions, num_errors, error_values);

    // Step 5: Apply corrections
    uint8_t corrected[12];
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        corrected[i] = received[i] & 0x0F;
    }

    for (int j = 0; j < num_errors; j++) {
        corrected[positions[j]] ^= error_values[j];
    }

    // Extract data symbols (positions 4-11 in [parity | data] format)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        decoded[i] = corrected[4 + i];
    }

    if (error_type) {
        *error_type = (num_errors == 1) ? RSErrorType::CORRECTED_1 : RSErrorType::CORRECTED_2;
    }
}

/**
 * Decode with packed I/O for true 48-bit density.
 *
 * @param in_6bytes: Input 6 bytes (48 bits, 12 nibbles)
 * @param data_out: Output 8 INT4s packed in uint32_t
 * @param error_type: Output error classification
 */
__device__ __forceinline__ void rs128_decode_packed(
    const uint8_t* in_6bytes,
    uint32_t* data_out,
    RSErrorType* error_type = nullptr
) {
    // Unpack 12 nibbles from 6 bytes
    uint8_t received[12];
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        received[2*i] = in_6bytes[i] & 0x0F;
        received[2*i + 1] = (in_6bytes[i] >> 4) & 0x0F;
    }

    // Decode
    uint8_t decoded[8];
    rs128_decode(received, decoded, error_type);

    // Pack 8 nibbles into uint32_t
    *data_out = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        *data_out |= ((uint32_t)(decoded[i] & 0x0F)) << (i * 4);
    }
}

// ============================================================================
// INT4 Quantization Integration
// ============================================================================

/**
 * Quantize FP value to unsigned INT4 (0-15)
 */
template <typename scalar_t>
__device__ __forceinline__ uint8_t quantize_to_uint4(scalar_t value, float scale) {
    float fval = static_cast<float>(value);
    float scaled = (scale > 0.0f) ? (fval / scale) : 0.0f;
    int ival = __float2int_rn(scaled);
    ival = max(-8, min(7, ival));
    return static_cast<uint8_t>(ival + 8) & 0x0F;
}

/**
 * Dequantize unsigned INT4 (0-15) to FP value
 */
template <typename scalar_t>
__device__ __forceinline__ scalar_t dequantize_from_uint4(uint8_t q, float scale) {
    int ival = static_cast<int>(q & 0x0F) - 8;
    return static_cast<scalar_t>(static_cast<float>(ival) * scale);
}

/**
 * Encode 8 FP values to RS(12,8) codeword with quantization.
 *
 * @param values_8: Input 8 FP values
 * @param scale: Quantization scale (typically absmax / 7.0)
 * @param out_6bytes: Output 6 bytes (48 bits)
 */
template <typename scalar_t>
__device__ __forceinline__ void int4_rs128_encode(
    const scalar_t* values_8,
    float scale,
    uint8_t* out_6bytes
) {
    // Quantize to uint32_t packed format
    uint32_t packed = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t q = quantize_to_uint4(values_8[i], scale);
        packed |= ((uint32_t)q) << (i * 4);
    }

    // Encode
    rs128_encode_packed(packed, out_6bytes);
}

/**
 * Decode RS(12,8) codeword to 8 FP values with dequantization.
 *
 * @param in_6bytes: Input 6 bytes (48 bits)
 * @param scale: Quantization scale
 * @param values_8: Output 8 FP values
 * @param error_type: Output error classification
 */
template <typename scalar_t>
__device__ __forceinline__ void int4_rs128_decode(
    const uint8_t* in_6bytes,
    float scale,
    scalar_t* values_8,
    RSErrorType* error_type = nullptr
) {
    // Decode
    uint32_t decoded_packed;
    rs128_decode_packed(in_6bytes, &decoded_packed, error_type);

    // Dequantize
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t q = (decoded_packed >> (i * 4)) & 0x0F;
        values_8[i] = dequantize_from_uint4<scalar_t>(q, scale);
    }
}

// ============================================================================
// Shortened RS for Remainder Blocks (k < 8)
// ============================================================================

/**
 * Encode k data symbols (k < 8) using shortened RS(k+4, k).
 *
 * Prepends (8-k) virtual zeros, encodes as RS(12,8), stores only 4 parity + k data.
 * Output format: [parity_0..3 | data_0..k-1] in packed nibbles.
 *
 * @param values: Input k FP values
 * @param k: Number of data symbols (1-7)
 * @param scale: Quantization scale
 * @param out_ptr: Output ceil((k+4)/2) bytes
 */
template <typename scalar_t>
__device__ __forceinline__ void int4_rs_shortened_encode(
    const scalar_t* values,
    int k,
    float scale,
    uint8_t* out_ptr
) {
    // Quantize k values, pad with zeros
    uint8_t data[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < k; i++) {
        data[i] = quantize_to_uint4(values[i], scale);
    }

    // Encode as full RS(12,8) - produces [parity | data] format
    uint8_t codeword[12];
    rs128_encode(data, codeword);

    // Output: 4 parity + k data = k+4 nibbles
    // codeword[0..3] = parity, codeword[4..4+k-1] = data
    // Pack into ceil((k+4)/2) bytes
    int n_symbols = k + 4;
    for (int i = 0; i < n_symbols; i += 2) {
        uint8_t lo = codeword[i];
        uint8_t hi = (i + 1 < n_symbols) ? codeword[i + 1] : 0;
        out_ptr[i / 2] = (lo & 0x0F) | ((hi & 0x0F) << 4);
    }
}

/**
 * Decode shortened RS(k+4, k) to k FP values.
 *
 * Input format: [parity_0..3 | data_0..k-1] in packed nibbles.
 * Reconstructs full 12-symbol codeword with virtual zeros at positions 4+k to 11.
 *
 * @param in_ptr: Input ceil((k+4)/2) bytes
 * @param k: Number of data symbols (1-7)
 * @param scale: Quantization scale
 * @param values: Output k FP values
 * @param error_type: Output error classification
 */
template <typename scalar_t>
__device__ __forceinline__ void int4_rs_shortened_decode(
    const uint8_t* in_ptr,
    int k,
    float scale,
    scalar_t* values,
    RSErrorType* error_type = nullptr
) {
    // Unpack k+4 nibbles from input bytes into [parity | data] format
    int n_symbols = k + 4;
    uint8_t received[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Input is packed as: [p0,p1,p2,p3,d0,...,d_{k-1}]
    // Place directly into received[0..k+3], rest are virtual zeros
    for (int i = 0; i < n_symbols; i += 2) {
        uint8_t byte = in_ptr[i / 2];
        received[i] = byte & 0x0F;
        if (i + 1 < n_symbols) {
            received[i + 1] = (byte >> 4) & 0x0F;
        }
    }
    // received[0..3] = parity, received[4..4+k-1] = data, received[4+k..11] = 0

    // Decode as full RS(12,8)
    uint8_t decoded[8];
    rs128_decode(received, decoded, error_type);

    // Output only k values (decoded already extracts from positions 4-11)
    for (int i = 0; i < k; i++) {
        values[i] = dequantize_from_uint4<scalar_t>(decoded[i], scale);
    }
}

}  // namespace ecc
}  // namespace vllm
