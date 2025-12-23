/**
 * Fault Injection for ECC Testing - CUDA Implementation
 *
 * Ported from: ecc_codecs/triton_kernels/fault_injection_triton.py
 *
 * Provides BER (Bit Error Rate) based fault injection for testing
 * ECC error correction capabilities. Uses deterministic RNG for
 * reproducible experiments.
 */

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>

namespace vllm {
namespace ecc {

/**
 * Device function: Inject bit errors into a single uint8 value.
 *
 * Uses deterministic RNG based on seed and element index to ensure
 * reproducibility across runs.
 *
 * @param value: Original uint8 value
 * @param ber: Bit error rate (probability of flipping each bit)
 * @param seed: Base random seed
 * @param idx: Element index (for RNG state)
 * @param n_bits: Number of bits to potentially flip (default 8)
 * @param bits_flipped: Optional output counter for flipped bits
 * @return: Corrupted value with random bit errors
 */
__device__ __forceinline__ uint8_t inject_bit_errors_uint8(
    uint8_t value,
    float ber,
    uint64_t seed,
    int64_t idx,
    int n_bits = 8,
    int* bits_flipped = nullptr
) {
    if (ber <= 0.0f) {
        if (bits_flipped) *bits_flipped = 0;
        return value;
    }

    uint8_t error_mask = 0;
    int flip_count = 0;

    // Initialize RNG state for this element
    curandState_t state;
    curand_init(seed, idx, 0, &state);

    // Check each bit for error
    #pragma unroll
    for (int bit = 0; bit < n_bits && bit < 8; bit++) {
        float rand_val = curand_uniform(&state);
        if (rand_val < ber) {
            error_mask |= (1 << bit);
            flip_count++;
        }
    }

    if (bits_flipped) {
        *bits_flipped = flip_count;
    }

    return value ^ error_mask;
}

/**
 * Kernel: Inject bit errors into a contiguous buffer of uint8 values.
 *
 * @param data: Input/output buffer (modified in-place)
 * @param size: Number of elements
 * @param ber: Bit error rate
 * @param seed: Random seed for reproducibility
 * @param n_bits: Bits per element to consider (default 8)
 * @param total_flipped: Optional atomic counter for total bits flipped
 */
__global__ void fault_inject_uint8_kernel(
    uint8_t* data,
    int64_t size,
    float ber,
    uint64_t seed,
    int n_bits = 8,
    int* total_flipped = nullptr
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    int bits_flipped = 0;
    data[idx] = inject_bit_errors_uint8(
        data[idx], ber, seed, idx, n_bits, &bits_flipped
    );

    // Accumulate total flipped bits (optional)
    if (total_flipped && bits_flipped > 0) {
        atomicAdd(total_flipped, bits_flipped);
    }
}

/**
 * Kernel: Inject bit errors into KV cache tensors.
 *
 * Processes both key and value caches with the same BER.
 * Uses separate seed offsets for K and V to ensure independence.
 *
 * @param key_cache: Key cache tensor (uint8)
 * @param value_cache: Value cache tensor (uint8)
 * @param k_size: Size of key cache
 * @param v_size: Size of value cache
 * @param ber: Bit error rate
 * @param seed: Random seed
 * @param k_errors: Output counter for key errors
 * @param v_errors: Output counter for value errors
 */
__global__ void fault_inject_kv_cache_kernel(
    uint8_t* key_cache,
    uint8_t* value_cache,
    int64_t k_size,
    int64_t v_size,
    float ber,
    uint64_t seed,
    int* k_errors,
    int* v_errors
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process key cache
    if (idx < k_size) {
        int bits_flipped = 0;
        key_cache[idx] = inject_bit_errors_uint8(
            key_cache[idx], ber, seed, idx, 8, &bits_flipped
        );
        if (k_errors && bits_flipped > 0) {
            atomicAdd(k_errors, bits_flipped);
        }
    }

    // Process value cache (offset seed to ensure independence)
    if (idx < v_size) {
        int bits_flipped = 0;
        value_cache[idx] = inject_bit_errors_uint8(
            value_cache[idx], ber, seed + k_size, idx, 8, &bits_flipped
        );
        if (v_errors && bits_flipped > 0) {
            atomicAdd(v_errors, bits_flipped);
        }
    }
}

/**
 * Host function: Launch fault injection on a uint8 buffer.
 *
 * @param data: Device pointer to uint8 buffer
 * @param size: Number of elements
 * @param ber: Bit error rate
 * @param seed: Random seed
 * @param stream: CUDA stream (optional)
 * @return: Estimated number of bits flipped
 */
inline int64_t inject_cache_errors_launch(
    uint8_t* data,
    int64_t size,
    float ber,
    uint64_t seed,
    cudaStream_t stream = 0
) {
    if (size == 0 || ber <= 0.0f) {
        return 0;
    }

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fault_inject_uint8_kernel<<<num_blocks, block_size, 0, stream>>>(
        data, size, ber, seed, 8, nullptr
    );

    // Return expected number of flips (actual count would require sync)
    return static_cast<int64_t>(size * 8 * ber);
}

/**
 * Statistics tracking for fault injection experiments.
 */
struct FaultInjectionStats {
    int64_t total_bits;          // Total bits in cache
    int64_t bits_flipped;        // Bits actually flipped
    int64_t elements_corrupted;  // Elements with at least one flip
    float actual_ber;            // Measured BER

    __host__ void compute_actual_ber() {
        actual_ber = (total_bits > 0) ?
            static_cast<float>(bits_flipped) / static_cast<float>(total_bits) : 0.0f;
    }
};

}  // namespace ecc
}  // namespace vllm
