/**
 * ECC Error Statistics Structure for Hybrid Golay+Hamming
 *
 * Provides counters for tracking error correction during decode operations.
 * Used with atomic operations in GPU kernels for telemetry.
 */

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace vllm {
namespace ecc {

/**
 * Statistics for hybrid Golay(24,12) + Hamming(8,4) error correction.
 *
 * Counters are updated atomically during decode operations.
 * Golay stats track per-triplet (3 INT4 values) error correction.
 * Hamming stats track per-remainder (1 INT4 value) error correction.
 */
struct HybridEccStats {
    // Golay(24,12) statistics - per triplet
    int64_t golay_no_error;       // No errors detected
    int64_t golay_corrected_1;    // 1-bit error corrected
    int64_t golay_corrected_2;    // 2-bit error corrected
    int64_t golay_corrected_3;    // 3-bit error corrected
    int64_t golay_uncorrectable;  // 4+ bit errors (uncorrectable)

    // Hamming(8,4) SECDED statistics - per remainder
    int64_t hamming_no_error;     // No errors detected
    int64_t hamming_corrected;    // Single-bit error corrected
    int64_t hamming_detected;     // Double-bit error detected (uncorrectable)

    /**
     * Reset all counters to zero
     */
    __host__ __device__ void reset() {
        golay_no_error = 0;
        golay_corrected_1 = 0;
        golay_corrected_2 = 0;
        golay_corrected_3 = 0;
        golay_uncorrectable = 0;
        hamming_no_error = 0;
        hamming_corrected = 0;
        hamming_detected = 0;
    }

    /**
     * Total Golay errors corrected (1, 2, or 3 bit)
     */
    __host__ __device__ int64_t total_golay_corrected() const {
        return golay_corrected_1 + golay_corrected_2 + golay_corrected_3;
    }

    /**
     * Total errors corrected across both codecs
     */
    __host__ __device__ int64_t total_corrected() const {
        return total_golay_corrected() + hamming_corrected;
    }

    /**
     * Total uncorrectable errors (Golay 4+ bit or Hamming double-bit)
     */
    __host__ __device__ int64_t total_uncorrectable() const {
        return golay_uncorrectable + hamming_detected;
    }

    /**
     * Total triplets processed (Golay)
     */
    __host__ __device__ int64_t total_golay_triplets() const {
        return golay_no_error + golay_corrected_1 + golay_corrected_2 +
               golay_corrected_3 + golay_uncorrectable;
    }

    /**
     * Total remainder values processed (Hamming)
     */
    __host__ __device__ int64_t total_hamming_values() const {
        return hamming_no_error + hamming_corrected + hamming_detected;
    }
};

// Array indices for atomic counter access (matches GolayErrorType enum)
constexpr int GOLAY_STATS_NO_ERROR = 0;
constexpr int GOLAY_STATS_CORRECTED_1 = 1;
constexpr int GOLAY_STATS_CORRECTED_2 = 2;
constexpr int GOLAY_STATS_CORRECTED_3 = 3;
constexpr int GOLAY_STATS_UNCORRECTABLE = 4;
constexpr int GOLAY_STATS_SIZE = 5;

// Array indices for Hamming stats (matches ErrorType enum)
constexpr int HAMMING_STATS_NO_ERROR = 0;
constexpr int HAMMING_STATS_CORRECTED = 1;
constexpr int HAMMING_STATS_DETECTED = 2;
constexpr int HAMMING_STATS_PARITY_ONLY = 3;
constexpr int HAMMING_STATS_SIZE = 4;

// Array indices for Reed-Solomon RS(12,8) stats (matches RSErrorType enum)
constexpr int RS_STATS_NO_ERROR = 0;
constexpr int RS_STATS_CORRECTED_1 = 1;
constexpr int RS_STATS_CORRECTED_2 = 2;
constexpr int RS_STATS_UNCORRECTABLE = 3;
constexpr int RS_STATS_SIZE = 4;

/**
 * Statistics for Reed-Solomon RS(12,8) error correction.
 *
 * Counters are updated atomically during decode operations.
 * Stats track per-block (8 INT4 values) error correction.
 */
struct RSEccStats {
    int64_t rs_no_error;        // No errors detected
    int64_t rs_corrected_1;     // 1 symbol error corrected
    int64_t rs_corrected_2;     // 2 symbol errors corrected
    int64_t rs_uncorrectable;   // 3+ symbol errors (uncorrectable)

    /**
     * Reset all counters to zero
     */
    __host__ __device__ void reset() {
        rs_no_error = 0;
        rs_corrected_1 = 0;
        rs_corrected_2 = 0;
        rs_uncorrectable = 0;
    }

    /**
     * Total RS errors corrected (1 or 2 symbol)
     */
    __host__ __device__ int64_t total_corrected() const {
        return rs_corrected_1 + rs_corrected_2;
    }

    /**
     * Total blocks processed
     */
    __host__ __device__ int64_t total_blocks() const {
        return rs_no_error + rs_corrected_1 + rs_corrected_2 + rs_uncorrectable;
    }
};

}  // namespace ecc
}  // namespace vllm
