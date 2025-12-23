/**
 * Golay(24,12) Syndrome Lookup Table
 *
 * Pre-computed table mapping 12-bit syndrome to 24-bit error pattern.
 * - Entry value >= 0: Correctable error pattern to XOR with codeword
 * - Entry value == -1: Uncorrectable (4+ bit errors)
 *
 * Table size: 4096 entries × 4 bytes = 16KB
 *
 * This table is too large for __constant__ memory, so it should be
 * stored in global memory and passed to kernels as a parameter.
 */

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/all.h>
#include <mutex>
#include <unordered_map>

namespace vllm {
namespace ecc {

// Marker for uncorrectable syndromes
constexpr int32_t GOLAY_UNCORRECTABLE = -1;

/**
 * Build the Golay syndrome table on host.
 *
 * Returns a CPU tensor that can be copied to GPU.
 * Should be called once at initialization.
 */
inline __host__ torch::Tensor build_golay_syndrome_table_host() {
    // H matrix row masks for syndrome computation
    // H = [B^T | I_12] where B is the Golay generator matrix extension
    const int32_t h_masks[12] = {
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

    // Lambda to compute syndrome for an error pattern
    auto compute_syndrome = [&h_masks](int32_t error_pattern) -> int32_t {
        int32_t syndrome = 0;
        for (int i = 0; i < 12; i++) {
            int32_t masked = error_pattern & h_masks[i];
            // Compute parity (popcount mod 2)
            int32_t parity = masked;
            parity ^= (parity >> 16);
            parity ^= (parity >> 8);
            parity ^= (parity >> 4);
            parity ^= (parity >> 2);
            parity ^= (parity >> 1);
            parity &= 1;
            syndrome |= (parity << i);
        }
        return syndrome;
    };

    // Initialize table with -1 (uncorrectable)
    auto table = torch::full({4096}, GOLAY_UNCORRECTABLE, torch::kInt32);
    int32_t* table_ptr = table.data_ptr<int32_t>();

    // Syndrome 0 -> no error
    table_ptr[0] = 0;

    // Single-bit errors (24 patterns)
    for (int i = 0; i < 24; i++) {
        int32_t error = 1 << i;
        int32_t syndrome = compute_syndrome(error);
        table_ptr[syndrome] = error;
    }

    // Double-bit errors (C(24,2) = 276 patterns)
    for (int i = 0; i < 24; i++) {
        for (int j = i + 1; j < 24; j++) {
            int32_t error = (1 << i) | (1 << j);
            int32_t syndrome = compute_syndrome(error);
            if (table_ptr[syndrome] == GOLAY_UNCORRECTABLE) {
                table_ptr[syndrome] = error;
            }
        }
    }

    // Triple-bit errors (C(24,3) = 2024 patterns)
    for (int i = 0; i < 24; i++) {
        for (int j = i + 1; j < 24; j++) {
            for (int k = j + 1; k < 24; k++) {
                int32_t error = (1 << i) | (1 << j) | (1 << k);
                int32_t syndrome = compute_syndrome(error);
                if (table_ptr[syndrome] == GOLAY_UNCORRECTABLE) {
                    table_ptr[syndrome] = error;
                }
            }
        }
    }

    return table;
}

/**
 * Global cache for Golay syndrome tables (one per device)
 */
class GolaySyndromeTableCache {
public:
    static __host__ torch::Tensor get_table(int device_id) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = tables_.find(device_id);
        if (it != tables_.end()) {
            return it->second;
        }

        // Build table and copy to device
        auto cpu_table = build_golay_syndrome_table_host();
        auto gpu_table = cpu_table.to(torch::Device(torch::kCUDA, device_id));

        tables_[device_id] = gpu_table;
        return gpu_table;
    }

private:
    static std::unordered_map<int, torch::Tensor> tables_;
    static std::mutex mutex_;
};

// Static member definitions (must be in .cpp file in actual build)
// For header-only, we use inline
inline std::unordered_map<int, torch::Tensor> GolaySyndromeTableCache::tables_;
inline std::mutex GolaySyndromeTableCache::mutex_;

/**
 * Get Golay syndrome table for current device
 */
inline const int32_t* get_golay_syndrome_table_ptr() {
    int device_id;
    cudaGetDevice(&device_id);
    auto table = GolaySyndromeTableCache::get_table(device_id);
    return table.data_ptr<int32_t>();
}

}  // namespace ecc
}  // namespace vllm
