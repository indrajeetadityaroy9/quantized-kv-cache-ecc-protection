/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "attention_dtypes.h"
#include "attention_utils.cuh"
#include "../cuda_compat.h"

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
  #include "../quantization/w8a8/fp8/amd/quant_utils.cuh"
typedef __hip_bfloat16 __nv_bfloat16;
#else
  #include "../quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#endif

// ECC-protected INT4 KV cache
#include "../ecc/hamming84.cuh"
#include "../ecc/hamming74.cuh"
#include "../ecc/golay2412.cuh"
#include "../ecc/golay_syndrome_table.cuh"
#include "../ecc/reed_solomon.cuh"
#include "../ecc/ecc_stats.cuh"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

namespace vllm {

// ECC vector conversion: decode Hamming(8,4) codewords to scalar values
// Matches fp8::scaled_convert signature for drop-in replacement
template <typename Tout, typename Tin>
__device__ __forceinline__ Tout ecc_scaled_convert(Tin in, float scale) {
  Tout out;
  const uint8_t* in_ptr = reinterpret_cast<const uint8_t*>(&in);
  using scalar_t = typename std::remove_reference<decltype(reinterpret_cast<
      typename std::conditional<std::is_same<Tout, uint32_t>::value, float,
          typename std::conditional<std::is_same<Tout, uint2>::value, float,
              typename std::conditional<std::is_same<Tout, uint4>::value, float,
                  __half>::type>::type>::type*>(&out)[0])>::type;
  scalar_t* out_ptr = reinterpret_cast<scalar_t*>(&out);
  constexpr int VEC_SIZE = sizeof(Tin) / sizeof(uint8_t);

  #pragma unroll
  for (int i = 0; i < VEC_SIZE; i++) {
    out_ptr[i] = ecc::int4_ecc_decode<scalar_t>(in_ptr[i], scale);
  }
  return out;
}

// Specialization for half precision vectors
template <>
__device__ __forceinline__ uint32_t ecc_scaled_convert<uint32_t, uint16_t>(
    uint16_t in, float scale) {
  uint32_t out;
  const uint8_t* in_ptr = reinterpret_cast<const uint8_t*>(&in);
  __half* out_ptr = reinterpret_cast<__half*>(&out);

  #pragma unroll
  for (int i = 0; i < 2; i++) {
    out_ptr[i] = ecc::int4_ecc_decode<__half>(in_ptr[i], scale);
  }
  return out;
}

template <>
__device__ __forceinline__ uint2 ecc_scaled_convert<uint2, uint32_t>(
    uint32_t in, float scale) {
  uint2 out;
  const uint8_t* in_ptr = reinterpret_cast<const uint8_t*>(&in);
  __half* out_ptr = reinterpret_cast<__half*>(&out);

  #pragma unroll
  for (int i = 0; i < 4; i++) {
    out_ptr[i] = ecc::int4_ecc_decode<__half>(in_ptr[i], scale);
  }
  return out;
}

template <>
__device__ __forceinline__ uint4 ecc_scaled_convert<uint4, uint2>(
    uint2 in, float scale) {
  uint4 out;
  const uint8_t* in_ptr = reinterpret_cast<const uint8_t*>(&in);
  __half* out_ptr = reinterpret_cast<__half*>(&out);

  #pragma unroll
  for (int i = 0; i < 8; i++) {
    out_ptr[i] = ecc::int4_ecc_decode<__half>(in_ptr[i], scale);
  }
  return out;
}

// Hamming(7,4) SEC vector conversion: decode H74 codewords to scalar values
// Matches fp8::scaled_convert signature for drop-in replacement
template <typename Tout, typename Tin>
__device__ __forceinline__ Tout h74_scaled_convert(Tin in, float scale) {
  Tout out;
  const uint8_t* in_ptr = reinterpret_cast<const uint8_t*>(&in);
  using scalar_t = typename std::remove_reference<decltype(reinterpret_cast<
      typename std::conditional<std::is_same<Tout, uint32_t>::value, float,
          typename std::conditional<std::is_same<Tout, uint2>::value, float,
              typename std::conditional<std::is_same<Tout, uint4>::value, float,
                  __half>::type>::type>::type*>(&out)[0])>::type;
  scalar_t* out_ptr = reinterpret_cast<scalar_t*>(&out);
  constexpr int VEC_SIZE = sizeof(Tin) / sizeof(uint8_t);

  #pragma unroll
  for (int i = 0; i < VEC_SIZE; i++) {
    out_ptr[i] = ecc::int4_h74_decode<scalar_t>(in_ptr[i], scale);
  }
  return out;
}

// Specializations for half precision vectors - Hamming(7,4)
template <>
__device__ __forceinline__ uint32_t h74_scaled_convert<uint32_t, uint16_t>(
    uint16_t in, float scale) {
  uint32_t out;
  const uint8_t* in_ptr = reinterpret_cast<const uint8_t*>(&in);
  __half* out_ptr = reinterpret_cast<__half*>(&out);

  #pragma unroll
  for (int i = 0; i < 2; i++) {
    out_ptr[i] = ecc::int4_h74_decode<__half>(in_ptr[i], scale);
  }
  return out;
}

template <>
__device__ __forceinline__ uint2 h74_scaled_convert<uint2, uint32_t>(
    uint32_t in, float scale) {
  uint2 out;
  const uint8_t* in_ptr = reinterpret_cast<const uint8_t*>(&in);
  __half* out_ptr = reinterpret_cast<__half*>(&out);

  #pragma unroll
  for (int i = 0; i < 4; i++) {
    out_ptr[i] = ecc::int4_h74_decode<__half>(in_ptr[i], scale);
  }
  return out;
}

template <>
__device__ __forceinline__ uint4 h74_scaled_convert<uint4, uint2>(
    uint2 in, float scale) {
  uint4 out;
  const uint8_t* in_ptr = reinterpret_cast<const uint8_t*>(&in);
  __half* out_ptr = reinterpret_cast<__half*>(&out);

  #pragma unroll
  for (int i = 0; i < 8; i++) {
    out_ptr[i] = ecc::int4_h74_decode<__half>(in_ptr[i], scale);
  }
  return out;
}

// Golay(24,12) triplet decode helper
// NOTE: Golay decodes 1 int32 codeword to 3 scalar values.
// This requires different handling than the 1:1 Hamming converters.
// The attention kernel needs to be modified to handle triplets for Golay.
template <typename scalar_t>
__device__ __forceinline__ void golay_scaled_convert_triplet(
    int32_t codeword, float scale, const int32_t* syndrome_lut,
    scalar_t& out0, scalar_t& out1, scalar_t& out2
) {
    ecc::int4_golay_decode_triplet<scalar_t>(
        codeword, scale, syndrome_lut, out0, out1, out2);
}

// Specialization for half precision
template <>
__device__ __forceinline__ void golay_scaled_convert_triplet<__half>(
    int32_t codeword, float scale, const int32_t* syndrome_lut,
    __half& out0, __half& out1, __half& out2
) {
    ecc::int4_golay_decode_triplet<__half>(
        codeword, scale, syndrome_lut, out0, out1, out2);
}

// Hybrid Golay+Hamming decode helper for K cache
// Decodes packed hybrid cache (Golay triplets + Hamming remainder) to scalar array
// Memory layout: [golay_0(4B)][golay_1(4B)]...[golay_N-1(4B)][hamming_0][hamming_1?]
template <typename scalar_t, int HEAD_SIZE>
__device__ __forceinline__ void decode_golay_hybrid_to_array(
    const uint8_t* __restrict__ cache_ptr,  // Pointer to hybrid cache for this token
    float scale,
    const int32_t* __restrict__ syndrome_lut,  // Golay syndrome table
    scalar_t* __restrict__ decoded  // Output array [HEAD_SIZE]
) {
    constexpr int num_triplets = HEAD_SIZE / 3;
    constexpr int remainder_count = HEAD_SIZE % 3;
    constexpr int golay_bytes = num_triplets * 4;

    // Decode Golay triplets
    #pragma unroll
    for (int t = 0; t < num_triplets; t++) {
        int32_t codeword = *reinterpret_cast<const int32_t*>(cache_ptr + t * 4);
        ecc::int4_golay_decode_triplet<scalar_t>(
            codeword, scale, syndrome_lut,
            decoded[t * 3], decoded[t * 3 + 1], decoded[t * 3 + 2]);
    }

    // Decode Hamming remainder
    if constexpr (remainder_count > 0) {
        #pragma unroll
        for (int r = 0; r < remainder_count; r++) {
            uint8_t hamming_cw = cache_ptr[golay_bytes + r];
            decoded[num_triplets * 3 + r] = ecc::int4_ecc_decode<scalar_t>(hamming_cw, scale);
        }
    }
}

// Specialization for half precision (explicit to avoid template issues)
template <>
__device__ __forceinline__ void decode_golay_hybrid_to_array<__half, 64>(
    const uint8_t* __restrict__ cache_ptr,
    float scale,
    const int32_t* __restrict__ syndrome_lut,
    __half* __restrict__ decoded
) {
    constexpr int num_triplets = 64 / 3;  // 21 triplets
    constexpr int remainder_count = 64 % 3;  // 1 remainder
    constexpr int golay_bytes = num_triplets * 4;  // 84 bytes

    // Decode 21 Golay triplets
    #pragma unroll
    for (int t = 0; t < num_triplets; t++) {
        int32_t codeword = *reinterpret_cast<const int32_t*>(cache_ptr + t * 4);
        ecc::int4_golay_decode_triplet<__half>(
            codeword, scale, syndrome_lut,
            decoded[t * 3], decoded[t * 3 + 1], decoded[t * 3 + 2]);
    }

    // Decode 1 Hamming remainder
    uint8_t hamming_cw = cache_ptr[golay_bytes];
    decoded[63] = ecc::int4_ecc_decode<__half>(hamming_cw, scale);
}

template <>
__device__ __forceinline__ void decode_golay_hybrid_to_array<__half, 128>(
    const uint8_t* __restrict__ cache_ptr,
    float scale,
    const int32_t* __restrict__ syndrome_lut,
    __half* __restrict__ decoded
) {
    constexpr int num_triplets = 128 / 3;  // 42 triplets
    constexpr int remainder_count = 128 % 3;  // 2 remainder
    constexpr int golay_bytes = num_triplets * 4;  // 168 bytes

    // Decode 42 Golay triplets
    #pragma unroll
    for (int t = 0; t < num_triplets; t++) {
        int32_t codeword = *reinterpret_cast<const int32_t*>(cache_ptr + t * 4);
        ecc::int4_golay_decode_triplet<__half>(
            codeword, scale, syndrome_lut,
            decoded[t * 3], decoded[t * 3 + 1], decoded[t * 3 + 2]);
    }

    // Decode 2 Hamming remainder
    decoded[126] = ecc::int4_ecc_decode<__half>(cache_ptr[golay_bytes], scale);
    decoded[127] = ecc::int4_ecc_decode<__half>(cache_ptr[golay_bytes + 1], scale);
}

template <>
__device__ __forceinline__ void decode_golay_hybrid_to_array<__half, 256>(
    const uint8_t* __restrict__ cache_ptr,
    float scale,
    const int32_t* __restrict__ syndrome_lut,
    __half* __restrict__ decoded
) {
    constexpr int num_triplets = 256 / 3;  // 85 triplets
    constexpr int golay_bytes = num_triplets * 4;  // 340 bytes

    // Decode 85 Golay triplets
    #pragma unroll
    for (int t = 0; t < 85; t++) {
        int32_t codeword = *reinterpret_cast<const int32_t*>(cache_ptr + t * 4);
        ecc::int4_golay_decode_triplet<__half>(
            codeword, scale, syndrome_lut,
            decoded[t * 3], decoded[t * 3 + 1], decoded[t * 3 + 2]);
    }

    // Decode 1 Hamming remainder (256 = 85*3 + 1)
    decoded[255] = ecc::int4_ecc_decode<__half>(cache_ptr[golay_bytes], scale);
}

// HEAD_SIZE=80: 26 triplets, 2 remainder (80 = 26*3 + 2)
template <>
__device__ __forceinline__ void decode_golay_hybrid_to_array<__half, 80>(
    const uint8_t* __restrict__ cache_ptr,
    float scale,
    const int32_t* __restrict__ syndrome_lut,
    __half* __restrict__ decoded
) {
    constexpr int num_triplets = 80 / 3;  // 26 triplets
    constexpr int golay_bytes = num_triplets * 4;  // 104 bytes

    // Decode 26 Golay triplets
    #pragma unroll
    for (int t = 0; t < 26; t++) {
        int32_t codeword = *reinterpret_cast<const int32_t*>(cache_ptr + t * 4);
        ecc::int4_golay_decode_triplet<__half>(
            codeword, scale, syndrome_lut,
            decoded[t * 3], decoded[t * 3 + 1], decoded[t * 3 + 2]);
    }

    // Decode 2 Hamming remainder
    decoded[78] = ecc::int4_ecc_decode<__half>(cache_ptr[golay_bytes], scale);
    decoded[79] = ecc::int4_ecc_decode<__half>(cache_ptr[golay_bytes + 1], scale);
}

// HEAD_SIZE=96: 32 triplets, 0 remainder (96 = 32*3) - Golay only!
template <>
__device__ __forceinline__ void decode_golay_hybrid_to_array<__half, 96>(
    const uint8_t* __restrict__ cache_ptr,
    float scale,
    const int32_t* __restrict__ syndrome_lut,
    __half* __restrict__ decoded
) {
    // Decode 32 Golay triplets (no Hamming remainder - perfect fit!)
    #pragma unroll
    for (int t = 0; t < 32; t++) {
        int32_t codeword = *reinterpret_cast<const int32_t*>(cache_ptr + t * 4);
        ecc::int4_golay_decode_triplet<__half>(
            codeword, scale, syndrome_lut,
            decoded[t * 3], decoded[t * 3 + 1], decoded[t * 3 + 2]);
    }
}

// HEAD_SIZE=112: 37 triplets, 1 remainder (112 = 37*3 + 1)
template <>
__device__ __forceinline__ void decode_golay_hybrid_to_array<__half, 112>(
    const uint8_t* __restrict__ cache_ptr,
    float scale,
    const int32_t* __restrict__ syndrome_lut,
    __half* __restrict__ decoded
) {
    constexpr int num_triplets = 112 / 3;  // 37 triplets
    constexpr int golay_bytes = num_triplets * 4;  // 148 bytes

    // Decode 37 Golay triplets
    #pragma unroll
    for (int t = 0; t < 37; t++) {
        int32_t codeword = *reinterpret_cast<const int32_t*>(cache_ptr + t * 4);
        ecc::int4_golay_decode_triplet<__half>(
            codeword, scale, syndrome_lut,
            decoded[t * 3], decoded[t * 3 + 1], decoded[t * 3 + 2]);
    }

    // Decode 1 Hamming remainder
    decoded[111] = ecc::int4_ecc_decode<__half>(cache_ptr[golay_bytes], scale);
}

// HEAD_SIZE=120: 40 triplets, 0 remainder (120 = 40*3) - Golay only!
template <>
__device__ __forceinline__ void decode_golay_hybrid_to_array<__half, 120>(
    const uint8_t* __restrict__ cache_ptr,
    float scale,
    const int32_t* __restrict__ syndrome_lut,
    __half* __restrict__ decoded
) {
    // Decode 40 Golay triplets (no Hamming remainder - perfect fit!)
    #pragma unroll
    for (int t = 0; t < 40; t++) {
        int32_t codeword = *reinterpret_cast<const int32_t*>(cache_ptr + t * 4);
        ecc::int4_golay_decode_triplet<__half>(
            codeword, scale, syndrome_lut,
            decoded[t * 3], decoded[t * 3 + 1], decoded[t * 3 + 2]);
    }
}

// HEAD_SIZE=192: 64 triplets, 0 remainder (192 = 64*3) - Golay only!
template <>
__device__ __forceinline__ void decode_golay_hybrid_to_array<__half, 192>(
    const uint8_t* __restrict__ cache_ptr,
    float scale,
    const int32_t* __restrict__ syndrome_lut,
    __half* __restrict__ decoded
) {
    // Decode 64 Golay triplets (no Hamming remainder - perfect fit!)
    #pragma unroll
    for (int t = 0; t < 64; t++) {
        int32_t codeword = *reinterpret_cast<const int32_t*>(cache_ptr + t * 4);
        ecc::int4_golay_decode_triplet<__half>(
            codeword, scale, syndrome_lut,
            decoded[t * 3], decoded[t * 3 + 1], decoded[t * 3 + 2]);
    }
}

// Hybrid Golay+Hamming decode with error statistics tracking
// Uses atomic counters to accumulate error correction statistics
// golay_stats: [5] int64 array (no_error, corrected_1, corrected_2, corrected_3, uncorrectable)
// hamming_stats: [4] int64 array (no_error, corrected, detected, parity_only)
template <typename scalar_t, int HEAD_SIZE>
__device__ __forceinline__ void decode_golay_hybrid_to_array_with_stats(
    const uint8_t* __restrict__ cache_ptr,
    float scale,
    const int32_t* __restrict__ syndrome_lut,
    scalar_t* __restrict__ decoded,
    int64_t* __restrict__ golay_stats,    // Optional: nullptr to skip Golay stats
    int64_t* __restrict__ hamming_stats   // Optional: nullptr to skip Hamming stats
) {
    constexpr int num_triplets = HEAD_SIZE / 3;
    constexpr int remainder_count = HEAD_SIZE % 3;
    constexpr int golay_bytes = num_triplets * 4;

    // Decode Golay triplets with error tracking
    #pragma unroll
    for (int t = 0; t < num_triplets; t++) {
        int32_t codeword = *reinterpret_cast<const int32_t*>(cache_ptr + t * 4);
        ecc::GolayErrorType etype;
        ecc::int4_golay_decode_triplet<scalar_t>(
            codeword, scale, syndrome_lut,
            decoded[t * 3], decoded[t * 3 + 1], decoded[t * 3 + 2], &etype);

        if (golay_stats) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&golay_stats[static_cast<int>(etype)]), 1ULL);
        }
    }

    // Decode Hamming remainder with error tracking
    if constexpr (remainder_count > 0) {
        #pragma unroll
        for (int r = 0; r < remainder_count; r++) {
            ecc::ErrorType etype;
            decoded[num_triplets * 3 + r] = ecc::int4_ecc_decode<scalar_t>(
                cache_ptr[golay_bytes + r], scale, &etype);

            if (hamming_stats) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&hamming_stats[static_cast<int>(etype)]), 1ULL);
            }
        }
    }
}

// Reed-Solomon RS(12,8) decode helper for K cache
// Decodes packed RS cache (octuplets + shortened remainder) to scalar array
// Memory layout: [rs_0(6B)][rs_1(6B)]...[rs_N-1(6B)][shortened_remainder?]
// where N = head_size / 8, remainder = head_size % 8
template <typename scalar_t, int HEAD_SIZE>
__device__ __forceinline__ void decode_rs128_to_array(
    const uint8_t* __restrict__ cache_ptr,  // Pointer to RS cache for this token
    float scale,
    scalar_t* __restrict__ decoded  // Output array [HEAD_SIZE]
) {
    constexpr int num_octuplets = HEAD_SIZE / 8;
    constexpr int remainder_count = HEAD_SIZE % 8;
    constexpr int full_rs_bytes = num_octuplets * 6;

    // Decode full RS(12,8) octuplets
    #pragma unroll
    for (int oct = 0; oct < num_octuplets; oct++) {
        ecc::int4_rs128_decode<scalar_t>(
            cache_ptr + oct * 6, scale, decoded + oct * 8, nullptr);
    }

    // Decode shortened RS remainder
    if constexpr (remainder_count > 0) {
        ecc::int4_rs_shortened_decode<scalar_t>(
            cache_ptr + full_rs_bytes, remainder_count, scale,
            decoded + num_octuplets * 8, nullptr);
    }
}

// Reed-Solomon RS(12,8) decode with error statistics tracking
// rs_stats: [4] int64 array (no_error, corrected_1symbol, corrected_2symbol, uncorrectable)
template <typename scalar_t, int HEAD_SIZE>
__device__ __forceinline__ void decode_rs128_to_array_with_stats(
    const uint8_t* __restrict__ cache_ptr,
    float scale,
    scalar_t* __restrict__ decoded,
    int64_t* __restrict__ rs_stats    // Optional: nullptr to skip RS stats
) {
    constexpr int num_octuplets = HEAD_SIZE / 8;
    constexpr int remainder_count = HEAD_SIZE % 8;
    constexpr int full_rs_bytes = num_octuplets * 6;

    // Decode full RS(12,8) octuplets with error tracking
    #pragma unroll
    for (int oct = 0; oct < num_octuplets; oct++) {
        ecc::RSErrorType etype;
        ecc::int4_rs128_decode<scalar_t>(
            cache_ptr + oct * 6, scale, decoded + oct * 8, &etype);

        if (rs_stats) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&rs_stats[static_cast<int>(etype)]), 1ULL);
        }
    }

    // Decode shortened RS remainder with error tracking
    if constexpr (remainder_count > 0) {
        ecc::RSErrorType etype;
        ecc::int4_rs_shortened_decode<scalar_t>(
            cache_ptr + full_rs_bytes, remainder_count, scale,
            decoded + num_octuplets * 8, &etype);

        if (rs_stats) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&rs_stats[static_cast<int>(etype)]), 1ULL);
        }
    }
}

// Utility function for attention softmax.
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Broadcast to other threads.
  return VLLM_SHFL_SYNC(sum, 0);
}

// TODO(woosuk): Merge the last two dimensions of the grid.
// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float* k_scale, const float* v_scale,
    // Golay hybrid ECC parameters (nullptr if not using Golay)
    const int32_t* __restrict__ golay_syndrome_lut,  // [4096] syndrome lookup table
    int64_t* __restrict__ golay_stats,               // [5] atomic error counters
    int64_t* __restrict__ hamming_stats,             // [4] atomic error counters
    // Reed-Solomon RS(12,8) ECC parameters (nullptr if not using RS)
    int64_t* __restrict__ rs_stats,                  // [4] atomic error counters
    const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int seq_len = seq_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= seq_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_seq_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE;  // Note: This assumes THREAD_GROUP_SIZE
                                        // divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope =
      alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16 /
  // (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the thread group size is 4, then the first thread in
  // the group has 0, 4, 8, ... th vectors of the query, and the second thread
  // has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
  // q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();  // TODO(naed90): possible speedup if this is replaced with a
                    // memory wall right before we use q_vecs

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(cache_t);
  float qk_max = -FLT_MAX;

  // Golay hybrid detection (used in both K and V cache processing)
  constexpr bool IS_GOLAY_HYBRID = (KV_DTYPE == Fp8KVCacheDataType::kInt4GolayHybrid);
  // Reed-Solomon RS(12,8) detection (used in both K and V cache processing)
  constexpr bool IS_RS128 = (KV_DTYPE == Fp8KVCacheDataType::kInt4ReedSolomon);

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  // blocksparse specific vars
  int bs_block_offset;
  int q_bs_block_id;
  if constexpr (IS_BLOCK_SPARSE) {
    // const int num_blocksparse_blocks = DIVIDE_ROUND_UP(seq_len,
    // blocksparse_block_size);
    q_bs_block_id = (seq_len - 1) / blocksparse_block_size;
    if (blocksparse_head_sliding_step >= 0)
      // sliding on q heads
      bs_block_offset =
          (tp_rank * num_heads + head_idx) * blocksparse_head_sliding_step + 1;
    else
      // sliding on kv heads
      bs_block_offset = (tp_rank * num_kv_heads + kv_head_idx) *
                            (-blocksparse_head_sliding_step) +
                        1;
  }

  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    // For blocksparse attention: skip computation on blocks that are not
    // attended
    if constexpr (IS_BLOCK_SPARSE) {
      const int k_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      const bool is_remote =
          ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
      const bool is_local =
          (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
      if (!is_remote && !is_local) {
        for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
          const int physical_block_offset =
              (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
          const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

          if (thread_group_offset == 0) {
            // NOTE(linxihui): assign very large number to skipped tokens to
            // avoid contribution to the sumexp softmax normalizer. This will
            // not be used at computing sum(softmax*v) as the blocks will be
            // skipped.
            logits[token_idx - start_token_idx] = -FLT_MAX;
          }
        }
        continue;
      }
    }
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the thread group size is 4, then the first thread in
    // the group has 0, 4, 8, ... th vectors of the key, and the second thread
    // has 1, 5, 9, ... th vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset =
          (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

      // Pre-decode for Golay hybrid (decode entire K head once per token)
      // Array size is minimal (1) for non-Golay types to avoid register waste
      //
      // NOTE: Register pressure consideration for large HEAD_SIZE:
      // - HEAD_SIZE=64:  128 bytes/thread (FP16) - fits comfortably in registers
      // - HEAD_SIZE=128: 256 bytes/thread (FP16) - moderate register pressure
      // - HEAD_SIZE=256: 512 bytes/thread (FP16) - may cause register spilling
      //
      // For HEAD_SIZE >= 256, consider using shared memory cooperative decode
      // if register spilling becomes a performance issue. The current approach
      // prioritizes code simplicity and works well for common HEAD_SIZE values
      // (64, 128). For production workloads with HEAD_SIZE=256, profile and
      // consider shared memory optimization if needed.
      scalar_t decoded_k_golay[IS_GOLAY_HYBRID ? HEAD_SIZE : 1];
      if constexpr (IS_GOLAY_HYBRID) {
        constexpr int num_triplets = HEAD_SIZE / 3;
        constexpr int remainder_count = HEAD_SIZE % 3;
        constexpr int golay_bytes = num_triplets * 4;
        constexpr int hybrid_head_bytes = golay_bytes + remainder_count;

        // Cache layout for Golay hybrid:
        // [block_idx][head_idx][token_offset * hybrid_head_bytes]
        const uint8_t* k_head_ptr = reinterpret_cast<const uint8_t*>(k_cache) +
            physical_block_number * kv_block_stride +
            kv_head_idx * kv_head_stride +
            physical_block_offset * hybrid_head_bytes;

        decode_golay_hybrid_to_array_with_stats<scalar_t, HEAD_SIZE>(
            k_head_ptr, *k_scale, golay_syndrome_lut, decoded_k_golay,
            golay_stats, hamming_stats);
      }

      // Pre-decode for RS(12,8) (decode entire K head once per token)
      scalar_t decoded_k_rs[IS_RS128 ? HEAD_SIZE : 1];
      if constexpr (IS_RS128) {
        constexpr int num_octuplets = HEAD_SIZE / 8;
        constexpr int remainder_count = HEAD_SIZE % 8;
        constexpr int full_rs_bytes = num_octuplets * 6;
        constexpr int remainder_bytes = (remainder_count > 0) ? ((remainder_count + 4 + 1) / 2) : 0;
        constexpr int rs_head_bytes = full_rs_bytes + remainder_bytes;

        // Cache layout for RS:
        // [block_idx][head_idx][token_offset * rs_head_bytes]
        const uint8_t* k_head_ptr = reinterpret_cast<const uint8_t*>(k_cache) +
            physical_block_number * kv_block_stride +
            kv_head_idx * kv_head_stride +
            physical_block_offset * rs_head_bytes;

        decode_rs128_to_array_with_stats<scalar_t, HEAD_SIZE>(
            k_head_ptr, *k_scale, decoded_k_rs, rs_stats);
      }

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const cache_t* k_ptr =
            k_cache + physical_block_number * kv_block_stride +
            kv_head_idx * kv_head_stride + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;

        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          k_vecs[j] = *reinterpret_cast<const K_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        } else if constexpr (KV_DTYPE == Fp8KVCacheDataType::kInt4Ecc) {
          // ECC decode: Hamming(8,4) codeword -> INT4 -> dequantize
          Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          k_vecs[j] = ecc_scaled_convert<K_vec, Quant_vec>(
              k_vec_quant, *k_scale);
        } else if constexpr (KV_DTYPE == Fp8KVCacheDataType::kInt4Hamming74) {
          // ECC decode: Hamming(7,4) codeword -> INT4 -> dequantize
          Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          k_vecs[j] = h74_scaled_convert<K_vec, Quant_vec>(
              k_vec_quant, *k_scale);
        } else if constexpr (IS_GOLAY_HYBRID) {
          // Golay hybrid: read from pre-decoded array
          #pragma unroll
          for (int e = 0; e < VEC_SIZE; e++) {
            reinterpret_cast<scalar_t*>(&k_vecs[j])[e] =
                decoded_k_golay[vec_idx * VEC_SIZE + e];
          }
        } else if constexpr (IS_RS128) {
          // RS(12,8): read from pre-decoded array
          #pragma unroll
          for (int e = 0; e < VEC_SIZE; e++) {
            reinterpret_cast<scalar_t*>(&k_vecs[j])[e] =
                decoded_k_rs[vec_idx * VEC_SIZE + e];
          }
        } else {
          // Vector conversion from Quant_vec to K_vec (FP8).
          Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(
              k_vec_quant, *k_scale);
        }
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
                             q_vecs[thread_group_offset], k_vecs);
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= seq_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = VLLM_SHFL_SYNC(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits +
                            seq_idx * num_heads * max_num_partitions +
                            head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
                          head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using V_quant_vec = typename Vec<cache_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    // For blocksparse attention: skip computation on blocks that are not
    // attended
    if constexpr (IS_BLOCK_SPARSE) {
      int v_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      if (!((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) &&
          !((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
        continue;
      }
    }
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx -
                                                           start_token_idx));

    const cache_t* v_ptr = v_cache + physical_block_number * kv_block_stride +
                           kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec;

        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        } else if constexpr (KV_DTYPE == Fp8KVCacheDataType::kInt4Ecc) {
          // ECC decode: Hamming(8,4) codeword -> INT4 -> dequantize
          V_quant_vec v_quant_vec =
              *reinterpret_cast<const V_quant_vec*>(v_ptr + offset);
          v_vec = ecc_scaled_convert<V_vec, V_quant_vec>(v_quant_vec, *v_scale);
        } else if constexpr (KV_DTYPE == Fp8KVCacheDataType::kInt4Hamming74) {
          // ECC decode: Hamming(7,4) codeword -> INT4 -> dequantize
          V_quant_vec v_quant_vec =
              *reinterpret_cast<const V_quant_vec*>(v_ptr + offset);
          v_vec = h74_scaled_convert<V_vec, V_quant_vec>(v_quant_vec, *v_scale);
        } else if constexpr (IS_GOLAY_HYBRID) {
          // Golay hybrid: decode each token's needed element on-demand
          // For each element in v_vec (from V_VEC_SIZE consecutive tokens):
          // - Find the triplet containing row_idx
          // - Decode that triplet (or Hamming remainder)
          // - Extract the row_idx element
          //
          // NOTE: Stats are counted per unique (token, triplet) decode operation.
          // Each j corresponds to a different token, so each decode is a unique
          // codeword. This matches the K cache counting semantics where we count
          // per codeword decoded (not per value extracted).
          constexpr int num_triplets = HEAD_SIZE / 3;
          constexpr int remainder_count = HEAD_SIZE % 3;
          constexpr int golay_bytes = num_triplets * 4;
          constexpr int hybrid_head_bytes = golay_bytes + remainder_count;

          const int triplet_idx = row_idx / 3;
          const int elem_in_triplet = row_idx % 3;

          #pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            const int token_offset = physical_block_offset + j;
            const uint8_t* v_head_ptr = reinterpret_cast<const uint8_t*>(v_cache) +
                physical_block_number * kv_block_stride +
                kv_head_idx * kv_head_stride +
                token_offset * hybrid_head_bytes;

            if (triplet_idx < num_triplets) {
              // Golay triplet: decode 3 values, extract the one at row_idx % 3
              int32_t codeword = *reinterpret_cast<const int32_t*>(v_head_ptr + triplet_idx * 4);
              scalar_t v0, v1, v2;
              ecc::GolayErrorType etype;
              ecc::int4_golay_decode_triplet<scalar_t>(
                  codeword, *v_scale, golay_syndrome_lut, v0, v1, v2, &etype);
              reinterpret_cast<scalar_t*>(&v_vec)[j] =
                  (elem_in_triplet == 0) ? v0 : (elem_in_triplet == 1) ? v1 : v2;
              if (golay_stats) atomicAdd(reinterpret_cast<unsigned long long*>(&golay_stats[static_cast<int>(etype)]), 1ULL);
            } else {
              // Hamming remainder
              const int remainder_idx = row_idx - num_triplets * 3;
              ecc::ErrorType etype;
              reinterpret_cast<scalar_t*>(&v_vec)[j] = ecc::int4_ecc_decode<scalar_t>(
                  v_head_ptr[golay_bytes + remainder_idx], *v_scale, &etype);
              if (hamming_stats) atomicAdd(reinterpret_cast<unsigned long long*>(&hamming_stats[static_cast<int>(etype)]), 1ULL);
            }
          }
        } else if constexpr (IS_RS128) {
          // RS(12,8): decode each token's needed element on-demand
          // For each element in v_vec (from V_VEC_SIZE consecutive tokens):
          // - Find the octuplet containing row_idx
          // - Decode that octuplet (or shortened RS remainder)
          // - Extract the row_idx element
          constexpr int num_octuplets = HEAD_SIZE / 8;
          constexpr int remainder_count = HEAD_SIZE % 8;
          constexpr int full_rs_bytes = num_octuplets * 6;
          constexpr int remainder_bytes = (remainder_count > 0) ? ((remainder_count + 4 + 1) / 2) : 0;
          constexpr int rs_head_bytes = full_rs_bytes + remainder_bytes;

          const int octuplet_idx = row_idx / 8;
          const int elem_in_octuplet = row_idx % 8;

          #pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            const int token_offset = physical_block_offset + j;
            const uint8_t* v_head_ptr = reinterpret_cast<const uint8_t*>(v_cache) +
                physical_block_number * kv_block_stride +
                kv_head_idx * kv_head_stride +
                token_offset * rs_head_bytes;

            if (octuplet_idx < num_octuplets) {
              // RS octuplet: decode 8 values, extract the one at row_idx % 8
              scalar_t decoded[8];
              ecc::RSErrorType etype;
              ecc::int4_rs128_decode<scalar_t>(
                  v_head_ptr + octuplet_idx * 6, *v_scale, decoded, &etype);
              reinterpret_cast<scalar_t*>(&v_vec)[j] = decoded[elem_in_octuplet];
              if (rs_stats) atomicAdd(reinterpret_cast<unsigned long long*>(&rs_stats[static_cast<int>(etype)]), 1ULL);
            } else if constexpr (remainder_count > 0) {
              // Shortened RS remainder
              scalar_t decoded_rem[8];  // Max 7 values
              ecc::RSErrorType etype;
              ecc::int4_rs_shortened_decode<scalar_t>(
                  v_head_ptr + full_rs_bytes, remainder_count, *v_scale, decoded_rem, &etype);
              const int remainder_idx = row_idx - num_octuplets * 8;
              reinterpret_cast<scalar_t*>(&v_vec)[j] = decoded_rem[remainder_idx];
              if (rs_stats) atomicAdd(reinterpret_cast<unsigned long long*>(&rs_stats[static_cast<int>(etype)]), 1ULL);
            }
          }
        } else {
          // Vector conversion from V_quant_vec to V_vec (FP8).
          V_quant_vec v_quant_vec =
              *reinterpret_cast<const V_quant_vec*>(v_ptr + offset);
          v_vec = fp8::scaled_convert<V_vec, V_quant_vec, KV_DTYPE>(v_quant_vec,
                                                                    *v_scale);
        }
        if (block_idx == num_seq_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the
          // context, we should explicitly zero out the values since they may
          // contain NaNs. See
          // https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for
  // logits is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr =
        out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

// Grid: (num_heads, num_seqs, 1).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,           // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float* k_scale, const float* v_scale,
    // Golay hybrid ECC parameters (nullptr if not using Golay hybrid)
    const int32_t* __restrict__ golay_syndrome_lut,  // [4096] syndrome lookup table
    int64_t* __restrict__ golay_stats,               // [5] atomic error counters
    int64_t* __restrict__ hamming_stats,             // [4] atomic error counters
    // Reed-Solomon RS(12,8) ECC parameters (nullptr if not using RS)
    int64_t* __restrict__ rs_stats,                  // [4] atomic error counters
    const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         KV_DTYPE, IS_BLOCK_SPARSE>(
      /* exp_sums */ nullptr, /* max_logits */ nullptr, out, q, k_cache,
      v_cache, num_kv_heads, scale, block_tables, seq_lens,
      max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,
      kv_head_stride, k_scale, v_scale,
      golay_syndrome_lut, golay_stats, hamming_stats, rs_stats,
      tp_rank, blocksparse_local_blocks,
      blocksparse_vert_stride, blocksparse_block_size,
      blocksparse_head_sliding_step);
}

// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,       // [num_seqs, num_heads,
                                          // max_num_partitions]
    scalar_t* __restrict__ tmp_out,       // [num_seqs, num_heads,
                                          // max_num_partitions, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float* k_scale, const float* v_scale,
    // Golay hybrid ECC parameters (nullptr if not using Golay hybrid)
    const int32_t* __restrict__ golay_syndrome_lut,  // [4096] syndrome lookup table
    int64_t* __restrict__ golay_stats,               // [5] atomic error counters
    int64_t* __restrict__ hamming_stats,             // [4] atomic error counters
    // Reed-Solomon RS(12,8) ECC parameters (nullptr if not using RS)
    int64_t* __restrict__ rs_stats,                  // [4] atomic error counters
    const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         KV_DTYPE, IS_BLOCK_SPARSE, PARTITION_SIZE>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, max_num_blocks_per_seq, alibi_slopes, q_stride,
      kv_block_stride, kv_head_stride, k_scale, v_scale,
      golay_syndrome_lut, golay_stats, hamming_stats, rs_stats,
      tp_rank, blocksparse_local_blocks, blocksparse_vert_stride,
      blocksparse_block_size, blocksparse_head_sliding_step);
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_reduce_kernel(
    scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads,
                                           // max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                           // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads,
                                           // max_num_partitions, head_size]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_partitions) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int seq_len = seq_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    // No need to reduce. Only copy tmp_out to out.
    scalar_t* out_ptr =
        out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const scalar_t* tmp_out_ptr =
        tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  // Size: 2 * num_partitions.
  extern __shared__ char shared_mem[];
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // Load max logits to shared memory.
  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  const float* max_logits_ptr = max_logits +
                                seq_idx * num_heads * max_num_partitions +
                                head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = fmaxf(max_logit, l);
  }
  __syncthreads();

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  __syncthreads();
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  // Broadcast the max value to all threads.
  max_logit = VLLM_SHFL_SYNC(max_logit, 0);

  // Load rescaled exp sums to shared memory.
  float* shared_exp_sums =
      reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
  const float* exp_sums_ptr = exp_sums +
                              seq_idx * num_heads * max_num_partitions +
                              head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  __syncthreads();
  global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const scalar_t* tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE;
  scalar_t* out_ptr =
      out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] *
             inv_global_exp_sum;
    }
    from_float(out_ptr[i], acc);
  }
}

}  // namespace vllm

#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
