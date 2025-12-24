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
#include "attention_kernels.cuh"
#include "../cuda_compat.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(                     \
      ((void*)vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE,        \
                                              BLOCK_SIZE, NUM_THREADS,      \
                                              KV_DTYPE, IS_BLOCK_SPARSE>),  \
      shared_mem_size);                                                     \
  vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,        \
                                  NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE>   \
      <<<grid, block, shared_mem_size, stream>>>(                           \
          out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, \
          scale, block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq,    \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,      \
          k_scale_ptr, v_scale_ptr,                                         \
          golay_syndrome_lut_ptr, golay_stats_ptr, hamming_stats_ptr,       \
          rs_stats_ptr, tp_rank, blocksparse_local_blocks,                  \
          blocksparse_vert_stride, blocksparse_block_size,                  \
          blocksparse_head_sliding_step);

// TODO(woosuk): Tune NUM_THREADS.
template <typename T, typename CACHE_T, int BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int NUM_THREADS = 128>
void paged_attention_v1_launcher(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int max_seq_len,
    const std::optional<torch::Tensor>& alibi_slopes, torch::Tensor& k_scale,
    torch::Tensor& v_scale,
    const std::optional<torch::Tensor>& golay_syndrome_lut,
    const std::optional<torch::Tensor>& golay_stats,
    const std::optional<torch::Tensor>& hamming_stats,
    const std::optional<torch::Tensor>& rs_stats,
    const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* value_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();
  const float* k_scale_ptr = reinterpret_cast<const float*>(k_scale.data_ptr());
  const float* v_scale_ptr = reinterpret_cast<const float*>(v_scale.data_ptr());

  // Golay hybrid ECC parameters (optional)
  const int32_t* golay_syndrome_lut_ptr =
      golay_syndrome_lut
          ? reinterpret_cast<const int32_t*>(golay_syndrome_lut.value().data_ptr())
          : nullptr;
  int64_t* golay_stats_ptr =
      golay_stats
          ? reinterpret_cast<int64_t*>(golay_stats.value().data_ptr())
          : nullptr;
  int64_t* hamming_stats_ptr =
      hamming_stats
          ? reinterpret_cast<int64_t*>(hamming_stats.value().data_ptr())
          : nullptr;
  int64_t* rs_stats_ptr =
      rs_stats
          ? reinterpret_cast<int64_t*>(rs_stats.value().data_ptr())
          : nullptr;

  const int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_seq_len =
      DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_seq_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(NUM_THREADS);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // NOTE: Reduced to GPT-2 (64) and LLaMA-3.1-8B (128) for faster builds
    case 64:
      LAUNCH_PAGED_ATTENTION_V1(64);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V1(128);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size,
                  ". Only 64 (GPT-2) and 128 (LLaMA) compiled.");
      break;
  }
}

#define CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, KV_DTYPE, IS_BLOCK_SPARSE)  \
  paged_attention_v1_launcher<T, CACHE_T, BLOCK_SIZE, KV_DTYPE,              \
                              IS_BLOCK_SPARSE>(                              \
      out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, \
      seq_lens, max_seq_len, alibi_slopes, k_scale, v_scale,                 \
      golay_syndrome_lut, golay_stats, hamming_stats, rs_stats, tp_rank,     \
      blocksparse_local_blocks, blocksparse_vert_stride,                     \
      blocksparse_block_size, blocksparse_head_sliding_step);

#define CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE) \
  if (is_block_sparse) {                                                   \
    CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, true);       \
  } else {                                                                 \
    CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, false);      \
  }

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V1_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DTYPE)         \
  switch (block_size) {                                           \
    case 16:                                                      \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 16, KV_DTYPE);        \
      break;                                                      \
    case 32:                                                      \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 32, KV_DTYPE);        \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size,  \
                  ". Only 16 and 32 compiled.");                  \
      break;                                                      \
  }

void paged_attention_v1(
    torch::Tensor& out,    // [num_seqs, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,       // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,  // [num_heads]
    double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale,
    const std::optional<torch::Tensor>& golay_syndrome_lut,  // [4096] int32
    const std::optional<torch::Tensor>& golay_stats,         // [5] int64
    const std::optional<torch::Tensor>& hamming_stats,       // [4] int64
    const std::optional<torch::Tensor>& rs_stats,            // [4] int64
    const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  const bool is_block_sparse = (blocksparse_vert_stride > 1);

  DISPATCH_BY_KV_CACHE_DTYPE(query.dtype(), kv_cache_dtype,
                             CALL_V1_LAUNCHER_BLOCK_SIZE)
}

#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
