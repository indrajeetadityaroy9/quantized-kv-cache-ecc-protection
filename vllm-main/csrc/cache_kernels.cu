#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Optional.h>

#include "cuda_utils.h"
#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "quantization/vectorization_utils.cuh"

#ifdef USE_ROCM
  #include "quantization/w8a8/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#endif

// ECC-protected INT4 KV cache
#include "ecc/hamming84.cuh"
#include "ecc/hamming74.cuh"
#include "ecc/golay2412.cuh"
#include "ecc/golay_syndrome_table.cuh"
#include "ecc/reed_solomon.cuh"
#include "ecc/fault_injection.cuh"
#include "ecc/ecc_stats.cuh"

#include <algorithm>
#include <cassert>
#include <cfloat>

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
typedef __hip_bfloat16 __nv_bfloat16;
#endif

void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    TORCH_CHECK(src_device.index() == dst_device.index(),
                "src and dst must be on the same GPU");
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  // NOTE(youkaichao): keep in mind that `block_mapping` should be
  // a cpu tensor, otherwise every `item` call will require a gpu-cpu
  // synchronization.
  TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");

  char* src_ptr = static_cast<char*>(src.data_ptr());
  char* dst_ptr = static_cast<char*>(dst.data_ptr());

  // We use the stride instead of numel in case the cache is padded for memory
  // alignment reasons, we assume the blocks data (inclusive of any padding)
  // is contiguous in memory
  const int64_t block_size_in_bytes = src.element_size() * src.stride(0);
  const at::cuda::OptionalCUDAGuard device_guard(
      src_device.is_cuda() ? src_device : dst_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // NOTE(woosuk): This can be slow if the number of blocks is large.
  const int64_t num_blocks = block_mapping.size(0);
  for (size_t i = 0; i < num_blocks; i++) {
    int64_t src_block_number = block_mapping[i][0].item<int64_t>();
    int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    cudaMemcpyAsync(dst_ptr + dst_offset, src_ptr + src_offset,
                    block_size_in_bytes, memcpy_type, stream);
  }
}

namespace vllm {

// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__global__ void copy_blocks_kernel(int64_t* key_cache_ptrs,
                                   int64_t* value_cache_ptrs,
                                   const int64_t* __restrict__ block_mapping,
                                   const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache =
      reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

// Kernel for MLA, which works on a single joint kv_cache
// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__global__ void copy_blocks_mla_kernel(
    int64_t* cache_ptrs, const int64_t* __restrict__ block_mapping,
    const int mem_footprint_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;
  scalar_t* cache = reinterpret_cast<scalar_t*>(cache_ptrs[layer_idx]);
  int64_t src_block = block_mapping[2 * pair_idx];
  int64_t dst_block = block_mapping[2 * pair_idx + 1];
  int64_t src_offset = src_block * mem_footprint_per_block;
  int64_t dst_offset = dst_block * mem_footprint_per_block;
  for (int i = threadIdx.x; i < mem_footprint_per_block; i += blockDim.x) {
    cache[dst_offset + i] = cache[src_offset + i];
  }
}

}  // namespace vllm

// Note: the key_caches and value_caches vectors are constant but
// not the Tensors they contain. The vectors need to be const refs
// in order to satisfy pytorch's C++ operator registration code.
void copy_blocks(std::vector<torch::Tensor> const& key_caches,
                 std::vector<torch::Tensor> const& value_caches,
                 const torch::Tensor& block_mapping) {
  int num_layers = key_caches.size();
  TORCH_CHECK(num_layers == value_caches.size());
  if (num_layers == 0) {
    return;
  }
  torch::Device cache_device = key_caches[0].device();
  TORCH_CHECK(cache_device.is_cuda());

  // Create data structures for the kernel.
  // Create an array of pointers to the key and value caches.
  int64_t key_cache_ptrs[num_layers];
  int64_t value_cache_ptrs[num_layers];
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    key_cache_ptrs[layer_idx] =
        reinterpret_cast<int64_t>(key_caches[layer_idx].data_ptr());
    value_cache_ptrs[layer_idx] =
        reinterpret_cast<int64_t>(value_caches[layer_idx].data_ptr());
  }

  // block_mapping is a 2D tensor with shape (num_pairs, 2).
  int num_pairs = block_mapping.size(0);

  // Move the data structures to the GPU.
  // NOTE: This synchronizes the CPU and GPU.
  torch::Tensor key_cache_ptrs_tensor =
      torch::from_blob(key_cache_ptrs, {num_layers}, torch::kInt64)
          .to(cache_device);
  torch::Tensor value_cache_ptrs_tensor =
      torch::from_blob(value_cache_ptrs, {num_layers}, torch::kInt64)
          .to(cache_device);

  // Launch the kernel.
  const int numel_per_block = key_caches[0][0].numel();
  dim3 grid(num_layers, num_pairs);
  dim3 block(std::min(1024, numel_per_block));
  const at::cuda::OptionalCUDAGuard device_guard(cache_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_AND_BYTE_TYPES(
      key_caches[0].scalar_type(), "copy_blocks_kernel", ([&] {
        vllm::copy_blocks_kernel<scalar_t><<<grid, block, 0, stream>>>(
            key_cache_ptrs_tensor.data_ptr<int64_t>(),
            value_cache_ptrs_tensor.data_ptr<int64_t>(),
            block_mapping.data_ptr<int64_t>(), numel_per_block);
      }));
}

// copy blocks kernel for MLA (assumes a joint KV-cache)
void copy_blocks_mla(std::vector<torch::Tensor> const& kv_caches,
                     const torch::Tensor& block_mapping) {
  int num_layers = kv_caches.size();
  if (num_layers == 0) {
    return;
  }
  torch::Device cache_device = kv_caches[0].device();
  TORCH_CHECK(cache_device.is_cuda(), "kv_cache must be on CUDA");

  std::vector<int64_t> cache_ptrs(num_layers);
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    cache_ptrs[layer_idx] =
        reinterpret_cast<int64_t>(kv_caches[layer_idx].data_ptr());
  }
  torch::Tensor cache_ptrs_tensor =
      torch::from_blob(cache_ptrs.data(), {num_layers}, torch::kInt64)
          .to(cache_device);

  int num_pairs = block_mapping.size(0);
  // We use the stride instead of numel in case the cache is padded for memory
  // alignment reasons, we assume the blocks data (inclusive of any padding)
  // is contiguous in memory
  int mem_footprint_per_block = kv_caches[0].stride(0);
  dim3 grid(num_layers, num_pairs);
  dim3 block(std::min(1024, mem_footprint_per_block));
  const at::cuda::OptionalCUDAGuard device_guard(cache_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_AND_BYTE_TYPES(
      kv_caches[0].scalar_type(), "copy_blocks_mla_kernel", ([&] {
        vllm::copy_blocks_mla_kernel<scalar_t><<<grid, block, 0, stream>>>(
            cache_ptrs_tensor.data_ptr<int64_t>(),
            block_mapping.data_ptr<int64_t>(), mem_footprint_per_block);
      }));
}

namespace vllm {

// Used to copy/convert one element
template <typename OutT, typename InT, Fp8KVCacheDataType kv_dt>
struct CopyWithScaleOp {
  float scale;

  __device__ __forceinline__ void operator()(OutT& dst, const InT src) const {
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      dst = static_cast<OutT>(src);
    } else if constexpr (kv_dt == Fp8KVCacheDataType::kInt4Ecc) {
      // INT4 quantization + Hamming(8,4) SECDED encoding
      // Scale is absmax/7.0 (passed in from per-block computation)
      dst = ecc::int4_ecc_encode<InT>(src, scale);
    } else if constexpr (kv_dt == Fp8KVCacheDataType::kInt4Hamming74) {
      // INT4 quantization + Hamming(7,4) SEC encoding
      // Scale is absmax/7.0 (passed in from per-block computation)
      dst = ecc::int4_h74_encode<InT>(src, scale);
    } else if constexpr (kv_dt == Fp8KVCacheDataType::kInt4Golay) {
      // NOTE: Golay(24,12) encodes 3 INT4 values into 1 int32 codeword.
      // This 3:1 mapping doesn't fit the 1:1 CopyWithScaleOp pattern.
      // Golay encoding requires reshape_and_cache_golay_kernel instead.
      // This branch should never be called at runtime - only included for template
      // instantiation compatibility. Use reshape_and_cache_golay_hybrid() instead.
      dst = OutT{0};  // Placeholder - this path is not used
    } else if constexpr (kv_dt == Fp8KVCacheDataType::kInt4GolayHybrid) {
      // Hybrid Golay+Hamming requires specialized kernel for triplet processing.
      // This branch should never be called at runtime - only included for template
      // instantiation compatibility. Use reshape_and_cache_golay_hybrid() instead.
      dst = OutT{0};  // Placeholder - this path is not used
    } else {
      dst = fp8::scaled_convert<OutT, InT, kv_dt>(src, scale);
    }
  }
};

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x,
                                         // block_size, x]
    cache_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size,
                                         // block_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x,
    const float* k_scale, const float* v_scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int h_block_count = head_size / x;  // head_size//x

  const int h_block_idx = threadIdx.x;
  if (h_block_idx >= num_heads * h_block_count) {
    return;
  }

  const int head_idx = h_block_idx / h_block_count;
  const int h_block = h_block_idx % h_block_count;

  const scalar_t* __restrict__ key_src =
      key + token_idx * key_stride + head_idx * head_size + h_block * x;
  const int64_t src_value_start =
      token_idx * value_stride + head_idx * head_size + h_block * x;

  cache_t* __restrict__ key_dst =
      key_cache + block_idx * num_heads * h_block_count * block_size * x +
      head_idx * h_block_count * block_size * x + h_block * block_size * x +
      block_offset * x;
  const int64_t tgt_value_start =
      block_idx * num_heads * h_block_count * x * block_size +
      head_idx * h_block_count * x * block_size + h_block * x * block_size +
      block_offset;

  constexpr int VEC_SIZE = (sizeof(scalar_t) == 2) ? 8 : 4;
  float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *k_scale;
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
  float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *v_scale;
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};

  vectorize_with_alignment<VEC_SIZE>(key_src, key_dst, x, 0, 1, k_op);

  const scalar_t* __restrict__ value_src = value + src_value_start;
  cache_t* __restrict__ value_dst = value_cache + tgt_value_start;
#pragma unroll
  for (int i = 0; i < x; i++) {
    v_op(value_dst[i * block_size], value_src[i]);
  }
}

// Specialized kernel for hybrid Golay(24,12) + Hamming(8,4) encoding
// Golay encodes 3 INT4 values into 1 int32 codeword (triplet processing)
// Hamming handles remainder values (head_size % 3)
//
// Memory layout per head:
//   [golay_0 (4B)] [golay_1 (4B)] ... [golay_N-1 (4B)] [hamming_0] [hamming_1?]
// where N = head_size / 3, remainder = head_size % 3
template <typename scalar_t, int HEAD_SIZE>
__global__ void reshape_and_cache_golay_hybrid_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    uint8_t* __restrict__ key_cache,     // [num_blocks, num_heads, hybrid_bytes, block_size]
    uint8_t* __restrict__ value_cache,   // [num_blocks, num_heads, hybrid_bytes, block_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride, const int value_stride,
    const int num_heads, const int block_size,
    const float* k_scale, const float* v_scale) {

  // Compile-time constants for hybrid layout
  constexpr int num_triplets = HEAD_SIZE / 3;
  constexpr int remainder_count = HEAD_SIZE % 3;
  constexpr int golay_bytes = num_triplets * 4;  // 4 bytes per int32 codeword
  constexpr int hybrid_head_bytes = golay_bytes + remainder_count;
  constexpr int work_items_per_head = num_triplets + remainder_count;

  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const float k_scale_val = *k_scale;
  const float v_scale_val = *v_scale;

  // Each thread handles one work item (triplet or remainder)
  const int total_work_items = num_heads * work_items_per_head;

  for (int work_idx = threadIdx.x; work_idx < total_work_items; work_idx += blockDim.x) {
    const int head_idx = work_idx / work_items_per_head;
    const int item_in_head = work_idx % work_items_per_head;

    if (head_idx >= num_heads) continue;

    // Source pointers for this head
    const scalar_t* __restrict__ k_src =
        key + token_idx * key_stride + head_idx * HEAD_SIZE;
    const scalar_t* __restrict__ v_src =
        value + token_idx * value_stride + head_idx * HEAD_SIZE;

    // Destination pointers for this head in cache
    // Layout: [num_blocks, num_heads, hybrid_head_bytes, block_size]
    uint8_t* __restrict__ k_dst =
        key_cache + block_idx * num_heads * hybrid_head_bytes * block_size
                  + head_idx * hybrid_head_bytes * block_size
                  + block_offset;
    uint8_t* __restrict__ v_dst =
        value_cache + block_idx * num_heads * hybrid_head_bytes * block_size
                    + head_idx * hybrid_head_bytes * block_size
                    + block_offset;

    if (item_in_head < num_triplets) {
      // Golay encode triplet (3 values -> 1 int32 codeword)
      const int triplet_idx = item_in_head;
      const int src_offset = triplet_idx * 3;
      const int dst_byte_offset = triplet_idx * 4;  // int32 = 4 bytes

      // Encode K triplet
      int32_t k_codeword = ecc::int4_golay_encode_triplet<scalar_t>(
          k_src[src_offset], k_src[src_offset + 1], k_src[src_offset + 2],
          k_scale_val);

      // Encode V triplet
      int32_t v_codeword = ecc::int4_golay_encode_triplet<scalar_t>(
          v_src[src_offset], v_src[src_offset + 1], v_src[src_offset + 2],
          v_scale_val);

      // Store codeword bytes at strided positions (matching cache layout)
      // Layout: [num_blocks, num_heads, hybrid_head_bytes, block_size]
      // Each byte must be stored at stride block_size in the hybrid_bytes dimension
      k_dst[(dst_byte_offset + 0) * block_size] = static_cast<uint8_t>(k_codeword & 0xFF);
      k_dst[(dst_byte_offset + 1) * block_size] = static_cast<uint8_t>((k_codeword >> 8) & 0xFF);
      k_dst[(dst_byte_offset + 2) * block_size] = static_cast<uint8_t>((k_codeword >> 16) & 0xFF);
      k_dst[(dst_byte_offset + 3) * block_size] = static_cast<uint8_t>((k_codeword >> 24) & 0xFF);

      v_dst[(dst_byte_offset + 0) * block_size] = static_cast<uint8_t>(v_codeword & 0xFF);
      v_dst[(dst_byte_offset + 1) * block_size] = static_cast<uint8_t>((v_codeword >> 8) & 0xFF);
      v_dst[(dst_byte_offset + 2) * block_size] = static_cast<uint8_t>((v_codeword >> 16) & 0xFF);
      v_dst[(dst_byte_offset + 3) * block_size] = static_cast<uint8_t>((v_codeword >> 24) & 0xFF);

    } else if constexpr (remainder_count > 0) {
      // Hamming encode remainder (1 value -> 1 uint8 codeword)
      const int rem_idx = item_in_head - num_triplets;
      const int src_offset = num_triplets * 3 + rem_idx;
      const int dst_byte_offset = golay_bytes + rem_idx;

      // Encode K remainder with Hamming(8,4)
      uint8_t k_codeword = ecc::int4_ecc_encode<scalar_t>(k_src[src_offset], k_scale_val);
      // Encode V remainder with Hamming(8,4)
      uint8_t v_codeword = ecc::int4_ecc_encode<scalar_t>(v_src[src_offset], v_scale_val);

      // Store codewords
      k_dst[dst_byte_offset * block_size] = k_codeword;
      v_dst[dst_byte_offset * block_size] = v_codeword;
    }
  }
}

// Specialized kernel for Reed-Solomon RS(12,8) encoding
// RS(12,8) encodes 8 INT4 values (32 bits) into 6 bytes (48 bits)
// Provides 2-symbol error correction with 50% overhead
//
// Memory layout per head:
//   [rs_octuplet_0 (6B)] [rs_octuplet_1 (6B)] ... [rs_N-1 (6B)] [shortened_remainder?]
// where N = head_size / 8, remainder = head_size % 8
// remainder_bytes = ceil((remainder + 4) / 2) if remainder > 0
template <typename scalar_t, int HEAD_SIZE>
__global__ void reshape_and_cache_rs_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    uint8_t* __restrict__ key_cache,     // [num_blocks, num_heads, rs_bytes, block_size]
    uint8_t* __restrict__ value_cache,   // [num_blocks, num_heads, rs_bytes, block_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride, const int value_stride,
    const int num_heads, const int block_size,
    const float* k_scale, const float* v_scale) {

  // Compile-time constants for RS layout
  constexpr int num_octuplets = HEAD_SIZE / 8;
  constexpr int remainder_count = HEAD_SIZE % 8;
  constexpr int full_rs_bytes = num_octuplets * 6;  // 6 bytes per RS(12,8) codeword
  // Shortened RS for remainder: 4 parity + remainder data = (remainder+4) nibbles
  // Bytes = ceil((remainder+4)/2)
  constexpr int remainder_bytes = (remainder_count > 0) ? ((remainder_count + 4 + 1) / 2) : 0;
  constexpr int rs_head_bytes = full_rs_bytes + remainder_bytes;
  constexpr int work_items_per_head = num_octuplets + ((remainder_count > 0) ? 1 : 0);

  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const float k_scale_val = *k_scale;
  const float v_scale_val = *v_scale;

  // Each thread handles one work item (octuplet or remainder)
  const int total_work_items = num_heads * work_items_per_head;

  for (int work_idx = threadIdx.x; work_idx < total_work_items; work_idx += blockDim.x) {
    const int head_idx = work_idx / work_items_per_head;
    const int item_in_head = work_idx % work_items_per_head;

    if (head_idx >= num_heads) continue;

    // Source pointers for this head
    const scalar_t* __restrict__ k_src =
        key + token_idx * key_stride + head_idx * HEAD_SIZE;
    const scalar_t* __restrict__ v_src =
        value + token_idx * value_stride + head_idx * HEAD_SIZE;

    // Destination pointers for this head in cache
    // Layout: [num_blocks, num_heads, rs_head_bytes, block_size]
    uint8_t* __restrict__ k_dst =
        key_cache + block_idx * num_heads * rs_head_bytes * block_size
                  + head_idx * rs_head_bytes * block_size
                  + block_offset;
    uint8_t* __restrict__ v_dst =
        value_cache + block_idx * num_heads * rs_head_bytes * block_size
                    + head_idx * rs_head_bytes * block_size
                    + block_offset;

    if (item_in_head < num_octuplets) {
      // RS(12,8) encode octuplet (8 values -> 6 bytes)
      const int oct_idx = item_in_head;
      const int src_offset = oct_idx * 8;
      const int dst_byte_offset = oct_idx * 6;

      // Encode K octuplet to temporary buffer
      uint8_t k_rs_bytes[6];
      ecc::int4_rs128_encode<scalar_t>(k_src + src_offset, k_scale_val, k_rs_bytes);

      // Encode V octuplet to temporary buffer
      uint8_t v_rs_bytes[6];
      ecc::int4_rs128_encode<scalar_t>(v_src + src_offset, v_scale_val, v_rs_bytes);

      // Store bytes at strided positions (matching cache layout)
      // Each byte stored at stride block_size in the rs_bytes dimension
      #pragma unroll
      for (int b = 0; b < 6; b++) {
        k_dst[(dst_byte_offset + b) * block_size] = k_rs_bytes[b];
        v_dst[(dst_byte_offset + b) * block_size] = v_rs_bytes[b];
      }

    } else if constexpr (remainder_count > 0) {
      // Shortened RS encode for remainder (remainder values -> ceil((remainder+4)/2) bytes)
      const int src_offset = num_octuplets * 8;
      const int dst_byte_offset = full_rs_bytes;

      // Encode K remainder with shortened RS
      uint8_t k_rem_bytes[4];  // Max 4 bytes for remainder_count <= 4
      ecc::int4_rs_shortened_encode<scalar_t>(k_src + src_offset, remainder_count, k_scale_val, k_rem_bytes);

      // Encode V remainder with shortened RS
      uint8_t v_rem_bytes[4];
      ecc::int4_rs_shortened_encode<scalar_t>(v_src + src_offset, remainder_count, v_scale_val, v_rem_bytes);

      // Store remainder bytes at strided positions
      #pragma unroll
      for (int b = 0; b < remainder_bytes; b++) {
        k_dst[(dst_byte_offset + b) * block_size] = k_rem_bytes[b];
        v_dst[(dst_byte_offset + b) * block_size] = v_rem_bytes[b];
      }
    }
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // NHD or HND, shape see comments below
    cache_t* __restrict__ value_cache,   // same above
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t block_stride, const int64_t page_stride,
    const int64_t head_stride, const int64_t key_stride,
    const int64_t value_stride, const int num_heads, const int head_size,
    const int block_size, const float* k_scale, const float* v_scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n_elems = num_heads * head_size;

  // pointers to the beginning of the source row for this token.
  const scalar_t* __restrict__ key_src = key + token_idx * key_stride;
  const scalar_t* __restrict__ value_src = value + token_idx * value_stride;

  // find the start position inside the kv-cache for this token.
  cache_t* __restrict__ key_dst =
      key_cache + block_idx * block_stride + block_offset * page_stride;
  cache_t* __restrict__ value_dst =
      value_cache + block_idx * block_stride + block_offset * page_stride;

  // this is true for the NHD layout where `head_stride == head_size`
  const bool is_contiguous_heads = (head_stride == head_size);

  float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *k_scale;
  float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *v_scale;
  constexpr int VEC_SIZE = (sizeof(scalar_t) == 2) ? 8 : 4;
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};
  if (is_contiguous_heads) {
    // NHD layout
    // kv cache: [num_blocks, block_size, num_heads, head_size]
    vectorize_with_alignment<VEC_SIZE>(key_src, key_dst, n_elems, threadIdx.x,
                                       blockDim.x, k_op);

    vectorize_with_alignment<VEC_SIZE>(value_src, value_dst, n_elems,
                                       threadIdx.x, blockDim.x, v_op);

  } else {
    // HND layout: heads are strided, but each head_size segment is contiguous
    // kv cache: [num_blocks, num_heads, block_size, head_size]
    const int lane = threadIdx.x & 31;     // 0..31 within warp
    const int warp_id = threadIdx.x >> 5;  // warp index within block
    const int warps_per_block = blockDim.x >> 5;

    for (int head = warp_id; head < num_heads; head += warps_per_block) {
      const scalar_t* __restrict__ k_src_h = key_src + head * head_size;
      const scalar_t* __restrict__ v_src_h = value_src + head * head_size;

      cache_t* __restrict__ k_dst_h =
          key_dst + static_cast<int64_t>(head) * head_stride;
      cache_t* __restrict__ v_dst_h =
          value_dst + static_cast<int64_t>(head) * head_stride;

      // within each head, let the 32 threads of the warp perform the vector
      // copy
      vectorize_with_alignment<VEC_SIZE>(k_src_h, k_dst_h, head_size, lane, 32,
                                         k_op);

      vectorize_with_alignment<VEC_SIZE>(v_src_h, v_dst_h, head_size, lane, 32,
                                         v_op);
    }
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         //
) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  auto copy = [&](const scalar_t* __restrict__ src, cache_t* __restrict__ dst,
                  int src_stride, int dst_stride, int size, int offset) {
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
      const int64_t src_idx = token_idx * src_stride + i;
      const int64_t dst_idx =
          block_idx * block_stride + block_offset * entry_stride + i + offset;
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        dst[dst_idx] = src[src_idx];
      } else {
        dst[dst_idx] =
            fp8::scaled_convert<cache_t, scalar_t, kv_dt>(src[src_idx], *scale);
      }
    }
  };

  copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);
  copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_ds_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         //
) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int64_t dst_idx_start =
      block_idx * block_stride + block_offset * entry_stride;

  // For the NoPE part, each tile of 128 elements is handled by half of one warp
  // (16 threads). There are 4 total tiles, so 2 warps (64 threads).
  // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
  // The RoPE part (last 64 elements) is handled by another 1 warp (32 threads).
  // So in total, we use 3 warps (96 threads) per block.

  // Cast kv_cache to 16_bit for RoPE values
  scalar_t* kv_cache_16bit =
      reinterpret_cast<scalar_t*>(&kv_cache[dst_idx_start]);

  // The last warp handles the RoPE part
  if (threadIdx.x >= 64) {
    // Each thread handles two elements of RoPE
    const int8_t pe_idx_start = (threadIdx.x - 64) * 2;
    const int64_t src_idx = token_idx * k_pe_stride + pe_idx_start;
    // Vectorized load of two 16-bit values, performed as one 32-bit load
    const int32_t vals = *reinterpret_cast<const int32_t*>(&k_pe[src_idx]);
    // RoPE values start after the packed 8-bit NoPE values and the
    // 32-bit scales
    const int64_t dst_idx = kv_lora_rank / 2 + 8 + pe_idx_start;
    // Vectorized store of two 16-bit values, performed as one 32-bit store
    *reinterpret_cast<int32_t*>(&kv_cache_16bit[dst_idx]) = vals;
    return;
  }

  // The first two warps handle the NoPE part
  const int8_t warp_idx = threadIdx.x >> 5;
  const int8_t lane_idx = threadIdx.x & 31;
  const int8_t tile_idx = warp_idx * 2 + (lane_idx >> 4);

  // Each thread handles 8 elements of NoPE
  // Load the NoPE elements for this thread into registers
  const int64_t src_idx_start = token_idx * kv_c_stride + (threadIdx.x * 8);
  // Vectorized load of eight 16-bit values, performed as an int4 load
  const int4 vals_i4 = *reinterpret_cast<const int4*>(&kv_c[src_idx_start]);
  const scalar_t* vals = reinterpret_cast<const scalar_t*>(&vals_i4);

  // Max absolute value of this thread's elements
  float max_abs = fmaxf(fmaxf(fmaxf(fabsf(vals[0]), fabsf(vals[1])),
                              fmaxf(fabsf(vals[2]), fabsf(vals[3]))),
                        fmaxf(fmaxf(fabsf(vals[4]), fabsf(vals[5])),
                              fmaxf(fabsf(vals[6]), fabsf(vals[7]))));

  // Warp-level reduction to find the max absolute value in each half-warp
#pragma unroll
  for (int offset = 8; offset > 0; offset /= 2) {
    max_abs = fmaxf(max_abs, VLLM_SHFL_XOR_SYNC_WIDTH(max_abs, offset, 16));
  }

  // Compute the scale for the tile
  float tile_scale = max_abs / 448.f;
  tile_scale = fmaxf(tile_scale, FLT_MIN);

  // The first lane of each half-warp writes the scale to kv_cache
  if ((lane_idx == 0) || (lane_idx == 16)) {
    float* kv_cache_32bit = reinterpret_cast<float*>(&kv_cache[dst_idx_start]);
    const uint64_t dst_idx = kv_lora_rank / 4 + tile_idx;
    kv_cache_32bit[dst_idx] = tile_scale;
  }

  // Now all threads in the block scale and write their elements
  // NoPE data is packed in the first kv_lora_rank/2 bytes (first 256 bytes)
  const int64_t dst_idx_base = dst_idx_start + (threadIdx.x * 8);

  uint8_t result[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    result[i] =
        fp8::scaled_convert<uint8_t, scalar_t, Fp8KVCacheDataType::kFp8E4M3>(
            vals[i], tile_scale);
  }

  // Store as aligned 64-bit writes
  *reinterpret_cast<uint64_t*>(&kv_cache[dst_idx_base]) =
      *reinterpret_cast<const uint64_t*>(result);
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void indexer_k_quant_and_cache_kernel(
    const scalar_t* __restrict__ k,  // [num_tokens, head_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, cache_stride]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int head_dim,                        // dimension of each head
    const int quant_block_size,                // quantization block size
    const int cache_block_size,                // cache block size
    const int cache_stride,  // stride for each token in kv_cache

    const bool use_ue8m0  // use ue8m0 scale format
) {
  constexpr int VEC_SIZE = 4;
  const int64_t token_idx = blockIdx.x;
  const int64_t head_dim_idx = (blockIdx.y * blockDim.y * blockDim.x +
                                threadIdx.y * blockDim.x + threadIdx.x) *
                               VEC_SIZE;
  const int64_t slot_idx = slot_mapping[token_idx];
  const int64_t block_idx = slot_idx / cache_block_size;
  const int64_t block_offset = slot_idx % cache_block_size;

  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0 || (head_dim_idx >= head_dim)) {
    return;
  }

  float2 k_val = (reinterpret_cast<const float2*>(
      k))[(token_idx * head_dim + head_dim_idx) / VEC_SIZE];
  scalar_t* k_val_ptr = reinterpret_cast<scalar_t*>(&k_val);
  float amax = 0.0f;
  for (int i = 0; i < VEC_SIZE; i++) {
    amax = fmaxf(amax, fabsf(float(k_val_ptr[i])));
  }
#ifndef USE_ROCM
  __syncwarp();
#endif

  // Reduced amax
  for (int mask = 16; mask > 0; mask /= 2) {
#ifdef USE_ROCM
    amax = fmaxf(amax, __shfl_xor_sync(uint64_t(-1), amax, mask));
#else
    amax = fmaxf(amax, __shfl_xor_sync(unsigned(-1), amax, mask));
#endif
  }
#ifndef USE_ROCM
  __syncwarp();
#endif
#if defined(__gfx942__)
  float scale = fmaxf(amax, 1e-4) / 224.0f;
#else
  float scale = fmaxf(amax, 1e-4) / 448.0f;
#endif
  if (use_ue8m0) {
    scale = exp2f(ceilf(log2f(scale)));
  }

  const int64_t dst_offset = block_idx * cache_block_size * cache_stride +
                             block_offset * head_dim + head_dim_idx;
  for (int i = 0; i < VEC_SIZE; i++) {
    kv_cache[dst_offset + i] =
        fp8::scaled_convert<cache_t, scalar_t, kv_dt>(k_val_ptr[i], scale);
  }
  if (threadIdx.x == 0) {
    const int64_t dst_scale_idx =
        block_idx * cache_block_size * cache_stride +
        cache_block_size * head_dim +
        (block_offset * head_dim + head_dim_idx) * 4 / quant_block_size;
    reinterpret_cast<float*>(kv_cache)[dst_scale_idx / 4] = scale;
  }
}

template <int BLOCK_Y_SIZE>
__global__ void cp_gather_indexer_k_quant_cache_kernel(
    const char* __restrict__ kv_cache,  // [num_blocks, block_size,
                                        // cache_stride]
    char* __restrict__ dst_k,           // [num_tokens, head_dim]
    char* __restrict__ dst_scale,  // [num_tokens, head_dim / quant_block_size *
                                   // 4]
    const int* __restrict__ block_table,  // [batch_size, num_blocks]
    const int* __restrict__ cu_seq_lens,  // [batch_size + 1]
    const int batch_size,                 // batch size
    const int64_t token_stride,           // stride for each token in dst_k
    const int64_t head_dim,               // dimension of each head
    const int64_t block_stride,           // stride for each block in kv_cache
    const int64_t cache_token_stride,     // stride for each token in kv_cache
    const int64_t cache_block_size,  // num_tokens for each block in kv_cache
    const int num_blocks,            // number of blocks
    const int num_tokens,            // number of tokens
    const int quant_block_size       // quantization block size
) {
  constexpr int VEC_SIZE = sizeof(float4) / sizeof(char);
  const int token_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int head_idx = (blockIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;
  // Find batch index within a block
  __shared__ int batch_idx[BLOCK_Y_SIZE];
  for (int iter = 0; iter < cuda_utils::ceil_div(batch_size, int(blockDim.x));
       iter++) {
    int tid = iter * blockDim.x + threadIdx.x;
    if (tid < batch_size) {
      const int seq_start = cu_seq_lens[tid];
      const int seq_end = cu_seq_lens[tid + 1];
      if (token_idx >= seq_start && token_idx < seq_end) {
        batch_idx[threadIdx.y] = tid;
      }
    }
  }

#ifndef USE_ROCM
  __syncwarp();
#endif

  if (head_idx >= head_dim || token_idx >= num_tokens) {
    return;
  }
  const int inbatch_seq_idx = token_idx - cu_seq_lens[batch_idx[threadIdx.y]];
  const int block_idx = block_table[batch_idx[threadIdx.y] * num_blocks +
                                    inbatch_seq_idx / cache_block_size];
  const int64_t src_block_offset = block_idx * block_stride;
  const int64_t cache_inblock_offset =
      (inbatch_seq_idx % cache_block_size) * head_dim + head_idx;
  const int64_t src_inblock_offset = src_block_offset + cache_inblock_offset;
  const int64_t dst_inblock_offset = token_idx * token_stride + head_idx;

  reinterpret_cast<float4*>(dst_k)[dst_inblock_offset / VEC_SIZE] =
      reinterpret_cast<const float4*>(kv_cache)[src_inblock_offset / VEC_SIZE];
  ;
  if (threadIdx.x == 0) {
    const int64_t src_scale_offset =
        src_block_offset + cache_block_size * head_dim +
        cache_inblock_offset * 4 / quant_block_size;
    reinterpret_cast<float*>(dst_scale)[dst_inblock_offset / quant_block_size] =
        reinterpret_cast<const float*>(kv_cache)[src_scale_offset / 4];
  }
}

}  // namespace vllm

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)               \
  vllm::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>             \
      <<<grid, block, 0, stream>>>(                                   \
          reinterpret_cast<KV_T*>(key.data_ptr()),                    \
          reinterpret_cast<KV_T*>(value.data_ptr()),                  \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),           \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),         \
          slot_mapping.data_ptr<int64_t>(), key_stride, value_stride, \
          num_heads, head_size, block_size, x,                        \
          reinterpret_cast<const float*>(k_scale.data_ptr()),         \
          reinterpret_cast<const float*>(v_scale.data_ptr()));

void reshape_and_cache(
    torch::Tensor& key,    // [num_tokens, num_heads, head_size]
    torch::Tensor& value,  // [num_tokens, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& slot_mapping,  // [num_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);
  int head_div_x = head_size / x;

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_div_x, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE);
}

// Dispatch macro for hybrid Golay kernel by head size
#define CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, head_size)                    \
  vllm::reshape_and_cache_golay_hybrid_kernel<scalar_t, head_size>             \
      <<<grid, block, 0, stream>>>(                                            \
          reinterpret_cast<scalar_t*>(key.data_ptr()),                         \
          reinterpret_cast<scalar_t*>(value.data_ptr()),                       \
          reinterpret_cast<uint8_t*>(key_cache.data_ptr()),                    \
          reinterpret_cast<uint8_t*>(value_cache.data_ptr()),                  \
          slot_mapping.data_ptr<int64_t>(), key_stride, value_stride,          \
          num_heads, block_size,                                               \
          reinterpret_cast<const float*>(k_scale.data_ptr()),                  \
          reinterpret_cast<const float*>(v_scale.data_ptr()));

// Specialized reshape_and_cache for hybrid Golay(24,12) + Hamming(8,4) encoding
// Uses compile-time head_size for optimized indexing
void reshape_and_cache_golay_hybrid(
    torch::Tensor& key,    // [num_tokens, num_heads, head_size]
    torch::Tensor& value,  // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,    // [num_blocks, num_heads, hybrid_bytes, block_size]
    torch::Tensor& value_cache,  // [num_blocks, num_heads, hybrid_bytes, block_size]
    torch::Tensor& slot_mapping,  // [num_tokens]
    torch::Tensor& k_scale,
    torch::Tensor& v_scale) {

  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);

  // Golay hybrid stores int32 codewords through uint8* pointers with block_size stride.
  // block_size must be a multiple of 4 to ensure 4-byte alignment for int32 stores.
  // NOTE: This constraint only applies to kInt4GolayHybrid cache dtype.
  // Standard block sizes 8, 16, 32 all satisfy this requirement.
  TORCH_CHECK(block_size % 4 == 0,
      "Golay hybrid KV cache (kInt4GolayHybrid) requires block_size to be a multiple of 4 "
      "for int32_t codeword alignment. Got block_size=", block_size,
      ". Use block_size 8, 16, or 32 instead.");

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  // Calculate work items: num_triplets + remainder per head
  int num_triplets = head_size / 3;
  int remainder = head_size % 3;
  int work_items_per_head = num_triplets + remainder;

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * work_items_per_head, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Dispatch by input dtype and head_size (compile-time optimization)
  // Supported head sizes: 64, 80, 96, 112, 120, 128, 192, 256
  if (key.dtype() == at::ScalarType::Half) {
    using scalar_t = uint16_t;  // __half stored as uint16_t
    switch (head_size) {
      case 64:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 64);
        break;
      case 80:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 80);
        break;
      case 96:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 96);
        break;
      case 112:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 112);
        break;
      case 120:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 120);
        break;
      case 128:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 128);
        break;
      case 192:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 192);
        break;
      case 256:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 256);
        break;
      default:
        TORCH_CHECK(false, "Unsupported head_size for Golay hybrid: ", head_size,
                    ". Supported sizes: 64, 80, 96, 112, 120, 128, 192, 256");
    }
  } else if (key.dtype() == at::ScalarType::BFloat16) {
    using scalar_t = __nv_bfloat16;
    switch (head_size) {
      case 64:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 64);
        break;
      case 80:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 80);
        break;
      case 96:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 96);
        break;
      case 112:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 112);
        break;
      case 120:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 120);
        break;
      case 128:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 128);
        break;
      case 192:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 192);
        break;
      case 256:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 256);
        break;
      default:
        TORCH_CHECK(false, "Unsupported head_size for Golay hybrid: ", head_size,
                    ". Supported sizes: 64, 80, 96, 112, 120, 128, 192, 256");
    }
  } else if (key.dtype() == at::ScalarType::Float) {
    using scalar_t = float;
    switch (head_size) {
      case 64:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 64);
        break;
      case 80:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 80);
        break;
      case 96:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 96);
        break;
      case 112:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 112);
        break;
      case 120:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 120);
        break;
      case 128:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 128);
        break;
      case 192:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 192);
        break;
      case 256:
        CALL_GOLAY_HYBRID_BY_HEAD_SIZE(scalar_t, 256);
        break;
      default:
        TORCH_CHECK(false, "Unsupported head_size for Golay hybrid: ", head_size,
                    ". Supported sizes: 64, 80, 96, 112, 120, 128, 192, 256");
    }
  } else {
    TORCH_CHECK(false, "Unsupported dtype for Golay hybrid: ", key.dtype());
  }
}

// Dispatch macro for RS kernel by head size
#define CALL_RS_BY_HEAD_SIZE(scalar_t, head_size)                              \
  vllm::reshape_and_cache_rs_kernel<scalar_t, head_size>                       \
      <<<grid, block, 0, stream>>>(                                            \
          reinterpret_cast<scalar_t*>(key.data_ptr()),                         \
          reinterpret_cast<scalar_t*>(value.data_ptr()),                       \
          reinterpret_cast<uint8_t*>(key_cache.data_ptr()),                    \
          reinterpret_cast<uint8_t*>(value_cache.data_ptr()),                  \
          slot_mapping.data_ptr<int64_t>(), key_stride, value_stride,          \
          num_heads, block_size,                                               \
          reinterpret_cast<const float*>(k_scale.data_ptr()),                  \
          reinterpret_cast<const float*>(v_scale.data_ptr()));

// Specialized reshape_and_cache for Reed-Solomon RS(12,8) encoding
// Uses compile-time head_size for optimized indexing
void reshape_and_cache_rs(
    torch::Tensor& key,    // [num_tokens, num_heads, head_size]
    torch::Tensor& value,  // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,    // [num_blocks, num_heads, rs_bytes, block_size]
    torch::Tensor& value_cache,  // [num_blocks, num_heads, rs_bytes, block_size]
    torch::Tensor& slot_mapping,  // [num_tokens]
    torch::Tensor& k_scale,
    torch::Tensor& v_scale) {

  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  // Calculate work items: num_octuplets + (1 if remainder > 0) per head
  int num_octuplets = head_size / 8;
  int remainder = head_size % 8;
  int work_items_per_head = num_octuplets + ((remainder > 0) ? 1 : 0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * work_items_per_head, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Dispatch by input dtype and head_size (compile-time optimization)
  // Supported head sizes: 64, 80, 96, 112, 120, 128, 192, 256
  if (key.dtype() == at::ScalarType::Half) {
    using scalar_t = uint16_t;  // __half stored as uint16_t
    switch (head_size) {
      case 64:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 64);
        break;
      case 80:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 80);
        break;
      case 96:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 96);
        break;
      case 112:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 112);
        break;
      case 120:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 120);
        break;
      case 128:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 128);
        break;
      case 192:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 192);
        break;
      case 256:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 256);
        break;
      default:
        TORCH_CHECK(false, "Unsupported head_size for RS: ", head_size,
                    ". Supported sizes: 64, 80, 96, 112, 120, 128, 192, 256");
    }
  } else if (key.dtype() == at::ScalarType::BFloat16) {
    using scalar_t = __nv_bfloat16;
    switch (head_size) {
      case 64:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 64);
        break;
      case 80:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 80);
        break;
      case 96:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 96);
        break;
      case 112:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 112);
        break;
      case 120:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 120);
        break;
      case 128:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 128);
        break;
      case 192:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 192);
        break;
      case 256:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 256);
        break;
      default:
        TORCH_CHECK(false, "Unsupported head_size for RS: ", head_size,
                    ". Supported sizes: 64, 80, 96, 112, 120, 128, 192, 256");
    }
  } else if (key.dtype() == at::ScalarType::Float) {
    using scalar_t = float;
    switch (head_size) {
      case 64:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 64);
        break;
      case 80:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 80);
        break;
      case 96:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 96);
        break;
      case 112:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 112);
        break;
      case 120:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 120);
        break;
      case 128:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 128);
        break;
      case 192:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 192);
        break;
      case 256:
        CALL_RS_BY_HEAD_SIZE(scalar_t, 256);
        break;
      default:
        TORCH_CHECK(false, "Unsupported head_size for RS: ", head_size,
                    ". Supported sizes: 64, 80, 96, 112, 120, 128, 192, 256");
    }
  } else {
    TORCH_CHECK(false, "Unsupported dtype for RS: ", key.dtype());
  }
}

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE_FLASH(KV_T, CACHE_T, KV_DTYPE)             \
  vllm::reshape_and_cache_flash_kernel<KV_T, CACHE_T, KV_DTYPE>           \
      <<<grid, block, 0, stream>>>(                                       \
          reinterpret_cast<KV_T*>(key.data_ptr()),                        \
          reinterpret_cast<KV_T*>(value.data_ptr()),                      \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),               \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),             \
          slot_mapping.data_ptr<int64_t>(), block_stride, page_stride,    \
          head_stride, key_stride, value_stride, num_heads, head_size,    \
          block_size, reinterpret_cast<const float*>(k_scale.data_ptr()), \
          reinterpret_cast<const float*>(v_scale.data_ptr()));

void reshape_and_cache_flash(
    torch::Tensor& key,        // [num_tokens, num_heads, head_size]
    torch::Tensor& value,      // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(1);

  int64_t key_stride = key.stride(0);
  int64_t value_stride = value.stride(0);
  int64_t block_stride = key_cache.stride(0);
  int64_t page_stride = key_cache.stride(1);
  int64_t head_stride = key_cache.stride(2);
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE_FLASH);
}

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_CONCAT_AND_CACHE_MLA(KV_T, CACHE_T, KV_DTYPE)              \
  vllm::concat_and_cache_mla_kernel<KV_T, CACHE_T, KV_DTYPE>            \
      <<<grid, block, 0, stream>>>(                                     \
          reinterpret_cast<KV_T*>(kv_c.data_ptr()),                     \
          reinterpret_cast<KV_T*>(k_pe.data_ptr()),                     \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), block_stride, entry_stride, \
          kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,   \
          reinterpret_cast<const float*>(scale.data_ptr()));

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
#define CALL_CONCAT_AND_CACHE_DS_MLA(KV_T, CACHE_T, KV_DTYPE)           \
  vllm::concat_and_cache_ds_mla_kernel<KV_T, CACHE_T, KV_DTYPE>         \
      <<<grid, block, 0, stream>>>(                                     \
          reinterpret_cast<KV_T*>(kv_c.data_ptr()),                     \
          reinterpret_cast<KV_T*>(k_pe.data_ptr()),                     \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), block_stride, entry_stride, \
          kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,   \
          reinterpret_cast<const float*>(scale.data_ptr()));

void concat_and_cache_mla(
    torch::Tensor& kv_c,          // [num_tokens, kv_lora_rank]
    torch::Tensor& k_pe,          // [num_tokens, pe_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, (kv_lora_rank +
                                  // pe_dim)]
    torch::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& scale) {
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int kv_lora_rank = kv_c.size(1);
  int pe_dim = k_pe.size(1);
  int block_size = kv_cache.size(1);

  if (kv_cache_dtype == "fp8_ds_mla") {
    TORCH_CHECK(kv_lora_rank == 512, "kv_lora_rank must be 512 for fp8_ds_mla");
    TORCH_CHECK(pe_dim == 64, "pe_dim must be 64 for fp8_ds_mla");
    TORCH_CHECK(kv_cache.size(2) == 656 / kv_cache.itemsize(),
                "kv_cache.size(2) must be 656 bytes for fp8_ds_mla");
    TORCH_CHECK(kv_c.itemsize() == 2,
                "kv_c.itemsize() must be 2 for fp8_ds_mla");
    TORCH_CHECK(k_pe.itemsize() == 2,
                "k_pe.itemsize() must be 2 for fp8_ds_mla");
  } else {
    TORCH_CHECK(kv_cache.size(2) == kv_lora_rank + pe_dim);
  }

  int kv_c_stride = kv_c.stride(0);
  int k_pe_stride = k_pe.stride(0);
  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_c));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (kv_cache_dtype == "fp8_ds_mla") {
    dim3 grid(num_tokens);
    // For the NoPE part, each tile of 128 elements is handled by half of one
    // warp (16 threads). There are 4 total tiles, so 2 warps (64 threads).
    // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
    // The RoPE part (last 64 elements) is handled by another 1 warp (32
    // threads). So in total, we use 3 warps (96 threads) per block.
    dim3 block(96);
    DISPATCH_BY_KV_CACHE_DTYPE(kv_c.dtype(), kv_cache_dtype,
                               CALL_CONCAT_AND_CACHE_DS_MLA);
  } else {
    dim3 grid(num_tokens);
    dim3 block(std::min(kv_lora_rank, 512));
    DISPATCH_BY_KV_CACHE_DTYPE(kv_c.dtype(), kv_cache_dtype,
                               CALL_CONCAT_AND_CACHE_MLA);
  }
}

namespace vllm {

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__global__ void convert_fp8_kernel(const Tin* __restrict__ src_cache,
                                   Tout* __restrict__ dst_cache,
                                   const float scale,
                                   const int64_t block_stride) {
  const int64_t block_idx = blockIdx.x;
  for (int i = threadIdx.x; i < block_stride; i += blockDim.x) {
    int64_t idx = block_idx * block_stride + i;
    dst_cache[idx] =
        fp8::scaled_convert<Tout, Tin, kv_dt>(src_cache[idx], scale);
  }
}

}  // namespace vllm

#define CALL_CONVERT_FP8(Tout, Tin, KV_DTYPE)                                \
  vllm::convert_fp8_kernel<Tout, Tin, KV_DTYPE><<<grid, block, 0, stream>>>( \
      reinterpret_cast<Tin*>(src_cache.data_ptr()),                          \
      reinterpret_cast<Tout*>(dst_cache.data_ptr()), scale, block_stride);

// Only for testing.
void convert_fp8(torch::Tensor& dst_cache, torch::Tensor& src_cache,
                 const double scale, const std::string& kv_cache_dtype) {
  torch::Device src_device = src_cache.device();
  torch::Device dst_device = dst_cache.device();
  TORCH_CHECK(src_device.is_cuda(), "src must be on a GPU")
  TORCH_CHECK(dst_device.is_cuda(), "dst must be on a GPU")
  TORCH_CHECK(src_device.index() == dst_device.index(),
              "src and dst must be on the same GPU");
  at::cuda::OptionalCUDAGuard device_guard(src_device);

  int64_t num_blocks = src_cache.size(0);
  int64_t block_stride = src_cache.stride(0);

  dim3 grid(num_blocks);
  dim3 block(std::min(block_stride, int64_t(512)));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (kv_cache_dtype == "auto") {
    if (src_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(uint8_t, float, vllm::Fp8KVCacheDataType::kAuto);
    } else if (src_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint8_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (src_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(uint8_t, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(float, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    }
  } else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    if (src_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(uint8_t, float, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (src_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint8_t, uint16_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (src_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(uint8_t, __nv_bfloat16,
                       vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(__nv_bfloat16, uint8_t,
                       vllm::Fp8KVCacheDataType::kFp8E4M3);
    }
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", kv_cache_dtype);
  }
}

namespace vllm {

// grid is launched with dimensions (batch, num_splits)
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt,
          int ENTRY_SIZE, int CTA_SIZE>
__global__ void gather_and_maybe_dequant_cache(
    const cache_t* __restrict__ src_cache,     // [NUM_BLOCKS, BLOCK_SIZE,
                                               // ENTRIES...]
    scalar_t* __restrict__ dst,                // [TOT_TOKENS, ENTRIES...]
    const int32_t* __restrict__ block_table,   // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ cu_seq_lens,   // [BATCH+1]
    const int32_t* __restrict__ token_to_seq,  // [MAX_TOKEN_ACROSS_CHUNK]
    const int32_t num_tokens, const int32_t block_size,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride,
    const float* __restrict__ scale,
    const int32_t* __restrict__ seq_starts) {  // Optional: starting offsets per
                                               // batch
  constexpr int vec_size = sizeof(float4) / sizeof(scalar_t);
  using ltype = vllm::vec_n_t<cache_t, vec_size>;
  using stype = vllm::vec_n_t<scalar_t, vec_size>;
  // We are adding this for code readability which will be optimized out when
  // build in release.
  assert(CTA_SIZE == blockDim.x);

#pragma unroll
  for (int token_id = blockIdx.x; token_id < num_tokens;
       token_id += gridDim.x) {
    int64_t batch_id = token_to_seq[token_id];
    int64_t batch_start = cu_seq_lens[batch_id];
    int64_t batch_end = cu_seq_lens[batch_id + 1];
    int32_t batch_offset = token_id - batch_start;

    if (token_id >= batch_end) return;
    int32_t offset = 0;
    if (seq_starts != nullptr) {
      offset = seq_starts[batch_id];
    }
    batch_offset += offset;
    int32_t block_table_id = batch_offset / block_size;
    int32_t slot_id = batch_offset % block_size;
    int32_t block_table_offset = batch_id * block_table_stride + block_table_id;
    int32_t block_id = block_table[block_table_offset];
    int64_t cache_offset =
        block_id * cache_block_stride + slot_id * cache_entry_stride;
    constexpr int32_t vec_iter_cnt = ENTRY_SIZE / vec_size;
    scalar_t* dst_ = dst + token_id * dst_entry_stride;
    cache_t* src_ = const_cast<cache_t*>(src_cache) + cache_offset;

#pragma unroll
    for (int idx = threadIdx.x; idx < vec_iter_cnt; idx += CTA_SIZE) {
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        reinterpret_cast<stype*>(dst_)[idx] =
            static_cast<stype>(reinterpret_cast<ltype*>(src_)[idx]);
      } else {
        ltype loaded_val = reinterpret_cast<ltype*>(src_)[idx];
        stype store_val;
#pragma unroll
        for (int j = 0; j < vec_size; ++j) {
          store_val.val[j] = fp8::scaled_convert<scalar_t, cache_t, kv_dt>(
              loaded_val.val[j], *scale);
        }
        reinterpret_cast<stype*>(dst_)[idx] = store_val;
      }
    }
    // process tail
    constexpr int32_t tail_cnt = ENTRY_SIZE % vec_size;
    dst_ = dst_ + ENTRY_SIZE - tail_cnt;
    src_ = src_ + ENTRY_SIZE - tail_cnt;
#pragma unroll
    for (int idx = threadIdx.x; idx < tail_cnt; idx += CTA_SIZE) {
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        dst_[idx] = static_cast<scalar_t>(src_[idx]);
      } else {
        dst_[idx] =
            fp8::scaled_convert<scalar_t, cache_t, kv_dt>(src_[idx], *scale);
      }
    }
  }
}

}  // namespace vllm

// Macro to dispatch the kernel based on the data type.
// SCALAR_T is the data type of the destination tensor.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_GATHER_CACHE(SCALAR_T, CACHE_T, KV_DTYPE)                        \
  vllm::gather_and_maybe_dequant_cache<SCALAR_T, CACHE_T, KV_DTYPE, 576,      \
                                       thread_block_size>                     \
      <<<grid, block, 0, stream>>>(                                           \
          reinterpret_cast<CACHE_T*>(src_cache.data_ptr()),                   \
          reinterpret_cast<SCALAR_T*>(dst.data_ptr()),                        \
          block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(),   \
          token_to_seq.data_ptr<int32_t>(), num_tokens, block_size,           \
          block_table_stride, cache_block_stride, cache_entry_stride,         \
          dst_entry_stride, reinterpret_cast<const float*>(scale.data_ptr()), \
          seq_starts_ptr);

// Gather sequences from the cache into the destination tensor.
//  - cu_seq_lens contains the cumulative sequence lengths for each batch
//  - block_table contains the cache block indices for each sequence
//  - token_to_seq contains the back mapping from token_id to batch_id
//  - Optionally, seq_starts (if provided) offsets the starting block index by
//  (seq_starts[bid] / page_size)
void gather_and_maybe_dequant_cache(
    torch::Tensor const& src_cache,     // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,           // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,   // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,   // [BATCH+1]
    torch::Tensor const& token_to_seq,  // [MAX_TOKEN_ACROSS_CHUNKS]
    int64_t num_tokens, const std::string& kv_cache_dtype,
    torch::Tensor const& scale,
    std::optional<torch::Tensor> seq_starts = std::nullopt) {
  at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int32_t block_size = src_cache.size(1);
  int32_t head_dim = dst.size(-1);

  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(cu_seq_lens.dtype() == torch::kInt32,
              "cu_seq_lens must be int32");
  if (seq_starts.has_value()) {
    TORCH_CHECK(seq_starts.value().dtype() == torch::kInt32,
                "seq_starts must be int32");
  }
  TORCH_CHECK(head_dim == 576,
              "gather_and_maybe_dequant_cache only support the head_dim to 576 "
              "for better performance")

  TORCH_CHECK(src_cache.device() == dst.device(),
              "src_cache and dst must be on the same device");
  TORCH_CHECK(src_cache.device() == block_table.device(),
              "src_cache and block_table must be on the same device");
  TORCH_CHECK(src_cache.device() == cu_seq_lens.device(),
              "src_cache and cu_seq_lens must be on the same device");
  if (seq_starts.has_value()) {
    TORCH_CHECK(src_cache.device() == seq_starts.value().device(),
                "src_cache and seq_starts must be on the same device");
  }

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  constexpr int32_t thread_block_size = 64;
  dim3 grid(num_tokens);
  dim3 block(thread_block_size);

  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().data_ptr<int32_t>() : nullptr;

  DISPATCH_BY_KV_CACHE_DTYPE(dst.dtype(), kv_cache_dtype, CALL_GATHER_CACHE);
}

namespace vllm {

// Gather and upconvert FP8 KV cache tokens to BF16 workspace
// Similar to cp_gather_cache but specifically for FP8->BF16 conversion
__global__ void cp_gather_and_upconvert_fp8_kv_cache(
    const uint8_t* __restrict__ src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, 656]
    __nv_bfloat16* __restrict__ dst,          // [TOT_TOKENS, 576]
    const int32_t* __restrict__ block_table,  // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ seq_lens,     // [BATCH]
    const int32_t* __restrict__ workspace_starts,  // [BATCH]
    const int32_t block_size, const int32_t head_dim,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride) {
  const int64_t bid = blockIdx.x;  // Batch ID
  const int32_t num_splits = gridDim.y;
  const int32_t split = blockIdx.y;
  const int32_t seq_start = workspace_starts[bid];
  const int32_t seq_len = seq_lens[bid];
  const int32_t tot_slots = seq_len;
  const int32_t split_slots = cuda_utils::ceil_div(tot_slots, num_splits);

  const int32_t split_start = split * split_slots;
  const int32_t split_end = min((split + 1) * split_slots, tot_slots);

  const bool is_active_split = (split_start < tot_slots);

  if (!is_active_split) return;

  // Adjust the pointer for the block_table for this batch
  const int32_t batch_offset = bid * block_table_stride;
  int32_t offset = split_start;
  int32_t offset_div = offset / block_size;
  offset = offset % block_size;
  const int32_t* batch_block_table = block_table + batch_offset;

  // Adjust dst pointer based on the cumulative sequence lengths
  dst += seq_start * dst_entry_stride;

  const int tid = threadIdx.x;

  // Process each token in this split
  for (int pid = split_start; pid < split_end; ++pid) {
    auto block_id = batch_block_table[offset_div];
    const uint8_t* token_ptr =
        src_cache + block_id * cache_block_stride + offset * cache_entry_stride;
    __nv_bfloat16* dst_ptr = dst + pid * dst_entry_stride;

    // FP8 format: 512 bytes fp8 + 16 bytes scales + 128 bytes rope (64 bf16)
    const uint8_t* no_pe_ptr = token_ptr;
    const float* scales_ptr = reinterpret_cast<const float*>(token_ptr + 512);
    const __nv_bfloat16* rope_ptr =
        reinterpret_cast<const __nv_bfloat16*>(token_ptr + 512 + 16);

    // Parallelize fp8 dequant (512 elements) and rope copy (64 elements)
    if (tid < 512) {
      // FP8 dequantization
      const int tile = tid >> 7;  // each tile is 128 elements
      const float scale = scales_ptr[tile];
      const uint8_t val = no_pe_ptr[tid];
      dst_ptr[tid] =
          fp8::scaled_convert<__nv_bfloat16, uint8_t,
                              vllm::Fp8KVCacheDataType::kFp8E4M3>(val, scale);
    } else if (tid < 576) {
      // Rope copy (64 bf16 elements)
      const int rope_idx = tid - 512;
      dst_ptr[512 + rope_idx] = rope_ptr[rope_idx];
    }

    // Move to next token
    offset += 1;
    if (offset == block_size) {
      offset_div += 1;
      offset = 0;
    }
  }
}

template <typename scalar_t>
// Note(hc): The cp_gather_cache allows seq_starts to no longer be divisible by
// block_size.
__global__ void cp_gather_cache(
    const scalar_t* __restrict__ src_cache,   // [NUM_BLOCKS, BLOCK_SIZE,
                                              // ENTRY_SIZE]
    scalar_t* __restrict__ dst,               // [TOT_TOKENS, ENTRY_SIZE]
    const int32_t* __restrict__ block_table,  // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ cu_seq_lens,  // [BATCH+1]
    const int32_t block_size, const int32_t entry_size,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride,
    const int32_t* __restrict__ seq_starts  // Optional: starting offsets per
                                            // batch
) {
  const int64_t bid = blockIdx.x;  // Batch ID
  const int32_t num_splits = gridDim.y;
  const int32_t split = blockIdx.y;
  const int32_t seq_start = cu_seq_lens[bid];
  const int32_t seq_end = cu_seq_lens[bid + 1];
  const int32_t seq_len = seq_end - seq_start;
  const int32_t tot_slots = seq_len;
  const int32_t split_slots = cuda_utils::ceil_div(tot_slots, num_splits);

  const int32_t split_start = split * split_slots;
  const int32_t split_end = min((split + 1) * split_slots, tot_slots);

  const bool is_active_split = (split_start < tot_slots);

  if (!is_active_split) return;

  // Adjust the pointer for the block_table for this batch.
  // If seq_starts is provided, compute an offset based on it
  const int32_t batch_offset = bid * block_table_stride;
  int32_t offset = split_start;
  if (seq_starts != nullptr) {
    offset += seq_starts[bid];
  }
  int32_t offset_div = offset / block_size;
  offset = offset % block_size;
  const int32_t* batch_block_table = block_table + batch_offset;

  // Adjust dst pointer based on the cumulative sequence lengths.
  dst += seq_start * dst_entry_stride;

  auto copy_entry = [&](const scalar_t* __restrict__ _src,
                        scalar_t* __restrict__ _dst) {
    for (int i = threadIdx.x; i < entry_size; i += blockDim.x)
      _dst[i] = _src[i];
  };

  for (int pid = split_start; pid < split_end; ++pid) {
    auto block_id = batch_block_table[offset_div];
    auto block_start_ptr = src_cache + block_id * cache_block_stride;
    auto block_dst_ptr = dst + pid * dst_entry_stride;
    copy_entry(block_start_ptr + offset * cache_entry_stride, block_dst_ptr);
    offset += 1;
    // bump to next block
    if (offset == block_size) {
      offset_div += 1;
      offset = 0;
    }
  }
}
}  // namespace vllm

// Macro to dispatch the kernel based on the data type.
#define CALL_CP_GATHER_CACHE(CPY_DTYPE)                                 \
  vllm::cp_gather_cache<CPY_DTYPE><<<grid, block, 0, stream>>>(         \
      reinterpret_cast<CPY_DTYPE*>(src_cache.data_ptr()),               \
      reinterpret_cast<CPY_DTYPE*>(dst.data_ptr()),                     \
      block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(), \
      block_size, entry_size, block_table_stride, cache_block_stride,   \
      cache_entry_stride, dst_entry_stride, seq_starts_ptr);

// Gather sequences from the cache into the destination tensor.
//  - cu_seq_lens contains the cumulative sequence lengths for each batch
//  - block_table contains the cache block indices for each sequence
//  - Optionally, seq_starts (if provided) offsets the starting slot index by
//  seq_starts[bid]
void cp_gather_cache(
    torch::Tensor const& src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,          // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,  // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,  // [BATCH+1]
    int64_t batch_size,
    std::optional<torch::Tensor> seq_starts = std::nullopt) {
  at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int32_t block_size = src_cache.size(1);
  int32_t entry_size = src_cache.flatten(2, -1).size(2);

  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(cu_seq_lens.dtype() == torch::kInt32,
              "cu_seq_lens must be int32");
  if (seq_starts.has_value()) {
    TORCH_CHECK(seq_starts.value().dtype() == torch::kInt32,
                "seq_starts must be int32");
  }

  TORCH_CHECK(src_cache.device() == dst.device(),
              "src_cache and dst must be on the same device");
  TORCH_CHECK(src_cache.device() == block_table.device(),
              "src_cache and block_table must be on the same device");
  TORCH_CHECK(src_cache.device() == cu_seq_lens.device(),
              "src_cache and cu_seq_lens must be on the same device");
  if (seq_starts.has_value()) {
    TORCH_CHECK(src_cache.device() == seq_starts.value().device(),
                "src_cache and seq_starts must be on the same device");
  }

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  // Decide on the number of splits based on the batch size.
  int num_splits = batch_size > 128 ? 2 : batch_size > 64 ? 4 : 16;
  dim3 grid(batch_size, num_splits);
  dim3 block(1024);

  TORCH_CHECK(src_cache.dtype() == dst.dtype(),
              "src_cache and dst must have the same dtype");

  const int dtype_bits = src_cache.element_size() * 8;
  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().data_ptr<int32_t>() : nullptr;

  if (dtype_bits == 32) {
    CALL_CP_GATHER_CACHE(uint32_t);
  } else if (dtype_bits == 16) {
    CALL_CP_GATHER_CACHE(uint16_t);
  } else if (dtype_bits == 8) {
    CALL_CP_GATHER_CACHE(uint8_t);
  } else {
    TORCH_CHECK(false, "Unsupported data type width: ", dtype_bits);
  }
}

void cp_gather_and_upconvert_fp8_kv_cache(
    torch::Tensor const& src_cache,         // [NUM_BLOCKS, BLOCK_SIZE, 656]
    torch::Tensor const& dst,               // [TOT_TOKENS, 576]
    torch::Tensor const& block_table,       // [BATCH, BLOCK_INDICES]
    torch::Tensor const& seq_lens,          // [BATCH]
    torch::Tensor const& workspace_starts,  // [BATCH]
    int64_t batch_size) {
  at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int32_t block_size = src_cache.size(1);
  int32_t head_dim = dst.size(1);

  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(workspace_starts.dtype() == torch::kInt32,
              "workspace_starts must be int32");

  TORCH_CHECK(src_cache.device() == dst.device(),
              "src_cache and dst must be on the same device");
  TORCH_CHECK(src_cache.device() == block_table.device(),
              "src_cache and block_table must be on the same device");
  TORCH_CHECK(src_cache.device() == seq_lens.device(),
              "src_cache and seq_lens must be on the same device");
  TORCH_CHECK(src_cache.device() == workspace_starts.device(),
              "src_cache and workspace_starts must be on the same device");

  TORCH_CHECK(src_cache.dtype() == torch::kUInt8, "src_cache must be uint8");
  TORCH_CHECK(dst.dtype() == torch::kBFloat16, "dst must be bfloat16");
  TORCH_CHECK(head_dim == 576, "head_dim must be 576 for MLA");

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  // Decide on the number of splits based on the batch size
  int num_splits = batch_size > 128 ? 2 : batch_size > 64 ? 4 : 16;
  dim3 grid(batch_size, num_splits);
  dim3 block(576);

  vllm::cp_gather_and_upconvert_fp8_kv_cache<<<grid, block, 0, stream>>>(
      src_cache.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()),
      block_table.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
      workspace_starts.data_ptr<int32_t>(), block_size, head_dim,
      block_table_stride, cache_block_stride, cache_entry_stride,
      dst_entry_stride);
}

// ============================================================================
// ECC Gather-Decode Kernel for INT4-ECC KV Cache
// ============================================================================
// Gathers ECC-encoded KV cache from paged blocks to contiguous workspace,
// decoding and dequantizing in the process. This enables FlashAttention
// (which doesn't have native ECC support) to work with ECC-protected caches.

namespace vllm {

// Template kernel for ECC gather-decode
// Handles Hamming(8,4) SECDED - one byte per INT4 value
template <typename scalar_t>
__global__ void cp_gather_and_ecc_decode_hamming84_kernel(
    const uint8_t* __restrict__ src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE]
    scalar_t* __restrict__ dst,               // [TOT_TOKENS, NUM_HEADS, HEAD_SIZE]
    const int32_t* __restrict__ block_table,  // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ cu_seq_lens,  // [BATCH+1]
    const int32_t* __restrict__ seq_starts,   // [BATCH] - optional offset per batch
    const float* __restrict__ scale,          // [1] or [NUM_HEADS]
    int64_t* __restrict__ hamming_stats,      // [HAMMING_STATS_SIZE] for error tracking
    const int32_t block_size,
    const int32_t num_heads,
    const int32_t head_size,
    const int64_t block_table_stride,
    const int64_t cache_block_stride,
    const int64_t cache_entry_stride,
    const int64_t dst_entry_stride) {
  const int64_t bid = blockIdx.x;  // Batch ID
  const int32_t num_splits = gridDim.y;
  const int32_t split = blockIdx.y;
  const int32_t seq_start = cu_seq_lens[bid];
  const int32_t seq_end = cu_seq_lens[bid + 1];
  const int32_t seq_len = seq_end - seq_start;
  const int32_t tot_slots = seq_len;
  const int32_t split_slots = cuda_utils::ceil_div(tot_slots, num_splits);

  const int32_t split_start = split * split_slots;
  const int32_t split_end = min((split + 1) * split_slots, tot_slots);

  if (split_start >= tot_slots) return;

  // Adjust block_table pointer for this batch
  const int32_t batch_offset = bid * block_table_stride;
  int32_t offset = split_start;
  if (seq_starts != nullptr) {
    offset += seq_starts[bid];
  }
  int32_t offset_div = offset / block_size;
  offset = offset % block_size;
  const int32_t* batch_block_table = block_table + batch_offset;

  // Adjust dst pointer based on cumulative sequence lengths
  dst += seq_start * dst_entry_stride;

  const float scale_val = scale[0];
  const int entry_size = num_heads * head_size;
  const int tid = threadIdx.x;

  // Process each token in this split
  for (int pid = split_start; pid < split_end; ++pid) {
    auto block_id = batch_block_table[offset_div];
    const uint8_t* token_ptr = src_cache + block_id * cache_block_stride +
                               offset * cache_entry_stride;
    scalar_t* dst_ptr = dst + pid * dst_entry_stride;

    // Decode each element in parallel
    for (int i = tid; i < entry_size; i += blockDim.x) {
      uint8_t codeword = token_ptr[i];
      ecc::ErrorType error_type;
      scalar_t decoded = ecc::int4_ecc_decode<scalar_t>(codeword, scale_val, &error_type);
      dst_ptr[i] = decoded;

      // Track error statistics
      if (hamming_stats != nullptr) {
        if (error_type == ecc::ErrorType::NO_ERROR) {
          atomicAdd(reinterpret_cast<unsigned long long*>(&hamming_stats[ecc::HAMMING_STATS_NO_ERROR]), 1ULL);
        } else if (error_type == ecc::ErrorType::SINGLE_CORRECTED) {
          atomicAdd(reinterpret_cast<unsigned long long*>(&hamming_stats[ecc::HAMMING_STATS_CORRECTED]), 1ULL);
        } else if (error_type == ecc::ErrorType::DOUBLE_DETECTED) {
          atomicAdd(reinterpret_cast<unsigned long long*>(&hamming_stats[ecc::HAMMING_STATS_DETECTED]), 1ULL);
        }
      }
    }

    // Move to next token
    offset += 1;
    if (offset == block_size) {
      offset_div += 1;
      offset = 0;
    }
  }
}

// Template kernel for Hamming(7,4) SEC gather-decode
template <typename scalar_t>
__global__ void cp_gather_and_ecc_decode_hamming74_kernel(
    const uint8_t* __restrict__ src_cache,
    scalar_t* __restrict__ dst,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seq_lens,
    const int32_t* __restrict__ seq_starts,
    const float* __restrict__ scale,
    int64_t* __restrict__ hamming_stats,
    const int32_t block_size,
    const int32_t num_heads,
    const int32_t head_size,
    const int64_t block_table_stride,
    const int64_t cache_block_stride,
    const int64_t cache_entry_stride,
    const int64_t dst_entry_stride) {
  const int64_t bid = blockIdx.x;
  const int32_t num_splits = gridDim.y;
  const int32_t split = blockIdx.y;
  const int32_t seq_start = cu_seq_lens[bid];
  const int32_t seq_end = cu_seq_lens[bid + 1];
  const int32_t seq_len = seq_end - seq_start;
  const int32_t tot_slots = seq_len;
  const int32_t split_slots = cuda_utils::ceil_div(tot_slots, num_splits);

  const int32_t split_start = split * split_slots;
  const int32_t split_end = min((split + 1) * split_slots, tot_slots);

  if (split_start >= tot_slots) return;

  const int32_t batch_offset = bid * block_table_stride;
  int32_t offset = split_start;
  if (seq_starts != nullptr) {
    offset += seq_starts[bid];
  }
  int32_t offset_div = offset / block_size;
  offset = offset % block_size;
  const int32_t* batch_block_table = block_table + batch_offset;

  dst += seq_start * dst_entry_stride;

  const float scale_val = scale[0];
  const int entry_size = num_heads * head_size;
  const int tid = threadIdx.x;

  for (int pid = split_start; pid < split_end; ++pid) {
    auto block_id = batch_block_table[offset_div];
    const uint8_t* token_ptr = src_cache + block_id * cache_block_stride +
                               offset * cache_entry_stride;
    scalar_t* dst_ptr = dst + pid * dst_entry_stride;

    for (int i = tid; i < entry_size; i += blockDim.x) {
      uint8_t codeword = token_ptr[i];
      ecc::ErrorType74 error_type;
      scalar_t decoded = ecc::int4_h74_decode<scalar_t>(codeword, scale_val, &error_type);
      dst_ptr[i] = decoded;

      if (hamming_stats != nullptr) {
        if (error_type == ecc::ErrorType74::NO_ERROR) {
          atomicAdd(reinterpret_cast<unsigned long long*>(&hamming_stats[ecc::HAMMING_STATS_NO_ERROR]), 1ULL);
        } else if (error_type == ecc::ErrorType74::SINGLE_CORRECTED) {
          atomicAdd(reinterpret_cast<unsigned long long*>(&hamming_stats[ecc::HAMMING_STATS_CORRECTED]), 1ULL);
        }
      }
    }

    offset += 1;
    if (offset == block_size) {
      offset_div += 1;
      offset = 0;
    }
  }
}

// Template kernel for Golay(24,12) + Hamming hybrid gather-decode
// Matches strided memory layout from reshape_and_cache_golay_hybrid_kernel:
//   Layout: [num_blocks, num_heads, hybrid_head_bytes, block_size]
// where hybrid_head_bytes = num_triplets * 4 + remainder_count
//
// For each token at (block_id, block_offset):
//   - Golay triplet i at byte offset i*4 is stored at:
//       base + head_idx * hybrid_head_bytes * block_size + i * 4 * block_size + block_offset
//   - Hamming remainder j is stored at:
//       base + head_idx * hybrid_head_bytes * block_size + golay_bytes * block_size + j * block_size + block_offset
template <typename scalar_t>
__global__ void cp_gather_and_ecc_decode_golay_kernel(
    const uint8_t* __restrict__ src_cache,    // [NUM_BLOCKS, NUM_HEADS, HYBRID_BYTES, BLOCK_SIZE]
    scalar_t* __restrict__ dst,               // [TOT_TOKENS, NUM_HEADS, HEAD_SIZE]
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seq_lens,
    const int32_t* __restrict__ seq_starts,
    const float* __restrict__ scale,
    const int32_t* __restrict__ syndrome_lut, // [4096] Golay syndrome lookup table
    int64_t* __restrict__ golay_stats,        // [GOLAY_STATS_SIZE] for error tracking
    int64_t* __restrict__ hamming_stats,      // [HAMMING_STATS_SIZE] for remainder stats
    const int32_t block_size,
    const int32_t num_heads,
    const int32_t head_size,
    const int64_t block_table_stride,
    const int64_t dst_entry_stride) {
  const int64_t bid = blockIdx.x;
  const int32_t num_splits = gridDim.y;
  const int32_t split = blockIdx.y;
  const int32_t seq_start = cu_seq_lens[bid];
  const int32_t seq_end = cu_seq_lens[bid + 1];
  const int32_t seq_len = seq_end - seq_start;
  const int32_t tot_slots = seq_len;
  const int32_t split_slots = cuda_utils::ceil_div(tot_slots, num_splits);

  const int32_t split_start = split * split_slots;
  const int32_t split_end = min((split + 1) * split_slots, tot_slots);

  if (split_start >= tot_slots) return;

  const int32_t batch_offset = bid * block_table_stride;
  int32_t kv_pos = split_start;
  if (seq_starts != nullptr) {
    kv_pos += seq_starts[bid];
  }
  int32_t block_table_idx = kv_pos / block_size;
  int32_t block_offset = kv_pos % block_size;
  const int32_t* batch_block_table = block_table + batch_offset;

  dst += seq_start * dst_entry_stride;

  const float scale_val = scale[0];
  const int num_triplets = head_size / 3;
  const int remainder = head_size % 3;
  const int golay_bytes = num_triplets * 4;           // Bytes for Golay codewords per head
  const int hybrid_head_bytes = golay_bytes + remainder;  // Total bytes per head
  const int64_t head_stride_bytes = static_cast<int64_t>(hybrid_head_bytes) * block_size;
  const int64_t cache_block_stride = static_cast<int64_t>(num_heads) * head_stride_bytes;
  const int tid = threadIdx.x;

  for (int pid = split_start; pid < split_end; ++pid) {
    const int32_t block_id = batch_block_table[block_table_idx];

    // Base pointer for this block in cache
    const uint8_t* block_base = src_cache + block_id * cache_block_stride;
    scalar_t* dst_ptr = dst + pid * dst_entry_stride;

    // Decode Golay triplets in parallel
    const int triplet_total = num_heads * num_triplets;
    for (int i = tid; i < triplet_total; i += blockDim.x) {
      const int head_idx = i / num_triplets;
      const int triplet_idx = i % num_triplets;

      // Compute address for this triplet's int32 codeword
      // Address = block_base + head_idx * head_stride_bytes + triplet_idx * 4 * block_size + block_offset
      const uint8_t* head_base = block_base + head_idx * head_stride_bytes;
      const uint8_t* triplet_addr = head_base + triplet_idx * 4 * block_size + block_offset;

      // Read int32 codeword (4 bytes at strided positions)
      // Bytes are at: triplet_addr, triplet_addr + block_size, triplet_addr + 2*block_size, triplet_addr + 3*block_size
      int32_t codeword = static_cast<int32_t>(triplet_addr[0]) |
                         (static_cast<int32_t>(triplet_addr[block_size]) << 8) |
                         (static_cast<int32_t>(triplet_addr[2 * block_size]) << 16) |
                         (static_cast<int32_t>(triplet_addr[3 * block_size]) << 24);

      ecc::GolayErrorType error_type;
      scalar_t v0, v1, v2;
      ecc::int4_golay_decode_triplet<scalar_t>(codeword, scale_val, syndrome_lut,
                                               v0, v1, v2, &error_type);

      const int base_idx = head_idx * head_size + triplet_idx * 3;
      dst_ptr[base_idx + 0] = v0;
      dst_ptr[base_idx + 1] = v1;
      dst_ptr[base_idx + 2] = v2;

      if (golay_stats != nullptr) {
        const int stat_idx = static_cast<int>(error_type);
        atomicAdd(reinterpret_cast<unsigned long long*>(&golay_stats[stat_idx]), 1ULL);
      }
    }

    // Handle remainder elements (not in triplets) - use Hamming(8,4)
    // For head_size=64: 21 triplets (63 values) + 1 remainder
    // For head_size=128: 42 triplets (126 values) + 2 remainders
    if (remainder > 0) {
      const int remainder_total = num_heads * remainder;
      for (int i = tid; i < remainder_total; i += blockDim.x) {
        const int head_idx = i / remainder;
        const int rem_idx = i % remainder;

        // Compute address for this remainder byte
        // Address = block_base + head_idx * head_stride_bytes + (golay_bytes + rem_idx) * block_size + block_offset
        const uint8_t* head_base = block_base + head_idx * head_stride_bytes;
        const uint8_t* rem_addr = head_base + (golay_bytes + rem_idx) * block_size + block_offset;

        const uint8_t codeword = *rem_addr;
        ecc::ErrorType error_type;
        scalar_t decoded = ecc::int4_ecc_decode<scalar_t>(codeword, scale_val, &error_type);

        // Write to correct position: after triplet values for this head
        const int dst_idx = head_idx * head_size + num_triplets * 3 + rem_idx;
        dst_ptr[dst_idx] = decoded;

        // Track Hamming error statistics for remainder
        if (hamming_stats != nullptr) {
          const int stat_idx = static_cast<int>(error_type);
          atomicAdd(reinterpret_cast<unsigned long long*>(&hamming_stats[stat_idx]), 1ULL);
        }
      }
    }

    block_offset += 1;
    if (block_offset == block_size) {
      block_table_idx += 1;
      block_offset = 0;
    }
  }
}

}  // namespace vllm

// Host wrapper for ECC gather-decode
void cp_gather_and_ecc_decode_kv_cache(
    torch::Tensor const& src_cache_k,      // [NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE]
    torch::Tensor const& src_cache_v,
    torch::Tensor& dst_k,                   // [TOT_TOKENS, NUM_HEADS, HEAD_SIZE]
    torch::Tensor& dst_v,
    torch::Tensor const& block_table,       // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,       // [BATCH+1]
    torch::Tensor const& seq_starts,        // [BATCH]
    torch::Tensor const& k_scale,
    torch::Tensor const& v_scale,
    const std::string& kv_cache_dtype,
    std::optional<torch::Tensor> golay_syndrome_lut,
    std::optional<torch::Tensor> golay_stats,
    std::optional<torch::Tensor> hamming_stats) {
  at::cuda::OptionalCUDAGuard device_guard(src_cache_k.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t batch_size = block_table.size(0);
  int32_t block_size = src_cache_k.size(1);
  int32_t num_heads = dst_k.size(1);
  int32_t head_size = dst_k.size(2);

  TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must be int32");
  TORCH_CHECK(cu_seq_lens.dtype() == torch::kInt32, "cu_seq_lens must be int32");
  TORCH_CHECK(seq_starts.dtype() == torch::kInt32, "seq_starts must be int32");

  TORCH_CHECK(src_cache_k.device() == dst_k.device(),
              "src_cache_k and dst_k must be on the same device");
  TORCH_CHECK(src_cache_v.device() == dst_v.device(),
              "src_cache_v and dst_v must be on the same device");

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride_k = src_cache_k.stride(0);
  int64_t cache_entry_stride_k = src_cache_k.stride(1);
  int64_t dst_entry_stride_k = dst_k.stride(0);
  int64_t cache_block_stride_v = src_cache_v.stride(0);
  int64_t cache_entry_stride_v = src_cache_v.stride(1);
  int64_t dst_entry_stride_v = dst_v.stride(0);

  // Decide on the number of splits based on the batch size
  // Larger batches need more splits for better parallelism
  // Smaller batches can use more splits per sequence
  int num_splits = batch_size < 32 ? 16 : batch_size < 64 ? 8 : batch_size < 128 ? 4 : 2;
  dim3 grid(batch_size, num_splits);
  dim3 block(256);  // 256 threads per block

  const int32_t* seq_starts_ptr = seq_starts.data_ptr<int32_t>();
  int64_t* hamming_stats_ptr = hamming_stats.has_value() ?
                               hamming_stats.value().data_ptr<int64_t>() : nullptr;
  int64_t* golay_stats_ptr = golay_stats.has_value() ?
                             golay_stats.value().data_ptr<int64_t>() : nullptr;

  if (kv_cache_dtype == "int4_ecc") {
    // Hamming(8,4) SECDED
    TORCH_CHECK(dst_k.dtype() == torch::kFloat16 || dst_k.dtype() == torch::kBFloat16,
                "dst must be float16 or bfloat16");

    if (dst_k.dtype() == torch::kFloat16) {
      vllm::cp_gather_and_ecc_decode_hamming84_kernel<__half><<<grid, block, 0, stream>>>(
          src_cache_k.data_ptr<uint8_t>(),
          reinterpret_cast<__half*>(dst_k.data_ptr()),
          block_table.data_ptr<int32_t>(),
          cu_seq_lens.data_ptr<int32_t>(),
          seq_starts_ptr,
          k_scale.data_ptr<float>(),
          hamming_stats_ptr,
          block_size, num_heads, head_size,
          block_table_stride, cache_block_stride_k, cache_entry_stride_k, dst_entry_stride_k);

      vllm::cp_gather_and_ecc_decode_hamming84_kernel<__half><<<grid, block, 0, stream>>>(
          src_cache_v.data_ptr<uint8_t>(),
          reinterpret_cast<__half*>(dst_v.data_ptr()),
          block_table.data_ptr<int32_t>(),
          cu_seq_lens.data_ptr<int32_t>(),
          seq_starts_ptr,
          v_scale.data_ptr<float>(),
          hamming_stats_ptr,
          block_size, num_heads, head_size,
          block_table_stride, cache_block_stride_v, cache_entry_stride_v, dst_entry_stride_v);
    } else {
      // bfloat16
      vllm::cp_gather_and_ecc_decode_hamming84_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
          src_cache_k.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(dst_k.data_ptr()),
          block_table.data_ptr<int32_t>(),
          cu_seq_lens.data_ptr<int32_t>(),
          seq_starts_ptr,
          k_scale.data_ptr<float>(),
          hamming_stats_ptr,
          block_size, num_heads, head_size,
          block_table_stride, cache_block_stride_k, cache_entry_stride_k, dst_entry_stride_k);

      vllm::cp_gather_and_ecc_decode_hamming84_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
          src_cache_v.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(dst_v.data_ptr()),
          block_table.data_ptr<int32_t>(),
          cu_seq_lens.data_ptr<int32_t>(),
          seq_starts_ptr,
          v_scale.data_ptr<float>(),
          hamming_stats_ptr,
          block_size, num_heads, head_size,
          block_table_stride, cache_block_stride_v, cache_entry_stride_v, dst_entry_stride_v);
    }
  } else if (kv_cache_dtype == "int4_h74") {
    // Hamming(7,4) SEC
    if (dst_k.dtype() == torch::kFloat16) {
      vllm::cp_gather_and_ecc_decode_hamming74_kernel<__half><<<grid, block, 0, stream>>>(
          src_cache_k.data_ptr<uint8_t>(),
          reinterpret_cast<__half*>(dst_k.data_ptr()),
          block_table.data_ptr<int32_t>(),
          cu_seq_lens.data_ptr<int32_t>(),
          seq_starts_ptr,
          k_scale.data_ptr<float>(),
          hamming_stats_ptr,
          block_size, num_heads, head_size,
          block_table_stride, cache_block_stride_k, cache_entry_stride_k, dst_entry_stride_k);

      vllm::cp_gather_and_ecc_decode_hamming74_kernel<__half><<<grid, block, 0, stream>>>(
          src_cache_v.data_ptr<uint8_t>(),
          reinterpret_cast<__half*>(dst_v.data_ptr()),
          block_table.data_ptr<int32_t>(),
          cu_seq_lens.data_ptr<int32_t>(),
          seq_starts_ptr,
          v_scale.data_ptr<float>(),
          hamming_stats_ptr,
          block_size, num_heads, head_size,
          block_table_stride, cache_block_stride_v, cache_entry_stride_v, dst_entry_stride_v);
    } else {
      vllm::cp_gather_and_ecc_decode_hamming74_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
          src_cache_k.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(dst_k.data_ptr()),
          block_table.data_ptr<int32_t>(),
          cu_seq_lens.data_ptr<int32_t>(),
          seq_starts_ptr,
          k_scale.data_ptr<float>(),
          hamming_stats_ptr,
          block_size, num_heads, head_size,
          block_table_stride, cache_block_stride_k, cache_entry_stride_k, dst_entry_stride_k);

      vllm::cp_gather_and_ecc_decode_hamming74_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
          src_cache_v.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(dst_v.data_ptr()),
          block_table.data_ptr<int32_t>(),
          cu_seq_lens.data_ptr<int32_t>(),
          seq_starts_ptr,
          v_scale.data_ptr<float>(),
          hamming_stats_ptr,
          block_size, num_heads, head_size,
          block_table_stride, cache_block_stride_v, cache_entry_stride_v, dst_entry_stride_v);
    }
  } else if (kv_cache_dtype == "int4_golay" || kv_cache_dtype == "int4_golay_hybrid") {
    // Golay(24,12) TEC or Hybrid Golay + Hamming
    // Both use the same kernel that handles triplets (Golay) and remainder (Hamming)
    TORCH_CHECK(golay_syndrome_lut.has_value(),
                "golay_syndrome_lut required for ", kv_cache_dtype);
    const int32_t* syndrome_lut_ptr = golay_syndrome_lut.value().data_ptr<int32_t>();

    if (dst_k.dtype() == torch::kFloat16) {
      vllm::cp_gather_and_ecc_decode_golay_kernel<__half><<<grid, block, 0, stream>>>(
          src_cache_k.data_ptr<uint8_t>(),
          reinterpret_cast<__half*>(dst_k.data_ptr()),
          block_table.data_ptr<int32_t>(),
          cu_seq_lens.data_ptr<int32_t>(),
          seq_starts_ptr,
          k_scale.data_ptr<float>(),
          syndrome_lut_ptr,
          golay_stats_ptr,
          hamming_stats_ptr,
          block_size, num_heads, head_size,
          block_table_stride, dst_entry_stride_k);

      vllm::cp_gather_and_ecc_decode_golay_kernel<__half><<<grid, block, 0, stream>>>(
          src_cache_v.data_ptr<uint8_t>(),
          reinterpret_cast<__half*>(dst_v.data_ptr()),
          block_table.data_ptr<int32_t>(),
          cu_seq_lens.data_ptr<int32_t>(),
          seq_starts_ptr,
          v_scale.data_ptr<float>(),
          syndrome_lut_ptr,
          golay_stats_ptr,
          hamming_stats_ptr,
          block_size, num_heads, head_size,
          block_table_stride, dst_entry_stride_v);
    } else {
      vllm::cp_gather_and_ecc_decode_golay_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
          src_cache_k.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(dst_k.data_ptr()),
          block_table.data_ptr<int32_t>(),
          cu_seq_lens.data_ptr<int32_t>(),
          seq_starts_ptr,
          k_scale.data_ptr<float>(),
          syndrome_lut_ptr,
          golay_stats_ptr,
          hamming_stats_ptr,
          block_size, num_heads, head_size,
          block_table_stride, dst_entry_stride_k);

      vllm::cp_gather_and_ecc_decode_golay_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
          src_cache_v.data_ptr<uint8_t>(),
          reinterpret_cast<__nv_bfloat16*>(dst_v.data_ptr()),
          block_table.data_ptr<int32_t>(),
          cu_seq_lens.data_ptr<int32_t>(),
          seq_starts_ptr,
          v_scale.data_ptr<float>(),
          syndrome_lut_ptr,
          golay_stats_ptr,
          hamming_stats_ptr,
          block_size, num_heads, head_size,
          block_table_stride, dst_entry_stride_v);
    }
  } else {
    TORCH_CHECK(false, "Unsupported kv_cache_dtype: ", kv_cache_dtype);
  }

  // Check for CUDA errors after kernel launch
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "cp_gather_and_ecc_decode_kv_cache kernel failed: ",
              cudaGetErrorString(err));
}

// Macro to dispatch the kernel based on the data type.
#define CALL_INDEXER_K_QUANT_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)         \
  vllm::indexer_k_quant_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>       \
      <<<grid, block, 0, stream>>>(                                     \
          reinterpret_cast<KV_T*>(k.data_ptr()),                        \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), head_dim, quant_block_size, \
          cache_block_size, cache_stride, use_ue8m0);

void indexer_k_quant_and_cache(
    torch::Tensor& k,             // [num_tokens, head_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, cache_stride]
    torch::Tensor& slot_mapping,  // [num_tokens]
    int64_t quant_block_size,     // quantization block size
    const std::string& scale_fmt) {
  int num_tokens = k.size(0);
  int head_dim = k.size(1);
  int cache_block_size = kv_cache.size(1);
  int cache_stride = kv_cache.size(2);
  bool use_ue8m0 = scale_fmt == "ue8m0";

  TORCH_CHECK(k.device() == kv_cache.device(),
              "k and kv_cache must be on the same device");
  TORCH_CHECK(k.device() == slot_mapping.device(),
              "k and slot_mapping must be on the same device");
  TORCH_CHECK(head_dim % quant_block_size == 0,
              "head_dim must be divisible by quant_block_size");

  constexpr int vec_size = 4;
  dim3 grid(num_tokens, (head_dim + quant_block_size * vec_size - 1) /
                            (quant_block_size * vec_size));
  dim3 block(32, vec_size);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(k));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(k.dtype(), "fp8_e4m3",
                             CALL_INDEXER_K_QUANT_AND_CACHE);
}

// Macro to dispatch the kernel based on the data amount.
#define CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(BLOCK_Y_SIZE)                  \
  vllm::cp_gather_indexer_k_quant_cache_kernel<BLOCK_Y_SIZE>                \
      <<<dim3((num_tokens + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE,               \
              (head_dim + 8 * vec_size - 1) / (8 * vec_size)),              \
         dim3(8, BLOCK_Y_SIZE), 0, stream>>>(                               \
          reinterpret_cast<char*>(kv_cache.data_ptr()),                     \
          reinterpret_cast<char*>(dst_k.data_ptr()),                        \
          reinterpret_cast<char*>(dst_scale.data_ptr()),                    \
          block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(), \
          batch_size, dst_k.stride(0), dst_k.size(1), kv_cache.stride(0),   \
          kv_cache.stride(1), kv_cache.size(1), block_table.size(1),        \
          num_tokens, quant_block_size);

void cp_gather_indexer_k_quant_cache(
    const torch::Tensor& kv_cache,  // [num_blocks, block_size, cache_stride]
    torch::Tensor& dst_k,           // [num_tokens, head_dim]
    torch::Tensor& dst_scale,  // [num_tokens, head_dim / quant_block_size * 4]
    const torch::Tensor& block_table,  // [batch_size, num_blocks]
    const torch::Tensor& cu_seq_lens   // [batch_size + 1]
) {
  int batch_size = block_table.size(0);
  int num_tokens = dst_k.size(0);
  int head_dim = dst_k.size(1);
  int quant_block_size = head_dim * 4 / dst_scale.size(1);

  TORCH_CHECK(kv_cache.device() == dst_k.device(),
              "kv_cache and dst_k must be on the same device");
  TORCH_CHECK(kv_cache.device() == dst_scale.device(),
              "kv_cache and dst_scale must be on the same device");
  TORCH_CHECK(kv_cache.device() == block_table.device(),
              "kv_cache and block_table must be on the same device");
  TORCH_CHECK(kv_cache.device() == cu_seq_lens.device(),
              "kv_cache and cu_seq_lens must be on the same device");
  TORCH_CHECK(head_dim % quant_block_size == 0,
              "head_dim must be divisible by quant_block_size");

  constexpr int vec_size = 16;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (num_tokens < 32) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(1);
  } else if (num_tokens < 64) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(2);
  } else if (num_tokens < 128) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(4);
  } else if (num_tokens < 256) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(8);
  } else if (num_tokens < 512) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(16);
  } else {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(32);
  }
}

// =====================================================================
// ECC Fault Injection for INT4-ECC Protected KV Cache
// =====================================================================

void inject_cache_errors(
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    double bit_error_rate,
    int64_t seed
) {
  // Validate inputs
  TORCH_CHECK(key_cache.is_cuda(), "key_cache must be a CUDA tensor");
  TORCH_CHECK(value_cache.is_cuda(), "value_cache must be a CUDA tensor");
  TORCH_CHECK(key_cache.dtype() == torch::kUInt8,
              "key_cache must be uint8 (ECC-encoded INT4)");
  TORCH_CHECK(value_cache.dtype() == torch::kUInt8,
              "value_cache must be uint8 (ECC-encoded INT4)");
  TORCH_CHECK(bit_error_rate >= 0.0 && bit_error_rate <= 1.0,
              "bit_error_rate must be in [0, 1]");

  if (bit_error_rate <= 0.0) {
    // No errors to inject
    return;
  }

  const int64_t k_size = key_cache.numel();
  const int64_t v_size = value_cache.numel();

  if (k_size == 0 && v_size == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Launch fault injection kernels
  const int block_size = 256;

  if (k_size > 0) {
    const int k_blocks = (k_size + block_size - 1) / block_size;
    vllm::ecc::fault_inject_uint8_kernel<<<k_blocks, block_size, 0, stream>>>(
        key_cache.data_ptr<uint8_t>(),
        k_size,
        static_cast<float>(bit_error_rate),
        static_cast<uint64_t>(seed),
        8,  // all 8 bits
        nullptr  // don't track count
    );
  }

  if (v_size > 0) {
    // Use different seed offset for value cache to ensure independence
    const int v_blocks = (v_size + block_size - 1) / block_size;
    vllm::ecc::fault_inject_uint8_kernel<<<v_blocks, block_size, 0, stream>>>(
        value_cache.data_ptr<uint8_t>(),
        v_size,
        static_cast<float>(bit_error_rate),
        static_cast<uint64_t>(seed + k_size),  // offset seed
        8,
        nullptr
    );
  }
}
