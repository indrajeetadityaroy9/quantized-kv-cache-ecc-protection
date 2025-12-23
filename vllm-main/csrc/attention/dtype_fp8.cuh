#pragma once

#include "attention_generic.cuh"

#include <stdint.h>
#ifdef ENABLE_FP8
  #ifndef USE_ROCM
    #include <cuda_fp8.h>
  #endif  // USE_ROCM
#endif    // ENABLE_FP8

namespace vllm {

enum class Fp8KVCacheDataType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
  kInt4Ecc = 3,        // INT4 quantization + Hamming(8,4) SECDED ECC
  kInt4Hamming74 = 4,  // INT4 quantization + Hamming(7,4) SEC ECC
  kInt4Golay = 5,      // INT4 quantization + Golay(24,12) 3-error correction
  kInt4GolayHybrid = 6,  // Hybrid: Golay(24,12) for triplets + Hamming(8,4) for remainder
  kInt4ReedSolomon = 7,  // INT4 + Reed-Solomon(12,8) - 2-symbol correction
};

// fp8 vector types for quantization of kv cache
template <>
struct Vec<uint8_t, 1> {
  using Type = uint8_t;
};

template <>
struct Vec<uint8_t, 2> {
  using Type = uint16_t;
};

template <>
struct Vec<uint8_t, 4> {
  using Type = uint32_t;
};

template <>
struct Vec<uint8_t, 8> {
  using Type = uint2;
};

// int32 vector types for Golay(24,12) ECC (one codeword per 3 values)
template <>
struct Vec<int32_t, 1> {
  using Type = int32_t;
};

template <>
struct Vec<int32_t, 2> {
  using Type = int2;
};

template <>
struct Vec<int32_t, 4> {
  using Type = int4;
};

}  // namespace vllm
