/*
 * INT32 vector type definitions for ECC cache storage.
 * Copyright (c) 2024, The vLLM team.
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
#pragma once

#include "attention_generic.cuh"

#include <stdint.h>

namespace vllm {

// Define custom INT32 vector data types for ECC-protected cache storage.
struct Int32_2 {
  int32_t x;
  int32_t y;
};

struct Int32_4 {
  int32_t x;
  int32_t y;
  int32_t z;
  int32_t w;
};

struct Int32_8 {
  int32_t data[8];
};

struct Int32_16 {
  int32_t data[16];
};

// INT32 vector types for ECC cache (Q, K, V storage).
// NOTE: Vec<int32_t, 1>, Vec<int32_t, 2>, Vec<int32_t, 4> are already defined
// in dtype_fp8.cuh. Only define the larger sizes here.

template <>
struct Vec<int32_t, 8> {
  using Type = Int32_8;
};

template <>
struct Vec<int32_t, 16> {
  using Type = Int32_16;
};

// Zero-out functions for int32 types.
inline __device__ void zero(int32_t& dst) { dst = 0; }

inline __device__ void zero(Int32_2& dst) {
  dst.x = 0;
  dst.y = 0;
}

inline __device__ void zero(Int32_4& dst) {
  dst.x = 0;
  dst.y = 0;
  dst.z = 0;
  dst.w = 0;
}

inline __device__ void zero(Int32_8& dst) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    dst.data[i] = 0;
  }
}

inline __device__ void zero(Int32_16& dst) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    dst.data[i] = 0;
  }
}

}  // namespace vllm
