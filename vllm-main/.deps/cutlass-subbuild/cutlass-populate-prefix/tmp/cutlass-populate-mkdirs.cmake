# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/cutlass-src")
  file(MAKE_DIRECTORY "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/cutlass-src")
endif()
file(MAKE_DIRECTORY
  "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/cutlass-build"
  "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/cutlass-subbuild/cutlass-populate-prefix"
  "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/cutlass-subbuild/cutlass-populate-prefix/tmp"
  "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp"
  "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/cutlass-subbuild/cutlass-populate-prefix/src"
  "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/cutlass-subbuild/cutlass-populate-prefix/src/cutlass-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
