# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/qutlass-src")
  file(MAKE_DIRECTORY "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/qutlass-src")
endif()
file(MAKE_DIRECTORY
  "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/qutlass-build"
  "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/qutlass-subbuild/qutlass-populate-prefix"
  "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/qutlass-subbuild/qutlass-populate-prefix/tmp"
  "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/qutlass-subbuild/qutlass-populate-prefix/src/qutlass-populate-stamp"
  "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/qutlass-subbuild/qutlass-populate-prefix/src"
  "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/qutlass-subbuild/qutlass-populate-prefix/src/qutlass-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/qutlass-subbuild/qutlass-populate-prefix/src/qutlass-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ubuntu/quantized-kv-cache-ecc-protection/vllm-main/.deps/qutlass-subbuild/qutlass-populate-prefix/src/qutlass-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
