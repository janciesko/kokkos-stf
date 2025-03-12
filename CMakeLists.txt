cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

# Disable in-source builds to prevent source tree corruption.
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(
    FATAL_ERROR
  )
endif()

project(KokkosSTF CXX)

#What would we like to do today
option(KOKKOSSTF_ENABLE_EXAMPLES "Whether to build examples" OFF)
option(KOKKOSSTF_ENABLE_CUDASTF "Whether to build with CUDA STF" OFF)

# Look for deps
find_package(Kokkos REQUIRED)
if(${KOKKOSSTF_ENABLE_CUDASTF})
  find_package(cudax REQUIRED)
  list(APPEND PUBLIC_DEPS cudax)
endif()

if(NOT KOKKOSSTF_ENABLE_CUDASTF)
 message(FATAL_ERROR "Must enable CUDA STF.")
endif()

if(KOKKOSSTF_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()