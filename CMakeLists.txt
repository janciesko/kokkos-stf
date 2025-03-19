cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

# Disable in-source builds to prevent source tree corruption.
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(
    FATAL_ERROR
  )
endif()

project(KokkosSTF CXX)
include(cmake/CPM.cmake)

option(KOKKOSSTF_ENABLE_EXAMPLES "Whether to build examples" ON)

# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY nvidia/cccl
    GIT_TAG main
    # The following is required to make the `CCCL::cudax` target available:
    OPTIONS "CCCL_ENABLE_UNSTABLE ON"
)

# Look for deps
find_package(Kokkos REQUIRED)

if(KOKKOSSTF_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()