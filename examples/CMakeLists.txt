#FILE(GLOB TEST_SRCS *.cpp)
FILE(GLOB TEST_SRCS axpy.cpp)
FILE(GLOB TEST_SRCS axpy-annotated.cpp)

foreach(FILE ${TEST_SRCS})
  get_filename_component(BINARY ${FILE} NAME_WE)
  add_library(${BINARY} ${FILE})
  target_link_libraries(${BINARY} PUBLIC Kokkos::kokkos CCCL::CCCL CCCL::cudax)
endforeach()

