
FILE(GLOB TEST_SRCS *.cpp)

foreach(FILE ${TEST_SRCS})
  get_filename_component(BINARY ${FILE} NAME_WE)
  add_library(${BINARY} ${FILE})
  target_link_libraries(${BINARY} PUBLIC Kokkos::kokkos ${PUBLIC_DEPS})
endforeach()

