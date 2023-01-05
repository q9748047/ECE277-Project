set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

add_subdirectory(pybind11_cuda_add)
add_subdirectory(pybind11_cuda_sub)
add_subdirectory(pybind11_cuda_mul)
add_subdirectory(pybind11_cuda_trans)
