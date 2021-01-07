cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(tt)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(tt_top torch_wrap.cpp)
target_link_libraries(tt_top "${TORCH_LIBRARIES}")
set_property(TARGET tt_top PROPERTY CXX_STANDARD 14)



