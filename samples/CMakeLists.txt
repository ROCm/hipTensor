cmake_minimum_required(VERSION 3.14)
project(hiptensor_app)
add_compile_options(-std=c++17)

find_package(hiptensor 1.0.0 REQUIRED COMPONENTS hiptensor)
find_package(hip REQUIRED PATHS /opt/rocm)

message(STATUS "Build with HIP ${hip_VERSION}")

add_subdirectory(01_contraction)