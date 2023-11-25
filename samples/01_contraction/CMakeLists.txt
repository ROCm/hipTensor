###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 # THE SOFTWARE.
 #
 ###############################################################################

# Check whether building within hiptensor context
if( CMAKE_PROJECT_NAME STREQUAL "hiptensor" )
    add_hiptensor_sample(simple_contraction_scale_f16 simple_scale_contraction_f16.cpp)
    add_hiptensor_sample(simple_contraction_scale_bf16 simple_scale_contraction_bf16.cpp)
    add_hiptensor_sample(simple_contraction_scale_f32 simple_scale_contraction_f32.cpp)
    add_hiptensor_sample(simple_contraction_scale_f64 simple_scale_contraction_f64.cpp)
    add_hiptensor_sample(simple_contraction_scale_f64_f32 simple_scale_contraction_f64_f32.cpp)
    add_hiptensor_sample(simple_contraction_scale_f32_f16 simple_scale_contraction_f32_f16.cpp)
    add_hiptensor_sample(simple_contraction_scale_f32_bf16 simple_scale_contraction_f32_bf16.cpp)
    add_hiptensor_sample(simple_contraction_bilinear_f16 simple_bilinear_contraction_f16.cpp)
    add_hiptensor_sample(simple_contraction_bilinear_bf16 simple_bilinear_contraction_bf16.cpp)
    add_hiptensor_sample(simple_contraction_bilinear_f32 simple_bilinear_contraction_f32.cpp)
    add_hiptensor_sample(simple_contraction_bilinear_f64 simple_bilinear_contraction_f64.cpp)
    add_hiptensor_sample(simple_contraction_bilinear_f64_f32 simple_bilinear_contraction_f64_f32.cpp)
    add_hiptensor_sample(simple_contraction_bilinear_f32_f16 simple_bilinear_contraction_f32_f16.cpp)
    add_hiptensor_sample(simple_contraction_bilinear_f32_bf16 simple_bilinear_contraction_f32_bf16.cpp)

# If building hipTensor samples as a standalone Cmake project
else()
    add_executable(simple_contraction_scale_f16 simple_scale_contraction_f16.cpp)
    target_link_libraries(simple_contraction_scale_f16 PRIVATE hiptensor::hiptensor)

    add_executable(simple_contraction_scale_bf16 simple_scale_contraction_bf16.cpp)
    target_link_libraries(simple_contraction_scale_bf16 PRIVATE hiptensor::hiptensor)

    add_executable(simple_contraction_scale_f32 simple_scale_contraction_f32.cpp)
    target_link_libraries(simple_contraction_scale_f32 PRIVATE hiptensor::hiptensor)

    add_executable(simple_contraction_scale_f32_bf16 simple_scale_contraction_f32_bf16.cpp)
    target_link_libraries(simple_contraction_scale_f32_bf16 PRIVATE hiptensor::hiptensor)

    add_executable(simple_contraction_scale_f32_f16 simple_scale_contraction_f32_f16.cpp)
    target_link_libraries(simple_contraction_scale_f32_f16 PRIVATE hiptensor::hiptensor)

    add_executable(simple_contraction_scale_f64 simple_scale_contraction_f64.cpp)
    target_link_libraries(simple_contraction_scale_f64 PRIVATE hiptensor::hiptensor)

    add_executable(simple_contraction_scale_f64_f32 simple_scale_contraction_f64_f32.cpp)
    target_link_libraries(simple_contraction_scale_f64_f32 PRIVATE hiptensor::hiptensor)

    add_executable(simple_contraction_bilinear_f16 simple_bilinear_contraction_f16.cpp)
    target_link_libraries(simple_contraction_bilinear_f16 PRIVATE hiptensor::hiptensor)

    add_executable(simple_contraction_bilinear_bf16 simple_bilinear_contraction_bf16.cpp)
    target_link_libraries(simple_contraction_bilinear_bf16 PRIVATE hiptensor::hiptensor)

    add_executable(simple_contraction_bilinear_f32 simple_bilinear_contraction_f32.cpp)
    target_link_libraries(simple_contraction_bilinear_f32 PRIVATE hiptensor::hiptensor)

    add_executable(simple_contraction_bilinear_f32_bf16 simple_bilinear_contraction_f32_bf16.cpp)
    target_link_libraries(simple_contraction_bilinear_f32_bf16 PRIVATE hiptensor::hiptensor)

    add_executable(simple_contraction_bilinear_f32_f16 simple_bilinear_contraction_f32_f16.cpp)
    target_link_libraries(simple_contraction_bilinear_f32_f16 PRIVATE hiptensor::hiptensor)

    add_executable(simple_contraction_bilinear_f64 simple_bilinear_contraction_f64.cpp)
    target_link_libraries(simple_contraction_bilinear_f64 PRIVATE hiptensor::hiptensor)

    add_executable(simple_contraction_bilinear_f64_f32 simple_bilinear_contraction_f64_f32.cpp)
    target_link_libraries(simple_contraction_bilinear_f64_f32 PRIVATE hiptensor::hiptensor)
endif()
