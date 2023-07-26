###############################################################################
 #
 # MIT License
 #
 # Copyright (c) 2023 Advanced Micro Devices, Inc.
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
set(ContractionCommonSources ${HIPTENSOR_COMMON_TEST_SOURCES}
                             ${CMAKE_CURRENT_SOURCE_DIR}/contraction_resource.cpp)

set(TestYamlSrc ${HIPTENSOR_COMMON_TEST_SOURCES}
                             ${CMAKE_CURRENT_SOURCE_DIR}/yaml_test.cpp)


set(BilinearContractionAcF32Sources ${ContractionCommonSources}
                                    ${CMAKE_CURRENT_SOURCE_DIR}/bilinear_contraction_ac_f32.cpp)

# Brute force tests
add_hiptensor_test(scale_contraction_f32_test scale_contraction_f32.cpp)
add_hiptensor_test(scale_contraction_f64_test scale_contraction_f64.cpp)
add_hiptensor_test(bilinear_contraction_f32_test bilinear_contraction_f32.cpp)
add_hiptensor_test(bilinear_contraction_f64_test bilinear_contraction_f64.cpp)

# Actor-critic tests
add_hiptensor_test(scale_contraction_ac_f32_test scale_contraction_ac_f32.cpp)
add_hiptensor_test(scale_contraction_ac_f64_test scale_contraction_ac_f64.cpp)
add_hiptensor_test(bilinear_contraction_ac_f32_test ${BilinearContractionAcF32Sources})
add_hiptensor_test(bilinear_contraction_ac_f64_test bilinear_contraction_ac_f64.cpp)
add_hiptensor_test(yaml_test ${TestYamlSrc})
