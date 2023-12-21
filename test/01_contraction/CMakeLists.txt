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
set(ContractionCommonSources ${HIPTENSOR_COMMON_TEST_SOURCES}
                             ${CMAKE_CURRENT_SOURCE_DIR}/contraction_resource.cpp
                             ${CMAKE_CURRENT_SOURCE_DIR}/contraction_test.cpp)

# Bilinear tests
set (BilinearContractionTestSources ${ContractionCommonSources}
                                    ${CMAKE_CURRENT_SOURCE_DIR}/bilinear_contraction_test.cpp)
set (BilinearContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bilinear_test_params.yaml)
add_hiptensor_test(bilinear_contraction_test ${BilinearContractionTestConfig}  ${BilinearContractionTestSources})

# Complex Bilinear tests
set (ComplexBilinearContractionTestSources ${ContractionCommonSources}
    ${CMAKE_CURRENT_SOURCE_DIR}/complex_bilinear_contraction_test.cpp)
set (ComplexBilinearContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/complex_bilinear_test_params.yaml)
add_hiptensor_test(complex_bilinear_contraction_test ${ComplexBilinearContractionTestConfig}  ${ComplexBilinearContractionTestSources})

# Scale tests
set (ScaleContractionTestSources ${ContractionCommonSources}
                                    ${CMAKE_CURRENT_SOURCE_DIR}/scale_contraction_test.cpp)
set (ScaleContractionTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/scale_test_params.yaml)
add_hiptensor_test(scale_contraction_test ${ScaleContractionTestConfig} ${ScaleContractionTestSources})


