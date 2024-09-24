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

# Bilinear M1N1K1 tests
set (BilinearContractionTestSources ${ContractionCommonSources}
                                    ${CMAKE_CURRENT_SOURCE_DIR}/bilinear_contraction_test.cpp)
set (BilinearContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/bilinear_test_params_rank1.yaml)
set (BilinearContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/bilinear_test_params_rank1.yaml)
set (BilinearContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/bilinear_test_params_rank1.yaml)
add_hiptensor_test(bilinear_contraction_test_m1n1k1-validate ${BilinearContractionTestConfigValidate}  ${BilinearContractionTestSources})
add_hiptensor_test(bilinear_contraction_test_m1n1k1-bench ${BilinearContractionTestConfigBench}  ${BilinearContractionTestSources})
add_hiptensor_test(bilinear_contraction_test_m1n1k1-extended ${BilinearContractionTestConfigExtended}  ${BilinearContractionTestSources})

# Bilinear M2N2K2 tests
set (BilinearContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/bilinear_test_params_rank2.yaml)
set (BilinearContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/bilinear_test_params_rank2.yaml)
set (BilinearContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/bilinear_test_params_rank2.yaml)
add_hiptensor_test(bilinear_contraction_test_m2n2k2-validate ${BilinearContractionTestConfigValidate}  ${BilinearContractionTestSources})
add_hiptensor_test(bilinear_contraction_test_m2n2k2-bench ${BilinearContractionTestConfigBench}  ${BilinearContractionTestSources})
add_hiptensor_test(bilinear_contraction_test_m2n2k2-extended ${BilinearContractionTestConfigExtended}  ${BilinearContractionTestSources})

# Bilinear M3N3K3 tests
set (BilinearContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/bilinear_test_params_rank3.yaml)
set (BilinearContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/bilinear_test_params_rank3.yaml)
set (BilinearContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/bilinear_test_params_rank3.yaml)
add_hiptensor_test(bilinear_contraction_test_m3n3k3-validate ${BilinearContractionTestConfigValidate}  ${BilinearContractionTestSources})
add_hiptensor_test(bilinear_contraction_test_m3n3k3-bench ${BilinearContractionTestConfigBench}  ${BilinearContractionTestSources})
add_hiptensor_test(bilinear_contraction_test_m3n3k3-extended ${BilinearContractionTestConfigExtended}  ${BilinearContractionTestSources})

# Bilinear M4N4K4 tests
set (BilinearContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/bilinear_test_params_rank4.yaml)
set (BilinearContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/bilinear_test_params_rank4.yaml)
set (BilinearContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/bilinear_test_params_rank4.yaml)
add_hiptensor_test(bilinear_contraction_test_m4n4k4-validate ${BilinearContractionTestConfigValidate}  ${BilinearContractionTestSources})
add_hiptensor_test(bilinear_contraction_test_m4n4k4-bench ${BilinearContractionTestConfigBench}  ${BilinearContractionTestSources})
add_hiptensor_test(bilinear_contraction_test_m4n4k4-extended ${BilinearContractionTestConfigExtended}  ${BilinearContractionTestSources})

# Bilinear M5N5K5 tests
set (BilinearContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/bilinear_test_params_rank5.yaml)
set (BilinearContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/bilinear_test_params_rank5.yaml)
set (BilinearContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/bilinear_test_params_rank5.yaml)
add_hiptensor_test(bilinear_contraction_test_m5n5k5-validate ${BilinearContractionTestConfigValidate}  ${BilinearContractionTestSources})
add_hiptensor_test(bilinear_contraction_test_m5n5k5-bench ${BilinearContractionTestConfigBench}  ${BilinearContractionTestSources})
add_hiptensor_test(bilinear_contraction_test_m5n5k5-extended ${BilinearContractionTestConfigExtended}  ${BilinearContractionTestSources})

# Bilinear M6N6K6 tests
set (BilinearContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/bilinear_test_params_rank6.yaml)
set (BilinearContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/bilinear_test_params_rank6.yaml)
set (BilinearContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/bilinear_test_params_rank6.yaml)
add_hiptensor_test(bilinear_contraction_test_m6n6k6-validate ${BilinearContractionTestConfigValidate}  ${BilinearContractionTestSources})
add_hiptensor_test(bilinear_contraction_test_m6n6k6-bench ${BilinearContractionTestConfigBench}  ${BilinearContractionTestSources})
add_hiptensor_test(bilinear_contraction_test_m6n6k6-extended ${BilinearContractionTestConfigExtended}  ${BilinearContractionTestSources})

# Complex Bilinear M1N1K1 tests
set (ComplexBilinearContractionTestSources ${ContractionCommonSources}
    ${CMAKE_CURRENT_SOURCE_DIR}/complex_bilinear_contraction_test.cpp)
set (ComplexBilinearContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_bilinear_test_params_rank1.yaml)
set (ComplexBilinearContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/complex_bilinear_test_params_rank1.yaml)
set (ComplexBilinearContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/complex_bilinear_test_params_rank1.yaml)
add_hiptensor_test(complex_bilinear_contraction_test_m1n1k1-validate ${ComplexBilinearContractionTestConfigValidate}  ${ComplexBilinearContractionTestSources})
add_hiptensor_test(complex_bilinear_contraction_test_m1n1k1-bench ${ComplexBilinearContractionTestConfigBench}  ${ComplexBilinearContractionTestSources})
add_hiptensor_test(complex_bilinear_contraction_test_m1n1k1-extended ${ComplexBilinearContractionTestConfigExtended}  ${ComplexBilinearContractionTestSources})

# Complex Bilinear M2N2K2 tests
set (ComplexBilinearContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_bilinear_test_params_rank2.yaml)
set (ComplexBilinearContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/complex_bilinear_test_params_rank2.yaml)
set (ComplexBilinearContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/complex_bilinear_test_params_rank2.yaml)
add_hiptensor_test(complex_bilinear_contraction_test_m2n2k2-validate ${ComplexBilinearContractionTestConfigValidate}  ${ComplexBilinearContractionTestSources})
add_hiptensor_test(complex_bilinear_contraction_test_m2n2k2-bench ${ComplexBilinearContractionTestConfigBench}  ${ComplexBilinearContractionTestSources})
add_hiptensor_test(complex_bilinear_contraction_test_m2n2k2-extended ${ComplexBilinearContractionTestConfigExtended}  ${ComplexBilinearContractionTestSources})

# Complex Bilinear M3N3K3 tests
set (ComplexBilinearContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_bilinear_test_params_rank3.yaml)
set (ComplexBilinearContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/complex_bilinear_test_params_rank3.yaml)
set (ComplexBilinearContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/complex_bilinear_test_params_rank3.yaml)
add_hiptensor_test(complex_bilinear_contraction_test_m3n3k3-validate ${ComplexBilinearContractionTestConfigValidate}  ${ComplexBilinearContractionTestSources})
add_hiptensor_test(complex_bilinear_contraction_test_m3n3k3-bench ${ComplexBilinearContractionTestConfigBench}  ${ComplexBilinearContractionTestSources})
add_hiptensor_test(complex_bilinear_contraction_test_m3n3k3-extended ${ComplexBilinearContractionTestConfigExtended}  ${ComplexBilinearContractionTestSources})

# Complex Bilinear M4N4K4 tests
set (ComplexBilinearContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_bilinear_test_params_rank4.yaml)
set (ComplexBilinearContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/complex_bilinear_test_params_rank4.yaml)
set (ComplexBilinearContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/complex_bilinear_test_params_rank4.yaml)
add_hiptensor_test(complex_bilinear_contraction_test_m4n4k4-validate ${ComplexBilinearContractionTestConfigValidate}  ${ComplexBilinearContractionTestSources})
add_hiptensor_test(complex_bilinear_contraction_test_m4n4k4-bench ${ComplexBilinearContractionTestConfigBench}  ${ComplexBilinearContractionTestSources})
add_hiptensor_test(complex_bilinear_contraction_test_m4n4k4-extended ${ComplexBilinearContractionTestConfigExtended}  ${ComplexBilinearContractionTestSources})

# Complex Bilinear M5N5K5 tests
set (ComplexBilinearContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_bilinear_test_params_rank5.yaml)
set (ComplexBilinearContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/complex_bilinear_test_params_rank5.yaml)
set (ComplexBilinearContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/complex_bilinear_test_params_rank5.yaml)
add_hiptensor_test(complex_bilinear_contraction_test_m5n5k5-validate ${ComplexBilinearContractionTestConfigValidate}  ${ComplexBilinearContractionTestSources})
add_hiptensor_test(complex_bilinear_contraction_test_m5n5k5-bench ${ComplexBilinearContractionTestConfigBench}  ${ComplexBilinearContractionTestSources})
add_hiptensor_test(complex_bilinear_contraction_test_m5n5k5-extended ${ComplexBilinearContractionTestConfigExtended}  ${ComplexBilinearContractionTestSources})

# Complex Bilinear M6N6K6 tests
set (ComplexBilinearContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_bilinear_test_params_rank6.yaml)
set (ComplexBilinearContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/complex_bilinear_test_params_rank6.yaml)
set (ComplexBilinearContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/complex_bilinear_test_params_rank6.yaml)
add_hiptensor_test(complex_bilinear_contraction_test_m6n6k6-validate ${ComplexBilinearContractionTestConfigValidate}  ${ComplexBilinearContractionTestSources})
add_hiptensor_test(complex_bilinear_contraction_test_m6n6k6-bench ${ComplexBilinearContractionTestConfigBench}  ${ComplexBilinearContractionTestSources})
add_hiptensor_test(complex_bilinear_contraction_test_m6n6k6-extended ${ComplexBilinearContractionTestConfigExtended}  ${ComplexBilinearContractionTestSources})

# Scale M2N2K2 tests
set (ScaleContractionTestSources ${ContractionCommonSources}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/scale_contraction_test.cpp)
set (ScaleContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/scale_test_params_rank1.yaml)
set (ScaleContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/scale_test_params_rank1.yaml)
set (ScaleContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/scale_test_params_rank1.yaml)
add_hiptensor_test(scale_contraction_test_m1n1k1-validate ${ScaleContractionTestConfigValidate} ${ScaleContractionTestSources})
add_hiptensor_test(scale_contraction_test_m1n1k1-bench ${ScaleContractionTestConfigBench} ${ScaleContractionTestSources})
add_hiptensor_test(scale_contraction_test_m1n1k1-extended ${ScaleContractionTestConfigExtended} ${ScaleContractionTestSources})

# Scale M2N2K2 tests
set (ScaleContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/scale_test_params_rank2.yaml)
set (ScaleContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/scale_test_params_rank2.yaml)
set (ScaleContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/scale_test_params_rank2.yaml)
add_hiptensor_test(scale_contraction_test_m2n2k2-validate ${ScaleContractionTestConfigValidate} ${ScaleContractionTestSources})
add_hiptensor_test(scale_contraction_test_m2n2k2-bench ${ScaleContractionTestConfigBench} ${ScaleContractionTestSources})
add_hiptensor_test(scale_contraction_test_m2n2k2-extended ${ScaleContractionTestConfigExtended} ${ScaleContractionTestSources})

# Scale M3N3K3 tests
set (ScaleContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/scale_test_params_rank3.yaml)
set (ScaleContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/scale_test_params_rank3.yaml)
set (ScaleContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/scale_test_params_rank3.yaml)
add_hiptensor_test(scale_contraction_test_m3n3k3-validate ${ScaleContractionTestConfigValidate} ${ScaleContractionTestSources})
add_hiptensor_test(scale_contraction_test_m3n3k3-bench ${ScaleContractionTestConfigBench} ${ScaleContractionTestSources})
add_hiptensor_test(scale_contraction_test_m3n3k3-extended ${ScaleContractionTestConfigExtended} ${ScaleContractionTestSources})

# Scale M4N4K4 tests
set (ScaleContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/scale_test_params_rank4.yaml)
set (ScaleContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/scale_test_params_rank4.yaml)
set (ScaleContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/scale_test_params_rank4.yaml)
add_hiptensor_test(scale_contraction_test_m4n4k4-validate ${ScaleContractionTestConfigValidate} ${ScaleContractionTestSources})
add_hiptensor_test(scale_contraction_test_m4n4k4-bench ${ScaleContractionTestConfigBench} ${ScaleContractionTestSources})
add_hiptensor_test(scale_contraction_test_m4n4k4-extended ${ScaleContractionTestConfigExtended} ${ScaleContractionTestSources})

# Scale M5N5K5 tests
set (ScaleContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/scale_test_params_rank5.yaml)
set (ScaleContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/scale_test_params_rank5.yaml)
set (ScaleContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/scale_test_params_rank5.yaml)
add_hiptensor_test(scale_contraction_test_m5n5k5-validate ${ScaleContractionTestConfigValidate} ${ScaleContractionTestSources})
add_hiptensor_test(scale_contraction_test_m5n5k5-bench ${ScaleContractionTestConfigBench} ${ScaleContractionTestSources})
add_hiptensor_test(scale_contraction_test_m5n5k5-extended ${ScaleContractionTestConfigExtended} ${ScaleContractionTestSources})

# Scale M6N6K6 tests
set (ScaleContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/scale_test_params_rank6.yaml)
set (ScaleContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/scale_test_params_rank6.yaml)
set (ScaleContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/scale_test_params_rank6.yaml)
add_hiptensor_test(scale_contraction_test_m6n6k6-validate ${ScaleContractionTestConfigValidate} ${ScaleContractionTestSources})
add_hiptensor_test(scale_contraction_test_m6n6k6-bench ${ScaleContractionTestConfigBench} ${ScaleContractionTestSources})
add_hiptensor_test(scale_contraction_test_m6n6k6-extended ${ScaleContractionTestConfigExtended} ${ScaleContractionTestSources})

# Complex Scale M1N1K1 tests
set (ComplexScaleContractionTestSources ${ContractionCommonSources}
    ${CMAKE_CURRENT_SOURCE_DIR}/complex_scale_contraction_test.cpp)
set (ComplexScaleContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_scale_test_params_rank1.yaml)
set (ComplexScaleContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/complex_scale_test_params_rank1.yaml)
set (ComplexScaleContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/complex_scale_test_params_rank1.yaml)
add_hiptensor_test(complex_scale_contraction_test_m1n1k1-validate ${ComplexScaleContractionTestConfigValidate}  ${ComplexScaleContractionTestSources})
add_hiptensor_test(complex_scale_contraction_test_m1n1k1-bench ${ComplexScaleContractionTestConfigBench}  ${ComplexScaleContractionTestSources})
add_hiptensor_test(complex_scale_contraction_test_m1n1k1-extended ${ComplexScaleContractionTestConfigExtended}  ${ComplexScaleContractionTestSources})

# Complex Scale M2N2K2 tests
set (ComplexScaleContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_scale_test_params_rank2.yaml)
set (ComplexScaleContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/complex_scale_test_params_rank2.yaml)
set (ComplexScaleContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/complex_scale_test_params_rank2.yaml)
add_hiptensor_test(complex_scale_contraction_test_m2n2k2-validate ${ComplexScaleContractionTestConfigValidate}  ${ComplexScaleContractionTestSources})
add_hiptensor_test(complex_scale_contraction_test_m2n2k2-bench ${ComplexScaleContractionTestConfigBench}  ${ComplexScaleContractionTestSources})
add_hiptensor_test(complex_scale_contraction_test_m2n2k2-extended ${ComplexScaleContractionTestConfigExtended}  ${ComplexScaleContractionTestSources})

# Complex Scale M3N3K3 tests
set (ComplexScaleContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_scale_test_params_rank3.yaml)
set (ComplexScaleContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/complex_scale_test_params_rank3.yaml)
set (ComplexScaleContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/complex_scale_test_params_rank3.yaml)
add_hiptensor_test(complex_scale_contraction_test_m3n3k3-validate ${ComplexScaleContractionTestConfigValidate}  ${ComplexScaleContractionTestSources})
add_hiptensor_test(complex_scale_contraction_test_m3n3k3-bench ${ComplexScaleContractionTestConfigBench}  ${ComplexScaleContractionTestSources})
add_hiptensor_test(complex_scale_contraction_test_m3n3k3-extended ${ComplexScaleContractionTestConfigExtended}  ${ComplexScaleContractionTestSources})

# Complex Scale M4N4K4 tests
set (ComplexScaleContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_scale_test_params_rank4.yaml)
set (ComplexScaleContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/complex_scale_test_params_rank4.yaml)
set (ComplexScaleContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/complex_scale_test_params_rank4.yaml)
add_hiptensor_test(complex_scale_contraction_test_m4n4k4-validate ${ComplexScaleContractionTestConfigValidate}  ${ComplexScaleContractionTestSources})
add_hiptensor_test(complex_scale_contraction_test_m4n4k4-bench ${ComplexScaleContractionTestConfigBench}  ${ComplexScaleContractionTestSources})
add_hiptensor_test(complex_scale_contraction_test_m4n4k4-extended ${ComplexScaleContractionTestConfigExtended}  ${ComplexScaleContractionTestSources})

# Complex Scale M5N5K5 tests
set (ComplexScaleContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_scale_test_params_rank5.yaml)
set (ComplexScaleContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/complex_scale_test_params_rank5.yaml)
set (ComplexScaleContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/complex_scale_test_params_rank5.yaml)
add_hiptensor_test(complex_scale_contraction_test_m5n5k5-validate ${ComplexScaleContractionTestConfigValidate}  ${ComplexScaleContractionTestSources})
add_hiptensor_test(complex_scale_contraction_test_m5n5k5-bench ${ComplexScaleContractionTestConfigBench}  ${ComplexScaleContractionTestSources})
add_hiptensor_test(complex_scale_contraction_test_m5n5k5-extended ${ComplexScaleContractionTestConfigExtended}  ${ComplexScaleContractionTestSources})

# Complex Scale M6N6K6 tests
set (ComplexScaleContractionTestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/complex_scale_test_params_rank6.yaml)
set (ComplexScaleContractionTestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/complex_scale_test_params_rank6.yaml)
set (ComplexScaleContractionTestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/complex_scale_test_params_rank6.yaml)
add_hiptensor_test(complex_scale_contraction_test_m6n6k6-validate ${ComplexScaleContractionTestConfigValidate}  ${ComplexScaleContractionTestSources})
add_hiptensor_test(complex_scale_contraction_test_m6n6k6-bench ${ComplexScaleContractionTestConfigBench}  ${ComplexScaleContractionTestSources})
add_hiptensor_test(complex_scale_contraction_test_m6n6k6-extended ${ComplexScaleContractionTestConfigExtended}  ${ComplexScaleContractionTestSources})


# Contraction mode tests
set (ContractionModeTestSources ${ContractionCommonSources}
    ${CMAKE_CURRENT_SOURCE_DIR}/contraction_mode_test.cpp)
set (ContractionModeTestConfig  ${CMAKE_CURRENT_SOURCE_DIR}/configs/mode_test_params.yaml)
add_hiptensor_test(contraction_mode_test ${ContractionModeTestConfig}  ${ContractionModeTestSources})
