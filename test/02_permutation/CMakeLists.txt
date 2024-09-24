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
set(PermutationCommonSources ${HIPTENSOR_COMMON_TEST_SOURCES}
                             ${CMAKE_CURRENT_SOURCE_DIR}/permutation_resource.cpp
                             ${CMAKE_CURRENT_SOURCE_DIR}/permutation_test.cpp)

#  tests
set (PermutationTestSources ${PermutationCommonSources}
    ${CMAKE_CURRENT_SOURCE_DIR}/permutation_column_major_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/permutation_cpu_impl_test.cpp
    )


# Rank 2 tests
set (PermutationRank2TestSources ${PermutationCommonSources}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/rank2_permutation_test.cpp)
set (PermutationRank2TestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank2_test_params.yaml)
set (PermutationRank2TestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/rank2_test_params.yaml)
set (PermutationRank2TestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/rank2_test_params.yaml)
add_hiptensor_test(rank2_permutation_test-validate ${PermutationRank2TestConfigValidate}  ${PermutationRank2TestSources})
add_hiptensor_test(rank2_permutation_test-bench ${PermutationRank2TestConfigBench}  ${PermutationRank2TestSources})
add_hiptensor_test(rank2_permutation_test-extended ${PermutationRank2TestConfigExtended}  ${PermutationRank2TestSources})

# Rank 3 tests
set (PermutationRank3TestSources ${PermutationCommonSources}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/rank3_permutation_test.cpp)
set (PermutationRank3TestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank3_test_params.yaml)
set (PermutationRank3TestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/rank3_test_params.yaml)
set (PermutationRank3TestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/rank3_test_params.yaml)
add_hiptensor_test(rank3_permutation_test-validate ${PermutationRank3TestConfigValidate}  ${PermutationRank3TestSources})
add_hiptensor_test(rank3_permutation_test-bench ${PermutationRank3TestConfigBench}  ${PermutationRank3TestSources})
add_hiptensor_test(rank3_permutation_test-extended ${PermutationRank3TestConfigExtended}  ${PermutationRank3TestSources})

# Rank 4 tests
set (PermutationRank4TestSources ${PermutationCommonSources}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/rank4_permutation_test.cpp)
set (PermutationRank4TestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank4_test_params.yaml)
set (PermutationRank4TestConfigBench ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/rank4_test_params.yaml)
set (PermutationRank4TestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/rank4_test_params.yaml)
add_hiptensor_test(rank4_permutation_test-validate ${PermutationRank4TestConfigValidate}  ${PermutationRank4TestSources})
add_hiptensor_test(rank4_permutation_test-bench ${PermutationRank4TestConfigBench}  ${PermutationRank4TestSources})
add_hiptensor_test(rank4_permutation_test-extended ${PermutationRank4TestConfigExtended}  ${PermutationRank4TestSources})

# Rank 5 tests
set (PermutationRank5TestSources ${PermutationCommonSources}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/rank5_permutation_test.cpp)
set (PermutationRank5TestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank5_test_params.yaml)
set (PermutationRank5TestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/rank5_test_params.yaml)
set (PermutationRank5TestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/rank5_test_params.yaml)
add_hiptensor_test(rank5_permutation_test-validate ${PermutationRank5TestConfigValidate}  ${PermutationRank5TestSources})
add_hiptensor_test(rank5_permutation_test-bench ${PermutationRank5TestConfigBench}  ${PermutationRank5TestSources})
add_hiptensor_test(rank5_permutation_test-extended ${PermutationRank5TestConfigExtended}  ${PermutationRank5TestSources})

# Rank 6 tests
set (PermutationRank6TestSources ${PermutationCommonSources}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/rank6_permutation_test.cpp)
set (PermutationRank6TestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank6_test_params.yaml)
set (PermutationRank6TestConfigBench ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/rank6_test_params.yaml)
set (PermutationRank6TestConfigExtended ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/rank6_test_params.yaml)
add_hiptensor_test(rank6_permutation_test-validate ${PermutationRank6TestConfigValidate}  ${PermutationRank6TestSources})
add_hiptensor_test(rank6_permutation_test-bench ${PermutationRank6TestConfigBench}  ${PermutationRank6TestSources})
add_hiptensor_test(rank6_permutation_test-extended ${PermutationRank6TestConfigExtended}  ${PermutationRank6TestSources})
