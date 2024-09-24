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
set(ReductionCommonSources ${HIPTENSOR_COMMON_TEST_SOURCES}
                             ${CMAKE_CURRENT_SOURCE_DIR}/reduction_resource.cpp
                             ${CMAKE_CURRENT_SOURCE_DIR}/reduction_test.cpp)

#  tests
set (ReductionTestSources ${ReductionCommonSources}
    ${CMAKE_CURRENT_SOURCE_DIR}/reduction_cpu_impl_test.cpp
    )
set (ReductionConfig ${CMAKE_CURRENT_SOURCE_DIR}/configs/reduction_params.yaml)
add_hiptensor_test(reduction_cpu_impl_test ${ReductionConfig} ${ReductionTestSources})

set (ReductionRank1TestSources ${ReductionCommonSources}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/rank1_reduction_test.cpp)
set (ReductionRank1TestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank1_test_params.yaml)
set (ReductionRank1TestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/rank1_test_params.yaml)
set (ReductionRank1TestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/rank1_test_params.yaml)
add_hiptensor_test(rank1_reduction_test-validate ${ReductionRank1TestConfigValidate}  ${ReductionRank1TestSources})
add_hiptensor_test(rank1_reduction_test-bench ${ReductionRank1TestConfigBench}  ${ReductionRank1TestSources})
add_hiptensor_test(rank1_reduction_test-extended ${ReductionRank1TestConfigExtended}  ${ReductionRank1TestSources})

set (ReductionRank2TestSources ${ReductionCommonSources}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/rank2_reduction_test.cpp)
set (ReductionRank2TestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank2_test_params.yaml)
set (ReductionRank2TestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/rank2_test_params.yaml)
set (ReductionRank2TestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/rank2_test_params.yaml)
add_hiptensor_test(rank2_reduction_test-validate ${ReductionRank2TestConfigValidate}  ${ReductionRank2TestSources})
add_hiptensor_test(rank2_reduction_test-bench ${ReductionRank2TestConfigBench}  ${ReductionRank2TestSources})
add_hiptensor_test(rank2_reduction_test-extended ${ReductionRank2TestConfigExtended}  ${ReductionRank2TestSources})

set (ReductionRank3TestSources ${ReductionCommonSources}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/rank3_reduction_test.cpp)
set (ReductionRank3TestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank3_test_params.yaml)
set (ReductionRank3TestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/rank3_test_params.yaml)
set (ReductionRank3TestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/rank3_test_params.yaml)
add_hiptensor_test(rank3_reduction_test-validate ${ReductionRank3TestConfigValidate}  ${ReductionRank3TestSources})
add_hiptensor_test(rank3_reduction_test-bench ${ReductionRank3TestConfigBench}  ${ReductionRank3TestSources})
add_hiptensor_test(rank3_reduction_test-extended ${ReductionRank3TestConfigExtended}  ${ReductionRank3TestSources})

set (ReductionRank4TestSources ${ReductionCommonSources}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/rank4_reduction_test.cpp)
set (ReductionRank4TestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank4_test_params.yaml)
set (ReductionRank4TestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/rank4_test_params.yaml)
set (ReductionRank4TestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/rank4_test_params.yaml)
add_hiptensor_test(rank4_reduction_test-validate ${ReductionRank4TestConfigValidate}  ${ReductionRank4TestSources})
add_hiptensor_test(rank4_reduction_test-bench ${ReductionRank4TestConfigBench}  ${ReductionRank4TestSources})
add_hiptensor_test(rank4_reduction_test-extended ${ReductionRank4TestConfigExtended}  ${ReductionRank4TestSources})

set (ReductionRank5TestSources ${ReductionCommonSources}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/rank5_reduction_test.cpp)
set (ReductionRank5TestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank5_test_params.yaml)
set (ReductionRank5TestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/rank5_test_params.yaml)
set (ReductionRank5TestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/rank5_test_params.yaml)
add_hiptensor_test(rank5_reduction_test-validate ${ReductionRank5TestConfigValidate}  ${ReductionRank5TestSources})
add_hiptensor_test(rank5_reduction_test-bench ${ReductionRank5TestConfigBench}  ${ReductionRank5TestSources})
add_hiptensor_test(rank5_reduction_test-extended ${ReductionRank5TestConfigExtended}  ${ReductionRank5TestSources})

set (ReductionRank6TestSources ${ReductionCommonSources}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/rank6_reduction_test.cpp)
set (ReductionRank6TestConfigValidate  ${CMAKE_CURRENT_SOURCE_DIR}/configs/validation/rank6_test_params.yaml)
set (ReductionRank6TestConfigBench  ${CMAKE_CURRENT_SOURCE_DIR}/configs/bench/rank6_test_params.yaml)
set (ReductionRank6TestConfigExtended  ${CMAKE_CURRENT_SOURCE_DIR}/configs/extended/rank6_test_params.yaml)
add_hiptensor_test(rank6_reduction_test-validate ${ReductionRank6TestConfigValidate}  ${ReductionRank6TestSources})
add_hiptensor_test(rank6_reduction_test-bench ${ReductionRank6TestConfigBench}  ${ReductionRank6TestSources})
add_hiptensor_test(rank6_reduction_test-extended ${ReductionRank6TestConfigExtended}  ${ReductionRank6TestSources})

