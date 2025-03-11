/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

// clang-format off

#ifndef HIPTENSOR_TEST_rank2_binary_op_test_params_YAML_HPP
#define HIPTENSOR_TEST_rank2_binary_op_test_params_YAML_HPP

#define HIPTENSOR_TEST_GET_YAML rank2_binary_op_test_params_get_yaml()

#include <string>

static inline std::string rank2_binary_op_test_params_get_yaml()
{
    return std::string(R"(---
Log Level:       [ HIPTENSOR_LOG_LEVEL_ERROR, HIPTENSOR_LOG_LEVEL_PERF_TRACE ]
Tensor Data Types:
  - [ HIP_R_32F, HIP_R_32F]
  - [ HIP_R_16F, HIP_R_16F]
Alphas:
  - 2.3
Gammas:
  - [1.3]
Lengths:
  - [ 1, 1]
  - [ 5, 2]
  - [ 3, 4]
  - [ 15, 12]
  - [ 23, 11]
Operators:
  - [HIPTENSOR_OP_IDENTITY, HIPTENSOR_OP_IDENTITY, HIPTENSOR_OP_ADD]
  - [HIPTENSOR_OP_NEG, HIPTENSOR_OP_NEG, HIPTENSOR_OP_ADD]
Permuted Dims:
  - [0, 1]
  - [1, 0]
...
)");
}

#endif // HIPTENSOR_TEST_rank2_binary_op_test_params_YAML_HPP_

// clang-format on
