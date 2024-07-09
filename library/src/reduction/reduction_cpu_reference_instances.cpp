/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "reduction_cpu_reference_instances.hpp"
#include "reduction_cpu_reference_impl.hpp"
#include <hiptensor/internal/types.hpp>

#define REG_CPU_SOLUTION(dim_count, reduced_dim_count, type)         \
    registerSolutions(enumerateReferenceSolutions<type,              \
                                                  type,              \
                                                  type,              \
                                                  dim_count,         \
                                                  reduced_dim_count, \
                                                  HIPTENSOR_OP_ADD,  \
                                                  true,              \
                                                  false>());         \
    registerSolutions(enumerateReferenceSolutions<type,              \
                                                  type,              \
                                                  type,              \
                                                  dim_count,         \
                                                  reduced_dim_count, \
                                                  HIPTENSOR_OP_MUL,  \
                                                  true,              \
                                                  false>());         \
    registerSolutions(enumerateReferenceSolutions<type,              \
                                                  type,              \
                                                  type,              \
                                                  dim_count,         \
                                                  reduced_dim_count, \
                                                  HIPTENSOR_OP_MIN,  \
                                                  true,              \
                                                  false>());         \
    registerSolutions(enumerateReferenceSolutions<type,              \
                                                  type,              \
                                                  type,              \
                                                  dim_count,         \
                                                  reduced_dim_count, \
                                                  HIPTENSOR_OP_MAX,  \
                                                  true,              \
                                                  false>());

namespace hiptensor
{
    ReductionCpuReferenceInstances::ReductionCpuReferenceInstances()
    {
        // @todo Add all reference instances. How to stop the explosion of number of instances? Wait for CK
        // registerSolutions(enumerateReferenceSolutions<hiptensor::bfloat16_t,
        // hiptensor::bfloat16_t,
        // hiptensor::bfloat16_t,
        // 4,
        // 2,
        // HIPTENSOR_OP_ADD,
        // true, // PropagateNan,
        // false>()); // OutputIndex,
        // registerSolutions(enumerateReferenceSolutions<hiptensor::float16_t,
        // hiptensor::float16_t,
        // hiptensor::float16_t,
        // 4,
        // 2,
        // HIPTENSOR_OP_ADD,
        // true, // PropagateNan,
        // false>()); // OutputIndex,

        REG_CPU_SOLUTION(1, 1, hiptensor::float32_t);
        REG_CPU_SOLUTION(2, 1, hiptensor::float32_t);
        REG_CPU_SOLUTION(2, 2, hiptensor::float32_t);
        REG_CPU_SOLUTION(3, 1, hiptensor::float32_t);
        REG_CPU_SOLUTION(3, 2, hiptensor::float32_t);
        REG_CPU_SOLUTION(3, 3, hiptensor::float32_t);
        REG_CPU_SOLUTION(4, 1, hiptensor::float32_t);
        REG_CPU_SOLUTION(4, 2, hiptensor::float32_t);
        REG_CPU_SOLUTION(4, 3, hiptensor::float32_t);
        REG_CPU_SOLUTION(4, 4, hiptensor::float32_t);
        REG_CPU_SOLUTION(5, 1, hiptensor::float32_t);
        REG_CPU_SOLUTION(5, 2, hiptensor::float32_t);
        REG_CPU_SOLUTION(5, 3, hiptensor::float32_t);
        REG_CPU_SOLUTION(5, 4, hiptensor::float32_t);
        REG_CPU_SOLUTION(5, 5, hiptensor::float32_t);
        REG_CPU_SOLUTION(6, 1, hiptensor::float32_t);
        REG_CPU_SOLUTION(6, 2, hiptensor::float32_t);
        REG_CPU_SOLUTION(6, 3, hiptensor::float32_t);
        REG_CPU_SOLUTION(6, 4, hiptensor::float32_t);
        REG_CPU_SOLUTION(6, 5, hiptensor::float32_t);
        REG_CPU_SOLUTION(6, 6, hiptensor::float32_t);

        REG_CPU_SOLUTION(1, 1, hiptensor::float64_t);
        REG_CPU_SOLUTION(2, 1, hiptensor::float64_t);
        REG_CPU_SOLUTION(2, 2, hiptensor::float64_t);
        REG_CPU_SOLUTION(3, 1, hiptensor::float64_t);
        REG_CPU_SOLUTION(3, 2, hiptensor::float64_t);
        REG_CPU_SOLUTION(3, 3, hiptensor::float64_t);
        REG_CPU_SOLUTION(4, 1, hiptensor::float64_t);
        REG_CPU_SOLUTION(4, 2, hiptensor::float64_t);
        REG_CPU_SOLUTION(4, 3, hiptensor::float64_t);
        REG_CPU_SOLUTION(4, 4, hiptensor::float64_t);
        REG_CPU_SOLUTION(5, 1, hiptensor::float64_t);
        REG_CPU_SOLUTION(5, 2, hiptensor::float64_t);
        REG_CPU_SOLUTION(5, 3, hiptensor::float64_t);
        REG_CPU_SOLUTION(5, 4, hiptensor::float64_t);
        REG_CPU_SOLUTION(5, 5, hiptensor::float64_t);
        REG_CPU_SOLUTION(6, 1, hiptensor::float64_t);
        REG_CPU_SOLUTION(6, 2, hiptensor::float64_t);
        REG_CPU_SOLUTION(6, 3, hiptensor::float64_t);
        REG_CPU_SOLUTION(6, 4, hiptensor::float64_t);
        REG_CPU_SOLUTION(6, 5, hiptensor::float64_t);
        REG_CPU_SOLUTION(6, 6, hiptensor::float64_t);
    }
} // namespace hiptensor
