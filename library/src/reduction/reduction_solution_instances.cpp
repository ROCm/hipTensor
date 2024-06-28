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

#include "reduction_solution_instances.hpp"
#include "hiptensor/internal/types.hpp"
#include "reduction_solution.hpp"

// Ensure access to
// #include "device/hiptensor_reduction_scale_instances.hpp"

namespace hiptensor
{
    ReductionSolutionInstances::ReductionSolutionInstances()
    {

        // add entries to mSolutionQuery
#if 0
        registerSolutions(enumerateReductionSolutions<hiptensor::float16_t,
                                                      hiptensor::float16_t,
                                                      hiptensor::float16_t,
                                                      4,
                                                      2,
                                                      HIPTENSOR_OP_ADD,
                                                      true, // PropagateNan,
                                                      false>()); // OutputIndex,
        registerSolutions(enumerateReductionSolutions<hiptensor::float16_t,
                                                      hiptensor::float16_t,
                                                      hiptensor::float16_t,
                                                      4,
                                                      2,
                                                      HIPTENSOR_OP_MUL,
                                                      true, // PropagateNan,
                                                      false>()); // OutputIndex,
        registerSolutions(enumerateReductionSolutions<hiptensor::float16_t,
                                                      hiptensor::float16_t,
                                                      hiptensor::float16_t,
                                                      4,
                                                      2,
                                                      HIPTENSOR_OP_MIN,
                                                      true, // PropagateNan,
                                                      false>()); // OutputIndex,
        registerSolutions(enumerateReductionSolutions<hiptensor::float16_t,
                                                      hiptensor::float16_t,
                                                      hiptensor::float16_t,
                                                      4,
                                                      2,
                                                      HIPTENSOR_OP_MAX,
                                                      true, // PropagateNan,
                                                      false>()); // OutputIndex,
        registerSolutions(enumerateReductionSolutions<hiptensor::bfloat16_t,
                                                      hiptensor::bfloat16_t,
                                                      hiptensor::bfloat16_t,
                                                      4,
                                                      2,
                                                      HIPTENSOR_OP_ADD,
                                                      true, // PropagateNan,
                                                      false>()); // OutputIndex,
        registerSolutions(enumerateReductionSolutions<hiptensor::bfloat16_t,
                                                      hiptensor::bfloat16_t,
                                                      hiptensor::bfloat16_t,
                                                      4,
                                                      2,
                                                      HIPTENSOR_OP_MUL,
                                                      true, // PropagateNan,
                                                      false>()); // OutputIndex,
        registerSolutions(enumerateReductionSolutions<hiptensor::bfloat16_t,
                                                      hiptensor::bfloat16_t,
                                                      hiptensor::bfloat16_t,
                                                      4,
                                                      2,
                                                      HIPTENSOR_OP_MIN,
                                                      true, // PropagateNan,
                                                      false>()); // OutputIndex,
        registerSolutions(enumerateReductionSolutions<hiptensor::bfloat16_t,
                                                      hiptensor::bfloat16_t,
                                                      hiptensor::bfloat16_t,
                                                      4,
                                                      2,
                                                      HIPTENSOR_OP_MAX,
                                                      true, // PropagateNan,
                                                      false>()); // OutputIndex,
#endif
        registerSolutions(enumerateReductionSolutions<hiptensor::float32_t,
                                                      hiptensor::float32_t,
                                                      hiptensor::float32_t,
                                                      4,
                                                      2,
                                                      HIPTENSOR_OP_ADD,
                                                      true, // PropagateNan,
                                                      false>()); // OutputIndex,
        registerSolutions(enumerateReductionSolutions<hiptensor::float32_t,
                                                      hiptensor::float32_t,
                                                      hiptensor::float32_t,
                                                      4,
                                                      2,
                                                      HIPTENSOR_OP_MUL,
                                                      true, // PropagateNan,
                                                      false>()); // OutputIndex,
        registerSolutions(enumerateReductionSolutions<hiptensor::float32_t,
                                                      hiptensor::float32_t,
                                                      hiptensor::float32_t,
                                                      4,
                                                      2,
                                                      HIPTENSOR_OP_MIN,
                                                      true, // PropagateNan,
                                                      false>()); // OutputIndex,
        registerSolutions(enumerateReductionSolutions<hiptensor::float32_t,
                                                      hiptensor::float32_t,
                                                      hiptensor::float32_t,
                                                      4,
                                                      2,
                                                      HIPTENSOR_OP_MAX,
                                                      true, // PropagateNan,
                                                      false>()); // OutputIndex,
        registerSolutions(enumerateReductionSolutions<hiptensor::float64_t,
                                                      hiptensor::float64_t,
                                                      hiptensor::float64_t,
                                                      4,
                                                      2,
                                                      HIPTENSOR_OP_ADD,
                                                      true, // PropagateNan,
                                                      false>()); // OutputIndex,
        registerSolutions(enumerateReductionSolutions<hiptensor::float64_t,
                                                      hiptensor::float64_t,
                                                      hiptensor::float64_t,
                                                      4,
                                                      2,
                                                      HIPTENSOR_OP_MUL,
                                                      true, // PropagateNan,
                                                      false>()); // OutputIndex,
        registerSolutions(enumerateReductionSolutions<hiptensor::float64_t,
                                                      hiptensor::float64_t,
                                                      hiptensor::float64_t,
                                                      4,
                                                      2,
                                                      HIPTENSOR_OP_MIN,
                                                      true, // PropagateNan,
                                                      false>()); // OutputIndex,
        registerSolutions(enumerateReductionSolutions<hiptensor::float64_t,
                                                      hiptensor::float64_t,
                                                      hiptensor::float64_t,
                                                      4,
                                                      2,
                                                      HIPTENSOR_OP_MAX,
                                                      true, // PropagateNan,
                                                      false>()); // OutputIndex,
    }
} // namespace hiptensor
