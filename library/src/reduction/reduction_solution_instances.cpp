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

        // @todo Add all reference instances. How to stop the explosion of number of instances? Wait for CK
        // genReductionSolutionF16F16Instances();
        // genReductionSolutionBf16Bf16Instances();
        genReductionSolution1x1F32F32Instances();
        genReductionSolution2x1F32F32Instances();
        genReductionSolution2x2F32F32Instances();
        genReductionSolution3x1F32F32Instances();
        genReductionSolution3x2F32F32Instances();
        genReductionSolution3x3F32F32Instances();
        genReductionSolution4x1F32F32Instances();
        genReductionSolution4x2F32F32Instances();
        genReductionSolution4x3F32F32Instances();
        genReductionSolution4x4F32F32Instances();
        genReductionSolution5x1F32F32Instances();
        genReductionSolution5x2F32F32Instances();
        genReductionSolution5x3F32F32Instances();
        genReductionSolution5x4F32F32Instances();
        genReductionSolution5x5F32F32Instances();
        genReductionSolution6x1F32F32Instances();
        genReductionSolution6x2F32F32Instances();
        genReductionSolution6x3F32F32Instances();
        genReductionSolution6x4F32F32Instances();
        genReductionSolution6x5F32F32Instances();
        genReductionSolution6x6F32F32Instances();
        genReductionSolutionF64F64Instances();
    }
} // namespace hiptensor
