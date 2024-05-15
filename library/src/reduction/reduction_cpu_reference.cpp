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

#include "reduction_cpu_reference.hpp"
#include "reduction_cpu_reference_impl.hpp"
#include "reduction_cpu_reference_instances.hpp"

hiptensorStatus_t hiptensorReductionReference(const void*                        alpha,
                                              const void*                        A,
                                              const hiptensorTensorDescriptor_t* descA,
                                              const int32_t                      modeA[],
                                              const void*                        beta,
                                              const void*                        C,
                                              const hiptensorTensorDescriptor_t* descC,
                                              const int32_t                      modeC[],
                                              void*                              D,
                                              const hiptensorTensorDescriptor_t* descD,
                                              const int32_t                      modeD[],
                                              hiptensorOperator_t                opReduce,
                                              hiptensorComputeType_t             typeCompute,
                                              hipStream_t                        stream)
{
    int  rankA        = descA->mLengths.size();
    int  numReduceDim = descA->mLengths.size() - descD->mLengths.size();
    auto ADataType    = descA->mType;
    auto DDataType    = descD->mType;

    auto& instances = hiptensor::ReductionCpuReferenceInstances::instance();
    auto  solutionQ = instances->querySolutions(ADataType,
                                               typeCompute,
                                               DDataType,
                                               rankA,
                                               numReduceDim,
                                               opReduce,
                                               true, // @TODO hardcode
                                               false); // @TODO hardcode

    double alphaD;
    if(alpha != nullptr)
    {
        alphaD = hiptensor::readVal<double>(alpha, typeCompute);
    }
    double betaD;
    if(beta != nullptr)
    {
        betaD = hiptensor::readVal<double>(beta, typeCompute);
    }

    for(auto [_, pSolution] : solutionQ.solutions())
    {
        // Perform reduction with timing if LOG_LEVEL_PERF_TRACE
        auto streamConfig        = StreamConfig{stream, false};
        auto [isSupported, time] = (*pSolution)(descA->mLengths,
                                                // @todo pass stride from descA
                                                {},
                                                {modeA, modeA + descA->mLengths.size()},
                                                descC->mLengths,
                                                {},
                                                {modeC, modeC + descC->mLengths.size()},
                                                alphaD,
                                                betaD,
                                                A,
                                                D,
                                                opReduce,
                                                streamConfig);
        if(isSupported)
        {
            if(time < 0)
            {
                return HIPTENSOR_STATUS_CK_ERROR;
            }
            return HIPTENSOR_STATUS_SUCCESS;
        }
    }

    return HIPTENSOR_STATUS_INTERNAL_ERROR;
}
