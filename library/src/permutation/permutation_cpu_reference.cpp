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

#include "permutation_cpu_reference.hpp"
#include "permutation_cpu_reference_impl.hpp"
#include "permutation_cpu_reference_instances.hpp"

hiptensorStatus_t hiptensorPermutationReference(const hiptensorHandle_t*           handle,
                                                const void*                        alpha,
                                                const void*                        A,
                                                const hiptensorTensorDescriptor_t* descA,
                                                const int32_t                      modeA[],
                                                void*                              B,
                                                const hiptensorTensorDescriptor_t* descB,
                                                const int32_t                      modeB[],
                                                const hipDataType                  typeScalar,
                                                const hipStream_t                  stream)
{
    const int32_t dim    = descA->mLengths.size();
    std::cout << " dim " << dim << " descA->mType " << descA->mType << " descB->mType " << descB->mType << " descA->mUnaryOp " << descA->mUnaryOp << " descB->mUnaryOp " << descB->mUnaryOp << std::endl;
    auto& instances   = hiptensor::PermutationCpuReferenceInstances::instance();
    std::cout << " sol count " << instances->allSolutions().solutionCount() << std::endl;
    auto  candidates  = instances->allSolutions().query(dim,
                                                        descA->mType,
                                                        descB->mType,
                                                        descA->mUnaryOp,
                                                        descB->mUnaryOp,
                                                        hiptensor::PermutationOpId_t::SCALE);

    auto toCKVec
        = [](auto& inputVec) { return std::vector<ck::index_t>(inputVec.begin(), inputVec.end()); };

    std::cout << candidates.solutionCount() << std::endl;
    if(candidates.solutionCount() != 1)
    {
        return HIPTENSOR_STATUS_INTERNAL_ERROR;
    }
    else
    {
        auto refCandidate = candidates.solutions().begin()->second;
        std::cout << " init args " << std::endl;
        if(refCandidate->initArgs(alpha,
                                  A,
                                  B,
                                  descA->mLengths,
                                  descA->mStrides,
                                  modeA,
                                  descB->mLengths,
                                  descB->mStrides,
                                  modeB,
                                  typeScalar))
        {
            std::cout << " Run candidate " << std::endl;
            (*refCandidate)();
        }
        return HIPTENSOR_STATUS_SUCCESS;
    }
}
