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
#ifndef HIPTENSOR_PERMUTATION_CPU_REFERENCE_IMPL_HPP
#define HIPTENSOR_PERMUTATION_CPU_REFERENCE_IMPL_HPP
#include <hiptensor/hiptensor.hpp>
#include <vector>
#include <unordered_map>

#include "permutation_cpu_reference.hpp"
#include "types.hpp"
#include "util.hpp"

namespace hiptensor
{
    namespace details
    {
        template <typename DataType>
        hiptensorStatus_t permuteByCpu(const void*                        alpha,
                                       const DataType*                    A,
                                       const hiptensorTensorDescriptor_t* descA,
                                       const int32_t                      modeA[],
                                       DataType*                          B,
                                       const hiptensorTensorDescriptor_t* descB,
                                       const int32_t                      modeB[],
                                       const hipDataType                  typeScalar)
        {
            const auto modeSize = descA->mLengths.size();
            assert(modeSize <= 4);

            std::unordered_map<int32_t, int32_t> bModeToIndex;
            for(int32_t index = 0; index < modeSize; index++)
            {
                bModeToIndex[modeB[index]] = index;
            }

            auto& aLens = descA->mLengths;
            // auto bStrides = descB->mStrides; // TODO descB->mStrides contains incorrect strides
            auto bStrides = std::vector<int32_t>(modeSize, 1);
            for(int i = 1; i < modeSize; i++)
            {
                bStrides[i] = descB->mLengths[i - 1] * bStrides[i - 1];
            }
            auto bIndices     = std::vector<int32_t>(modeSize, 0);
            auto elementCount = hiptensor::elementsFromLengths(aLens);
            for(int elementIndex = 0; elementIndex < elementCount; elementIndex++)
            {
                auto index = elementIndex;
                for(int modeIndex = 0; modeIndex < modeSize; modeIndex++)
                {
                    bIndices[bModeToIndex[modeA[modeIndex]]] = index % aLens[modeIndex];
                    index /= aLens[modeIndex];
                }
                auto bOffset
                    = std::inner_product(bIndices.begin(), bIndices.end(), bStrides.begin(), 0);
                B[bOffset] = A[elementIndex];
            }

            return HIPTENSOR_STATUS_SUCCESS;
        }
    }
}
#endif //HIPTENSOR_PERMUTATION_CPU_REFERENCE_IMPL_HPP
