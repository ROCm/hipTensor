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

#ifndef HIPTENSOR_PERMUTATION_CPU_REFERENCE_HPP
#define HIPTENSOR_PERMUTATION_CPU_REFERENCE_HPP
#include <hiptensor/hiptensor.hpp>
namespace hiptensor
{
    namespace detail
    {
        template <typename DataType>
        hiptensorStatus_t permuteByCpu(const void*                        alpha,
                                       const DataType*                    A,
                                       const hiptensorTensorDescriptor_t* descA,
                                       const int32_t                      modeA[],
                                       DataType*                          B,
                                       const hiptensorTensorDescriptor_t* descB,
                                       const int32_t                      modeB[],
                                       const hipDataType                  typeScalar);
    }
}

#include "permutation_cpu_reference_impl.hpp"
#endif // HIPTENSOR_PERMUTATION_CPU_REFERENCE_HPP
