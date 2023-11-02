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
#ifndef HIPTENSOR_PERMUTATION_CK_COL_IMPL_HPP
#define HIPTENSOR_PERMUTATION_CK_COL_IMPL_HPP
#include <cstdlib>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_elementwise_scale_impl.hpp>
#include <ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp>

#include "types.hpp"

namespace hiptensor
{
    namespace detail
    {
        template <typename DataType>
        hiptensorStatus_t permuteByCk(const void*                        alpha,
                                      const DataType*                    A,
                                      const hiptensorTensorDescriptor_t* descA,
                                      const int32_t                      modeA[],
                                      DataType*                          B,
                                      const hiptensorTensorDescriptor_t* descB,
                                      const int32_t                      modeB[],
                                      const hipDataType                  typeScalar,
                                      const hipStream_t                  stream)
        {
            using PassThrough = ck::tensor_operation::element_wise::PassThrough;
            using UnaryOp     = ck::tensor_operation::element_wise::PassThrough;
            using Scale       = ck::tensor_operation::element_wise::Scale;
            using DeviceElementwisePermuteInstance
                = ck::tensor_operation::device::DeviceElementwiseImpl<
                    ck::Tuple<DataType>, // InDataTypeTuple
                    ck::Tuple<DataType>, // OutDataTypeTuple
                    PassThrough, // ElementwiseOp
                    UnaryOp, // UnaryOp
                    Scale, // Scalar
                    4, // NumDim
                    1, // MPerThread
                    ck::Sequence<1>, // InScalarPerVectorSeq
                    ck::Sequence<1>>; // OutScalarPerVectorSeq

            const auto modeSize = descA->mLengths.size();
            assert(modeSize == 4);

            std::unordered_map<int32_t, int32_t>
                modeToLength; // for example {'n': 1, 'c': 2, 'w': 3, 'h':0}

            for(int32_t index = 0; index < modeSize; index++)
            {
                modeToLength[modeA[index]] = descA->mLengths[index];
            }

            std::unordered_map<int32_t, int32_t> bModeToStrides;
            int32_t                              stride = 1;
            bModeToStrides[modeB[0]]                    = stride;
            for(int32_t index = 1; index < modeSize; index++)
            {
                stride *= modeToLength[modeB[index - 1]];
                bModeToStrides[modeB[index]] = stride;
            }

            float                      alphaValue = readVal<float>(alpha, typeScalar);
            std::array<const void*, 1> input      = {A};
            std::array<void*, 1>       output     = {B};
            std::array<ck::index_t, 4> a_strides
                = {1,
                   modeToLength[modeA[0]],
                   modeToLength[modeA[0]] * modeToLength[modeA[1]],
                   modeToLength[modeA[0]] * modeToLength[modeA[1]] * modeToLength[modeA[2]]};
            std::array<ck::index_t, 4> b_strides        = {bModeToStrides[modeA[0]],
                                                           bModeToStrides[modeA[1]],
                                                           bModeToStrides[modeA[2]],
                                                           bModeToStrides[modeA[3]]};
            std::array<ck::index_t, 4> ab_lengths       = {modeToLength[modeA[0]],
                                                           modeToLength[modeA[1]],
                                                           modeToLength[modeA[2]],
                                                           modeToLength[modeA[3]]};
            auto                       broadcastPermute = DeviceElementwisePermuteInstance{};
            auto                       argument = broadcastPermute.MakeArgumentPointer(ab_lengths,
                                                                                       {a_strides},
                                                                                       {b_strides},
                                                                 input,
                                                                 output,
                                                                 PassThrough{},
                                                                 UnaryOp{},
                                                                 Scale{alphaValue});

            if(!broadcastPermute.IsSupportedArgument(argument.get()))
            {
                return HIPTENSOR_STATUS_NOT_SUPPORTED;
            };

            auto broadcastPermute_invoker_ptr = broadcastPermute.MakeInvokerPointer();
            broadcastPermute_invoker_ptr->Run(argument.get(), StreamConfig{stream, false});
            return HIPTENSOR_STATUS_SUCCESS;
        }
    }
}
#endif // HIPTENSOR_PERMUTATION_CK_COL_IMPL_HPP
