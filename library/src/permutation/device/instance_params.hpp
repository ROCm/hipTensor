/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef INSTANCE_PARAMS_HPP
#define INSTANCE_PARAMS_HPP

#include "../permutation_types.hpp"
#include "data_types.hpp"
#include "hash.hpp"

namespace ck::tensor_operation::device::instance
{
    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename Aop,
              typename Bop,
              typename Scale,
              index_t NumDim,
              index_t BlockSize                  = 0,
              index_t M0PerBlock                 = 0,
              index_t M1PerBlock                 = 0,
              index_t M0PerThread                = 0,
              index_t M1PerThread                = 0,
              typename ThreadClusterArrangeOrder = ck::Sequence<0, 0>,
              typename InScalarPerVectorSeq      = ck::Sequence<0>,
              typename OutScalarPerVectorSeq     = ck::Sequence<0>>
    struct DeviceElementwiseParams
    {
        static hiptensor::Uid hashCode()
        {
            return hiptensor::Hash{}(
                hiptensor::HipDataType_v<typename ck::tuple_element_t<0, InDataTypeTuple>>,
                hiptensor::HipDataType_v<typename ck::tuple_element_t<0, OutDataTypeTuple>>,
                hiptensor::ElementWiseOperatorType_v<Aop>,
                hiptensor::ElementWiseOperatorType_v<Bop>,
                hiptensor::PermutationOperatorType_v<Scale>,
                NumDim,
                BlockSize,
                M0PerBlock,
                M1PerBlock,
                M0PerThread,
                M1PerThread,
                ThreadClusterArrangeOrder::At(0),
                ThreadClusterArrangeOrder::At(1),
                InScalarPerVectorSeq::At(0),
                OutScalarPerVectorSeq::At(0));
        }
    };

    // Ck requires that the length of fastest changing dimonsion must be multiple times of InScalarPerVectorSeq
    // and OutScalarPerVectorSeq. So `getHashCodesWithAllInOutScalarPerVectorSeq` always return a vector of hash code
    // which represent In/OutScalarPerVectorSeq from 16 to 0.
    // The caller should test the returned hash code in order since instance with In/OutScalarPerVectorSeq of 16
    //  has the best performance on average and instance with In/OutScalarPerVectorSeq of 1 can handle inputs of
    // all shapes which is the last resort.
    std::vector<hiptensor::Uid> getHashCodesWithAllInOutScalarPerVectorSeq(
        hipDataType                  typeIn,
        hipDataType                  typeOut,
        hiptensorOperator_t          aOp,
        hiptensorOperator_t          bOp,
        hiptensor::PermutationOpId_t scale,
        index_t                      numDim,
        index_t                      blockSize,
        index_t                      m0PerBlock,
        index_t                      m1PerBlock,
        index_t                      m0PerThread,
        index_t                      m1PerThread,
        std::pair<index_t, index_t>  threadClusterArrangeOrder);

}
#endif //  INSTANCE_PARAMS_HPP
