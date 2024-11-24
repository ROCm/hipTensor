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

#include "data_types.hpp"
#include "hash.hpp"

namespace ck::tensor_operation::device::instance
{
    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename ElementwiseOperation,
              index_t NumDim,
              index_t BlockSize,
              index_t M0PerBlock,
              index_t M1PerBlock,
              index_t M0PerThread,
              index_t M1PerThread,
              typename ThreadClusterArrangeOrder,
              typename InScalarPerVectorSeq,
              typename OutScalarPerVectorSeq>
    struct DeviceElementwiseParams
    {
        // using InDataTypeTuple = InDataTypeTuple;
        // using OutDataTypeTuple = OutDataTypeTuple;
        // using ElementwiseOperation = ElementwiseOperation;
        // using ThreadClusterArrangeOrder = ThreadClusterArrangeOrder;
        // using InScalarPerVectorSeq = InScalarPerVectorSeq;
        // using OutScalarPerVectorSeq = OutScalarPerVectorSeq;
        // ck::index_t getNumDim() const {return NumDim;}
        // ck::index_t getBlockSize() const{return BlockSize;}
        // ck::index_t getM0PerBlock() const{return M0PerBlock;}
        // ck::index_t getM1PerBlock() const{return M1PerBlock;}
        // ck::index_t getM0PerThread() const{return M0PerThread;}
        // ck::index_t getM1PerThread() const{return M1PerThread;}

        static std::vector<std::size_t> queryIdsWithAllInOutScalarPerVectorSeq()
        {
            std::vector<std::size_t> ids;
            return ids;
        }
        static std::size_t id()
        {
            return hiptensor::Hash{}(
                hiptensor::HipDataType_v<typename ck::tuple_element_t<0, InDataTypeTuple>>,
                hiptensor::HipDataType_v<typename ck::tuple_element_t<0, OutDataTypeTuple>>,
                typeid(ElementwiseOperation).hash_code(),
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
}
#endif //  INSTANCE_PARAMS_HPP
