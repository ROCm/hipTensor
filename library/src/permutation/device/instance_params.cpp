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

#include "instance_params.hpp"

namespace ck::tensor_operation::device::instance
{
    std::vector<hiptensor::Uid> getHashCodesWithAllInOutScalarPerVectorSeq(
        hipDataType                           typeIn,
        hipDataType                           typeOut,
        hiptensorOperator_t                   aOp,
        hiptensorOperator_t                   bOp,
        hiptensor::PermutationOpId_t          scale,
        index_t                               numDim,
        hiptensor::InstanceHyperParams const& hyperParams)
    {
        std::vector<hiptensor::Uid> hashCodes;

        // - IMPORTANT: keep the order of scalarPerVectorSeq from 16 to 0 so that the instance
        //      with greater scalarPerVectorSeq will be chose.
        // - scalarPerVectorSeq is 0 when it is CPU reference instance.
        // - `hashCodes` may contain hash codes that not represent any instances. It is not a problem
        //      since these hash codes will be ignored.
        auto                        scalarPerVectorSeqs = std::vector<index_t>{16, 8, 4, 2, 1, 0};
        index_t                     blockSize           = std::get<0>(hyperParams);
        index_t                     m0PerBlock          = std::get<1>(hyperParams);
        index_t                     m1PerBlock          = std::get<2>(hyperParams);
        index_t                     m0PerThread         = std::get<3>(hyperParams);
        index_t                     m1PerThread         = std::get<4>(hyperParams);
        std::pair<index_t, index_t> threadClusterArrangeOrder = std::get<5>(hyperParams);
        index_t                     inScalarPerVectorSeq      = std::get<6>(hyperParams);
        for(auto scalarPerVectorSeq : scalarPerVectorSeqs)
        {
            if(scalarPerVectorSeq <= inScalarPerVectorSeq)
            {
                hashCodes.push_back(hiptensor::Hash{}(typeIn,
                                                      typeOut,
                                                      aOp,
                                                      bOp,
                                                      scale,
                                                      numDim,
                                                      blockSize,
                                                      m0PerBlock,
                                                      m1PerBlock,
                                                      m0PerThread,
                                                      m1PerThread,
                                                      threadClusterArrangeOrder.first,
                                                      threadClusterArrangeOrder.second,
                                                      scalarPerVectorSeq,
                                                      scalarPerVectorSeq));
            }
        }
        return hashCodes;
    }
}
