/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_PERMUTATION_SCALE_INSTANCES_HPP
#define HIPTENSOR_PERMUTATION_SCALE_INSTANCES_HPP

// Stdlib includes
#include <cstdlib>
#include <memory>
#include <vector>

// CK includes
#include <add_device_operation_instance.hpp>
#include <ck.hpp>
#include <ck/tensor_operation/gpu/device/device_elementwise.hpp>
#include <combined_element_wise_operation.hpp>
#include <device_elementwise_dynamic_vector_dims_impl.hpp>

#include "instance_params.hpp"

namespace ck
{
    namespace tensor_operation
    {
        namespace device
        {
            namespace instance
            {
                template <typename InDataTypeTuple,
                          typename OutDataTypeTuple,
                          typename ElementwiseOperation,
                          index_t NumDim>
                struct DeviceOperationInstanceFactory<
                    ck::tensor_operation::device::DeviceElementwise<InDataTypeTuple,
                                                                    OutDataTypeTuple,
                                                                    ElementwiseOperation,
                                                                    NumDim>>
                {
                    using DeviceOp = DeviceElementwise<InDataTypeTuple,
                                                       OutDataTypeTuple,
                                                       ElementwiseOperation,
                                                       NumDim>;

                    template <index_t BlockSize,
                              index_t M0PerBlock,
                              index_t M1PerBlock,
                              index_t M0PerThread,
                              index_t M1PerThread,
                              typename ThreadClusterArrangeOrder,
                              typename InScalarPerVectorSeq,
                              typename OutScalarPerVectorSeq,
                              typename Container>
                    static void addInstance(Container& container)
                    {
                        container.insert(
                            {DeviceElementwiseParams<InDataTypeTuple,
                                                     OutDataTypeTuple,
                                                     ElementwiseOperation,
                                                     NumDim,
                                                     BlockSize,
                                                     M0PerBlock,
                                                     M1PerBlock,
                                                     M0PerThread,
                                                     M1PerThread,
                                                     ThreadClusterArrangeOrder,
                                                     InScalarPerVectorSeq,
                                                     OutScalarPerVectorSeq>::id(),
                             std::make_unique<DeviceElementwiseImpl<InDataTypeTuple,
                                                                    OutDataTypeTuple,
                                                                    ElementwiseOperation,
                                                                    NumDim,
                                                                    BlockSize,
                                                                    M0PerBlock,
                                                                    M1PerBlock,
                                                                    M0PerThread,
                                                                    M1PerThread,
                                                                    ThreadClusterArrangeOrder,
                                                                    InScalarPerVectorSeq,
                                                                    OutScalarPerVectorSeq>>()});
                    }
                    static auto GetInstances()
                    {
                        std::unordered_map<std::size_t, std::unique_ptr<DeviceOp>> opPtrs;
                        // clang-format off
                        addInstance<256, 64, 64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>(opPtrs);
                        addInstance<256, 64,  64,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>(opPtrs);
                        addInstance<256, 128, 32,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>(opPtrs);
                        addInstance<256, 32,  128, 4, 4,  ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>(opPtrs);
                        addInstance<128, 64,  32,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>(opPtrs);
                        addInstance<128, 32,  64,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>(opPtrs);
                        addInstance<128, 16,  128, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>(opPtrs);
                        addInstance<128, 128, 16,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>(opPtrs);
                        addInstance<64,  32,  32,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>(opPtrs);
                        addInstance<64,  16,  64,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>(opPtrs);
                        addInstance<64,  64,  16,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>(opPtrs);
                        addInstance<32,  32,  16,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>(opPtrs);
                        addInstance<32,  16,  32,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>(opPtrs);

                        addInstance<256, 128, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>(opPtrs);
                        addInstance<256, 256, 64,  8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>(opPtrs);
                        addInstance<256,  64, 256, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>(opPtrs);
                        addInstance<128, 128, 64,  8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>(opPtrs);
                        addInstance<128,  64, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>(opPtrs);
                        addInstance<128,  32, 256, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>(opPtrs);
                        addInstance<128, 256, 32,  8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>(opPtrs);
                        addInstance<64,   64, 64,  8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>(opPtrs);
                        addInstance<64,   32, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>(opPtrs);
                        addInstance<64,  128, 32,  8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>(opPtrs);
                        addInstance<32,   64, 32,  8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>(opPtrs);
                        addInstance<32,   32, 64,  8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>(opPtrs);


                        addInstance<256,  64,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>(opPtrs);
                        addInstance<256, 128,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>(opPtrs);
                        addInstance<256,  32, 128, 4, 4,  ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>(opPtrs);
                        addInstance<128,  64,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>(opPtrs);
                        addInstance<128,  32,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>(opPtrs);
                        addInstance<128,  16, 128, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>(opPtrs);
                        addInstance<128, 128,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>(opPtrs);
                        addInstance<64,   32,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>(opPtrs);
                        addInstance<64,   16,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>(opPtrs);
                        addInstance<64,   64,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>(opPtrs);
                        addInstance<32,   32,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>(opPtrs);
                        addInstance<32,   16,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>(opPtrs);
                        // clang-format on
                        return opPtrs;
                    }
                };

            } // namespace instance
        } // namespace device
    } // namespace tensor_operation
} // namespace ck

#endif // HIPTENSOR_PERMUTATION_SCALE_INSTANCES_HPP
