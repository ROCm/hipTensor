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

#ifndef HIPTENSOR_PERMUTATION_META_TRAITS_HPP
#define HIPTENSOR_PERMUTATION_META_TRAITS_HPP

// CK includes
#include <device_elementwise_scale_impl.hpp>
#include <element_wise_operation.hpp>
#include <device_operation_instance_factory.hpp>

// hiptensor includes
#include "device/hiptensor_permutation_scale_instances.hpp"
#include "data_types.hpp"
#include "meta_traits.hpp"

namespace hiptensor
{
    // Meta traits for Scalar permutation
    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename ElementwiseOperation,
              typename UnaryOperation,
              typename Scale,
              ck::index_t NumDim>
    struct MetaTraits<ck::tensor_operation::device::DeviceElementwise<
                                                                    InDataTypeTuple,
                                                                    OutDataTypeTuple,
                                                                    ElementwiseOperation,
                                                                    UnaryOperation,
                                                                    Scale,
                                                                    NumDim>>
    {
        constexpr static ck::index_t NDim = NumDim;

        using InDataT   = InDataTypeTuple;
        using OutDataT  = OutDataTypeTuple;
        
        using ElementOp = ElementwiseOperation;
        using UnaryOp   = UnaryOperation;
        using ScaleOp   = Scale;
    };
} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_META_TRAITS_HPP
