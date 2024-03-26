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

#ifndef HIPTENSOR_PERMUTATION_TYPES_IMPL_HPP
#define HIPTENSOR_PERMUTATION_TYPES_IMPL_HPP

// CK includes
#include <element_wise_operation.hpp>

#include "permutation_types.hpp"
#include <hiptensor/hiptensor_types.hpp>

namespace hiptensor
{
    // Specialize overrides for runtime ElementWiseOperatorType
    template <>
    struct ElementWiseOperatorType<ck::tensor_operation::element_wise::PassThrough>
    {
        static constexpr auto value = hiptensorOperator_t::HIPTENSOR_OP_IDENTITY;
    };

    template <>
    struct ElementWiseOperatorType<ck::tensor_operation::element_wise::UnarySquare>
    {
        static constexpr auto value = hiptensorOperator_t::HIPTENSOR_OP_SQRT;
    };

    // Specialize overrides for runtime PermutationOperatorType
    template <>
    struct PermutationOperatorType<ck::tensor_operation::element_wise::Scale>
    {
        static constexpr auto value = PermutationOpId_t::SCALE;
    };

} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_TYPES_IMPL_HPP
