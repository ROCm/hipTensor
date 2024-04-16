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

#ifndef PERMUTATION_SCALE_F16_INSTANCES_HPP
#define PERMUTATION_SCALE_F16_INSTANCES_HPP

#include "common.hpp"

namespace ck
{
    namespace tensor_operation
    {
        namespace device
        {
            namespace instance
            {
                using F16 = ck::half_t;

                // clang-format off
                template <typename AOp,
                          typename BOp,
                          typename Scale,
                          index_t NDims>
                using device_permute_scale_f16_instances =
                    std::tuple <
                        DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, AOp, BOp, Scale, NDims, 1, ck::Sequence<1>, ck::Sequence<1>>,
                        DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, AOp, BOp, Scale, NDims, 8, ck::Sequence<8>, ck::Sequence<1>>,
                        DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, AOp, BOp, Scale, NDims, 4, ck::Sequence<4>, ck::Sequence<1>>,
                        DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, AOp, BOp, Scale, NDims, 2, ck::Sequence<2>, ck::Sequence<1>>
                    >;
                // clang-format on
            } // namespace instance
        } // namespace device
    } // namespace tensor_operation
} // namespace ck

#endif // PERMUTATION_SCALE_F16_INSTANCES_HPP
