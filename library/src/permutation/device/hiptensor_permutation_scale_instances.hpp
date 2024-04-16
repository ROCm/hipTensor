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

#ifndef HIPTENSOR_PERMUTATION_SCALE_INSTANCES_HPP
#define HIPTENSOR_PERMUTATION_SCALE_INSTANCES_HPP

#include "permutation_scale_2d_instances.hpp"
#include "permutation_scale_3d_instances.hpp"
#include "permutation_scale_4d_instances.hpp"
#include "permutation_scale_5d_instances.hpp"
#include "permutation_scale_6d_instances.hpp"

namespace ck
{
    namespace tensor_operation
    {
        namespace device
        {
            namespace instance
            {
                using F16 = ck::half_t;
                using F32 = float;

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

                template <typename AOp,
                          typename BOp,
                          typename Scale,
                          index_t NDims>
                using device_permute_scale_f32_instances = std::tuple<
                        DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, AOp, BOp, Scale, NDims, 1, ck::Sequence<1>, ck::Sequence<1>>,
                        DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, AOp, BOp, Scale, NDims, 8, ck::Sequence<8>, ck::Sequence<1>>,
                        DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, AOp, BOp, Scale, NDims, 4, ck::Sequence<4>, ck::Sequence<1>>,
                        DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, AOp, BOp, Scale, NDims, 2, ck::Sequence<2>, ck::Sequence<1>>
                    >;
                // clang-format on

                template <typename InDataTypeTuple,
                          typename OutDataTypeTuple,
                          typename AOp,
                          typename BOp,
                          typename Scale,
                          index_t NumDim>
                struct DeviceOperationInstanceFactory<
                    ck::tensor_operation::device::DeviceElementwise<InDataTypeTuple,
                                                                    OutDataTypeTuple,
                                                                    AOp,
                                                                    BOp,
                                                                    Scale,
                                                                    NumDim>>
                {
                    using DeviceOp = DeviceElementwise<InDataTypeTuple,
                                                       OutDataTypeTuple,
                                                       AOp,
                                                       BOp,
                                                       Scale,
                                                       NumDim>;

                    static auto GetInstances()
                    {
                        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
                        if constexpr(NumDim == 1)
                        {
                            if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F32>> &&
                                         is_same_v<OutDataTypeTuple, ck::Tuple<F32>>)
                            {
                                add_device_permute_scale_1d_f32_instances(op_ptrs);
                            }
                            else if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F16>> &&
                                              is_same_v<OutDataTypeTuple, ck::Tuple<F16>>)
                            {
                                add_device_permute_scale_1d_f16_instances(op_ptrs);
                            }
                        }
                        else if constexpr(NumDim == 2)
                        {
                            if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F32>> &&
                                         is_same_v<OutDataTypeTuple, ck::Tuple<F32>>)
                            {
                                add_device_permute_scale_2d_f32_instances(op_ptrs);
                            }
                            else if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F16>> &&
                                              is_same_v<OutDataTypeTuple, ck::Tuple<F16>>)
                            {
                                add_device_permute_scale_2d_f16_instances(op_ptrs);
                            }
                        }
                        else if constexpr(NumDim == 3)
                        {
                            if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F32>> &&
                                         is_same_v<OutDataTypeTuple, ck::Tuple<F32>>)
                            {
                                add_device_permute_scale_3d_f32_instances(op_ptrs);
                            }
                            else if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F16>> &&
                                              is_same_v<OutDataTypeTuple, ck::Tuple<F16>>)
                            {
                                add_device_permute_scale_3d_f16_instances(op_ptrs);
                            }
                        }
                        else if constexpr(NumDim == 4)
                        {
                            if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F32>> &&
                                         is_same_v<OutDataTypeTuple, ck::Tuple<F32>>)
                            {
                                add_device_permute_scale_4d_f32_instances(op_ptrs);
                            }
                            else if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F16>> &&
                                              is_same_v<OutDataTypeTuple, ck::Tuple<F16>>)
                            {
                                add_device_permute_scale_4d_f16_instances(op_ptrs);
                            }
                        }
                        else if constexpr(NumDim == 5)
                        {
                            if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F32>> &&
                                         is_same_v<OutDataTypeTuple, ck::Tuple<F32>>)
                            {
                                add_device_permute_scale_5d_f32_instances(op_ptrs);
                            }
                            else if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F16>> &&
                                              is_same_v<OutDataTypeTuple, ck::Tuple<F16>>)
                            {
                                add_device_permute_scale_5d_f16_instances(op_ptrs);
                            }
                        }
                        else if constexpr(NumDim == 6)
                        {
                            if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F32>> &&
                                         is_same_v<OutDataTypeTuple, ck::Tuple<F32>>)
                            {
                                add_device_permute_scale_6d_f32_instances(op_ptrs);
                            }
                            else if constexpr(is_same_v<InDataTypeTuple, ck::Tuple<F16>> &&
                                              is_same_v<OutDataTypeTuple, ck::Tuple<F16>>)
                            {
                                add_device_permute_scale_6d_f16_instances(op_ptrs);
                            }
                        }
                        return op_ptrs;
                    }
                };

            } // namespace instance
        } // namespace device
    } // namespace tensor_operation
} // namespace ck

#endif // HIPTENSOR_PERMUTATION_SCALE_INSTANCES_HPP
