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

// This (ifndef) is a hack to use customized behavior for buffer load rather than using default
// setting Don't use this hack unless absolutely necessary!
// FIXME: make the behavior of buffer load a configurable (template) parameter of each device op
#define CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK 1

#include "hiptensor_permutation_scale_instances.hpp"

namespace ck
{
    namespace tensor_operation
    {
        namespace device
        {
            namespace instance
            {
                void add_device_permute_scale_4d_f16_instances(
                    std::vector<std::unique_ptr<
                        DeviceElementwise<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, UnaryOp, Scale, 4>>>& instances)
                {
                    add_device_operation_instances(instances,
                                                   device_permute_scale_f16_instances<1>{});
                }

                void add_device_permute_scale_4d_f32_instances(
                    std::vector<std::unique_ptr<
                        DeviceElementwise<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, UnaryOp, Scale, 4>>>& instances)
                {
                    add_device_operation_instances(instances,
                                                   device_permute_scale_f32_instances<1>{});
                }

            } // namespace instance
        } // namespace device
    } // namespace tensor_operation
} // namespace ck
