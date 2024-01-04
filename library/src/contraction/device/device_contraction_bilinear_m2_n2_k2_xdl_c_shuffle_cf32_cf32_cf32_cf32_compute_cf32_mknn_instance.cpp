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

#include "common.hpp"
#include "device_contraction_bilinear_complex.hpp"

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/contraction/device_contraction_instance.hpp"
#include "ck/tensor_operation/gpu/device/device_contraction_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck
{
    namespace tensor_operation
    {
        namespace device
        {
            namespace instance
            {
                using F32               = float;
                using CF32              = hipFloatComplex;
                using CF32_Tuple        = ck::Tuple<CF32>;
                using BilinearComplex   = element_wise::BilinearComplex;

                // A[m0, m1, k0, k1] * B[n0, n1, k0, k1] + D[m0, m1, n0, n1] = E[m0, m1, n0, n1]
                // m/k/n/n are the fast changing dimension for A/B/D/E
                using device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_cf32_cf32_cf32_cf32_compute_cf32_mknn_instance
                    = device_contraction_mk_instance<CF32,
                                                     CF32,
                                                     F32,
                                                     F32,
                                                     CF32_Tuple,
                                                     CF32,
                                                     CF32,
                                                     PassThrough,
                                                     PassThrough,
                                                     BilinearComplex>;

                void
                    add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_cf32_cf32_cf32_cf32_compute_cf32_mknn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                               2,
                                                                               2,
                                                                               CF32,
                                                                               CF32,
                                                                               CF32_Tuple,
                                                                               CF32,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               BilinearComplex,
                                                                               CF32>>>& instances)
                {
                    add_device_operation_instances(
                        instances,
                        device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_cf32_cf32_cf32_cf32_compute_cf32_mknn_instance{});
                }

            } // namespace instance
        } // namespace device
    } // namespace tensor_operation
} // namespace ck
