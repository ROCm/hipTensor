/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#ifndef CONTRACTION_BILINEAR_HPP
#define CONTRACTION_BILINEAR_HPP

#include "common.hpp"

namespace ck
{
    namespace tensor_operation
    {
        namespace device
        {
            namespace instance
            {

                // float
                void
                    add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                               2,
                                                                               2,
                                                                               F32,
                                                                               F32,
                                                                               F32_Tuple,
                                                                               F32,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               Bilinear>>>&
                            instances);

                void
                    add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_knnn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                               2,
                                                                               2,
                                                                               F32,
                                                                               F32,
                                                                               F32_Tuple,
                                                                               F32,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               Bilinear>>>&
                            instances);

                void
                    add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_mknn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                               2,
                                                                               2,
                                                                               F32,
                                                                               F32,
                                                                               F32_Tuple,
                                                                               F32,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               Bilinear>>>&
                            instances);

                void
                    add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_mnnn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                               2,
                                                                               2,
                                                                               F32,
                                                                               F32,
                                                                               F32_Tuple,
                                                                               F32,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               Bilinear>>>&
                            instances);

                // double
                void
                    add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_kknn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                               2,
                                                                               2,
                                                                               F64,
                                                                               F64,
                                                                               F64_Tuple,
                                                                               F64,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               Bilinear>>>&
                            instances);

                void
                    add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_knnn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                               2,
                                                                               2,
                                                                               F64,
                                                                               F64,
                                                                               F64_Tuple,
                                                                               F64,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               Bilinear>>>&
                            instances);

                void
                    add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_mknn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                               2,
                                                                               2,
                                                                               F64,
                                                                               F64,
                                                                               F64_Tuple,
                                                                               F64,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               Bilinear>>>&
                            instances);

                void
                    add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_mnnn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                               2,
                                                                               2,
                                                                               F64,
                                                                               F64,
                                                                               F64_Tuple,
                                                                               F64,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               Bilinear>>>&
                            instances);

                // Contraction + Bilinear
                template <index_t NumDimM,
                          index_t NumDimN,
                          index_t NumDimK,
                          typename ADataType,
                          typename BDataType,
                          typename DDataType,
                          typename EDataType>
                struct DeviceOperationInstanceFactory<
                    ck::tensor_operation::device::DeviceContractionMultipleD<
                        NumDimM,
                        NumDimN,
                        NumDimK,
                        ADataType,
                        BDataType,
                        ck::Tuple<DDataType>,
                        EDataType,
                        ck::tensor_operation::element_wise::PassThrough,
                        ck::tensor_operation::element_wise::PassThrough,
                        ck::tensor_operation::element_wise::Bilinear>>
                {
                    using DeviceOp = DeviceContractionMultipleD<
                        NumDimM,
                        NumDimN,
                        NumDimK,
                        ADataType,
                        BDataType,
                        ck::Tuple<DDataType>,
                        EDataType,
                        ck::tensor_operation::element_wise::PassThrough,
                        ck::tensor_operation::element_wise::PassThrough,
                        ck::tensor_operation::element_wise::Bilinear>;

                    static auto GetInstances()
                    {
                        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

                        if constexpr(is_same_v<ADataType, float> && is_same_v<BDataType, float>
                                     && is_same_v<DDataType, float> && is_same_v<EDataType, float>)
                        {
                            if constexpr(NumDimM == 2 && NumDimN == 2 && NumDimK == 2)
                            {
                                add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_knnn_instance(
                                    op_ptrs);
                                add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance(
                                    op_ptrs);
                                add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_mnnn_instance(
                                    op_ptrs);
                                add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_mknn_instance(
                                    op_ptrs);
                            }
                        }

                        if constexpr(is_same_v<ADataType, double> && is_same_v<BDataType, double>
                                     && is_same_v<DDataType, double>
                                     && is_same_v<EDataType, double>)
                        {
                            if constexpr(NumDimM == 2 && NumDimN == 2 && NumDimK == 2)
                            {
                                add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_kknn_instance(
                                    op_ptrs);
                                add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_knnn_instance(
                                    op_ptrs);
                                add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_mnnn_instance(
                                    op_ptrs);
                                add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_mknn_instance(
                                    op_ptrs);
                            }
                        }

                        return op_ptrs;
                    }
                };

            } // namespace instance
        } // namespace device
    } // namespace tensor_operation
} // namespace ck

#endif // CONTRACTION_BILINEAR_HPP
