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

#ifndef CONTRACTION_SCALE_HPP
#define CONTRACTION_SCALE_HPP

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
                void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_kkn_instance(
                    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                           2,
                                                                           2,
                                                                           F32,
                                                                           F32,
                                                                           Empty_Tuple,
                                                                           F32,
                                                                           PassThrough,
                                                                           PassThrough,
                                                                           Scale>>>& instances);

                void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_knn_instance(
                    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                           2,
                                                                           2,
                                                                           F32,
                                                                           F32,
                                                                           Empty_Tuple,
                                                                           F32,
                                                                           PassThrough,
                                                                           PassThrough,
                                                                           Scale>>>& instances);

                void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_mkn_instance(
                    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                           2,
                                                                           2,
                                                                           F32,
                                                                           F32,
                                                                           Empty_Tuple,
                                                                           F32,
                                                                           PassThrough,
                                                                           PassThrough,
                                                                           Scale>>>& instances);

                void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_mnn_instance(
                    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                           2,
                                                                           2,
                                                                           F32,
                                                                           F32,
                                                                           Empty_Tuple,
                                                                           F32,
                                                                           PassThrough,
                                                                           PassThrough,
                                                                           Scale>>>& instances);

                // double
                void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_kkn_instance(
                    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                           2,
                                                                           2,
                                                                           F64,
                                                                           F64,
                                                                           Empty_Tuple,
                                                                           F64,
                                                                           PassThrough,
                                                                           PassThrough,
                                                                           Scale>>>& instances);

                void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_knn_instance(
                    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                           2,
                                                                           2,
                                                                           F64,
                                                                           F64,
                                                                           Empty_Tuple,
                                                                           F64,
                                                                           PassThrough,
                                                                           PassThrough,
                                                                           Scale>>>& instances);

                void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_mkn_instance(
                    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                           2,
                                                                           2,
                                                                           F64,
                                                                           F64,
                                                                           Empty_Tuple,
                                                                           F64,
                                                                           PassThrough,
                                                                           PassThrough,
                                                                           Scale>>>& instances);

                void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_mnn_instance(
                    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                                           2,
                                                                           2,
                                                                           F64,
                                                                           F64,
                                                                           Empty_Tuple,
                                                                           F64,
                                                                           PassThrough,
                                                                           PassThrough,
                                                                           Scale>>>& instances);

                // Contraction + Scale
                template <index_t NumDimM,
                          index_t NumDimN,
                          index_t NumDimK,
                          typename ADataType,
                          typename BDataType,
                          typename EDataType>
                struct HipTensorDeviceOperationInstanceFactory<
                    ck::tensor_operation::device::DeviceContractionMultipleD<
                        NumDimM,
                        NumDimN,
                        NumDimK,
                        ADataType,
                        BDataType,
                        ck::Tuple<>,
                        EDataType,
                        ck::tensor_operation::element_wise::PassThrough,
                        ck::tensor_operation::element_wise::PassThrough,
                        ck::tensor_operation::element_wise::Scale>>
                {
                    using DeviceOp = DeviceContractionMultipleD<
                        NumDimM,
                        NumDimN,
                        NumDimK,
                        ADataType,
                        BDataType,
                        ck::Tuple<>,
                        EDataType,
                        ck::tensor_operation::element_wise::PassThrough,
                        ck::tensor_operation::element_wise::PassThrough,
                        ck::tensor_operation::element_wise::Scale>;

                    static auto GetInstances()
                    {
                        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

                        if constexpr(is_same_v<ADataType, float> && is_same_v<BDataType, float>
                                     && is_same_v<EDataType, float>)
                        {
                            if constexpr(NumDimM == 2 && NumDimN == 2 && NumDimK == 2)
                            {
                                add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_kkn_instance(
                                    op_ptrs);
                                add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_knn_instance(
                                    op_ptrs);
                                add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_mkn_instance(
                                    op_ptrs);
                                add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_mnn_instance(
                                    op_ptrs);
                            }
                        }

                        if constexpr(is_same_v<ADataType, double> && is_same_v<BDataType, double>
                                     && is_same_v<EDataType, double>)
                        {
                            if constexpr(NumDimM == 2 && NumDimN == 2 && NumDimK == 2)
                            {
                                add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_kkn_instance(
                                    op_ptrs);
                                add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_knn_instance(
                                    op_ptrs);
                                add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_mkn_instance(
                                    op_ptrs);
                                add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_mnn_instance(
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

#endif // CONTRACTION_SCALE_HPP
