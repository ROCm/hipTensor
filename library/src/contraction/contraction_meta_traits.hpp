/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2023 Advanced Micro Devices, Inc.
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

#ifndef HIPTENSOR_CONTRACTION_META_TRAITS_HPP
#define HIPTENSOR_CONTRACTION_META_TRAITS_HPP

// CK includes
#include <contraction_bilinear.hpp>
#include <contraction_scale.hpp>
#include <device_contraction_multiple_d.hpp>
#include <element_wise_operation.hpp>

// hiptensor includes
#include "meta_traits.hpp"

namespace hiptensor
{

    // Partial specialize for Bilinear contraction
    template <ck::index_t NumDimsM,
              ck::index_t NumDimsN,
              ck::index_t NumDimsK,
              typename ADataType,
              typename BDataType,
              typename DsDataType,
              typename EDataType,
              typename AElementwiseOperation,
              typename BElementwiseOperation>
    struct MetaTraits<ck::tensor_operation::device::DeviceContractionMultipleD<
        NumDimsM,
        NumDimsN,
        NumDimsK,
        ADataType,
        BDataType,
        ck::Tuple<DsDataType>,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        ck::tensor_operation::element_wise::Bilinear>>
    {
        constexpr static ck::index_t DimsM = NumDimsM;
        constexpr static ck::index_t DimsN = NumDimsN;
        constexpr static ck::index_t DimsK = NumDimsK;
        using ADataT                       = ADataType;
        using BDataT                       = BDataType;
        using DDataT                       = DsDataType;
        using EDataT                       = EDataType;
        using AOp                          = AElementwiseOperation;
        using BOp                          = BElementwiseOperation;
        using CDEOp                        = ck::tensor_operation::element_wise::Bilinear;
    };

    // Partial specialize for Scale contraction
    template <ck::index_t NumDimsM,
              ck::index_t NumDimsN,
              ck::index_t NumDimsK,
              typename ADataType,
              typename BDataType,
              typename EDataType,
              typename AElementwiseOperation,
              typename BElementwiseOperation>
    struct MetaTraits<ck::tensor_operation::device::DeviceContractionMultipleD<
        NumDimsM,
        NumDimsN,
        NumDimsK,
        ADataType,
        BDataType,
        ck::Tuple<>,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        ck::tensor_operation::element_wise::Scale>>
    {
        constexpr static ck::index_t DimsM = NumDimsM;
        constexpr static ck::index_t DimsN = NumDimsN;
        constexpr static ck::index_t DimsK = NumDimsK;
        using ADataT                       = ADataType;
        using BDataT                       = BDataType;
        using EDataT                       = EDataType;
        using AOp                          = AElementwiseOperation;
        using BOp                          = BElementwiseOperation;
        using CDEOp                        = ck::tensor_operation::element_wise::Scale;
    };

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_META_TRAITS_HPP
