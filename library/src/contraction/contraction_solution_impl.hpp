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

#ifndef HIPTENSOR_CONTRACTION_SOLUTION_IMPL_HPP
#define HIPTENSOR_CONTRACTION_SOLUTION_IMPL_HPP

#include <numeric>

#include "contraction_solution.hpp"

namespace hiptensor
{

    template <
        typename DeviceOp,
        typename std::enable_if_t<std::is_same_v<typename MetaTraits<DeviceOp>::CDEOp,
                                                 ck::tensor_operation::element_wise::Bilinear>,
                                  void*> /*= nullptr*/>
    ContractionSolution::ContractionSolution(std::unique_ptr<DeviceOp>&& deviceOp)
        : mM(0)
        , mN(0)
        , mK(0)
        , mBytes(0)
        , mValid(false)
        , mDeviceOp(deviceOp.release()) // Take ownership, but store as opaque BaseOperator ptr
    {
        mInitArgs = [](ContractionSolution&                         launcher,
                       void const*                                  alpha,
                       void const*                                  A,
                       void const*                                  B,
                       void const*                                  beta,
                       void const*                                  D,
                       void*                                        E,
                       std::vector<ck::index_t> const&              a_ms_ns_lengths,
                       std::vector<ck::index_t> const&              a_ms_ks_strides,
                       std::vector<ck::index_t> const&              b_ns_ks_lengths,
                       std::vector<ck::index_t> const&              b_ns_ks_strides,
                       std::vector<std::vector<ck::index_t>> const& ds_ms_ns_lengths,
                       std::vector<std::vector<ck::index_t>> const& ds_ms_ns_strides,
                       std::vector<ck::index_t> const&              e_ms_ns_lengths,
                       std::vector<ck::index_t> const&              e_ms_ns_strides) {
            using Traits = MetaTraits<DeviceOp>;

            // Promote to derived class for necessary functions such as
            // MakeArgumentPointer and MakeInvokerPointer.
            auto* deviceOp = dynamic_cast<DeviceOp*>(launcher.mDeviceOp.get());

            // This specialization is for bilinear contraction
            launcher.mOpId = ContractionOpId_t::BILINEAR;

            // Initialize the argument pointer
            launcher.mArgPtr = std::move(deviceOp->MakeArgumentPointer(
                A,
                B,
                std::array<const void*, 1>{D},
                E,
                a_ms_ns_lengths,
                a_ms_ks_strides,
                b_ns_ks_lengths,
                b_ns_ks_strides,
                std::array<std::vector<ck::index_t>, 1>{ds_ms_ns_lengths[0]},
                std::array<std::vector<ck::index_t>, 1>{ds_ms_ns_strides[0]},
                e_ms_ns_lengths,
                e_ms_ns_strides,
                typename Traits::AOp{},
                typename Traits::BOp{},
                typename Traits::CDEOp{*(float*)alpha, *(float*)beta}));

            // Initialize the invoker
            launcher.mInvokerPtr = std::move(deviceOp->MakeInvokerPointer());

            // Get the kernel name
            launcher.mKernelName = deviceOp->GetTypeString();

            // Fill problem metrics
            launcher.mM = std::accumulate(e_ms_ns_lengths.begin(),
                                          e_ms_ns_lengths.begin() + Traits::DimsM,
                                          ck::index_t{1},
                                          std::multiplies<ck::index_t>{});

            launcher.mN = std::accumulate(e_ms_ns_lengths.begin() + Traits::DimsM,
                                          e_ms_ns_lengths.begin() + Traits::DimsM + Traits::DimsN,
                                          ck::index_t{1},
                                          std::multiplies<ck::index_t>{});

            launcher.mK = std::accumulate(e_ms_ns_lengths.begin() + Traits::DimsM,
                                          e_ms_ns_lengths.begin() + Traits::DimsM + Traits::DimsK,
                                          ck::index_t{1},
                                          std::multiplies<ck::index_t>{});

            // Byte count
            launcher.mBytes = sizeof(typename Traits::ADataT) * launcher.mM * launcher.mK
                              + sizeof(typename Traits::BDataT) * launcher.mK * launcher.mN
                              + sizeof(typename Traits::DDataT) * launcher.mM * launcher.mN
                              + sizeof(typename Traits::EDataT) * launcher.mM * launcher.mN;

            // Arg test
            launcher.mValid = deviceOp->IsSupportedArgument(launcher.mArgPtr.get());
        };
    }

    template <typename DeviceOp,
              typename std::enable_if_t<std::is_same_v<typename MetaTraits<DeviceOp>::CDEOp,
                                                       ck::tensor_operation::element_wise::Scale>,
                                        void*> /* = nullptr */>
    ContractionSolution::ContractionSolution(std::unique_ptr<DeviceOp>&& deviceOp)
        : mM(0)
        , mN(0)
        , mK(0)
        , mBytes(0)
        , mValid(false)
        , mDeviceOp(deviceOp.release()) // Take ownership, but store as opaque BaseOperator ptr
    {
        mInitArgs = [](ContractionSolution&                         launcher,
                       void const*                                  alpha,
                       void const*                                  A,
                       void const*                                  B,
                       void const*                                  beta,
                       void const*                                  D,
                       void*                                        E,
                       std::vector<ck::index_t> const&              a_ms_ns_lengths,
                       std::vector<ck::index_t> const&              a_ms_ks_strides,
                       std::vector<ck::index_t> const&              b_ns_ks_lengths,
                       std::vector<ck::index_t> const&              b_ns_ks_strides,
                       std::vector<std::vector<ck::index_t>> const& ds_ms_ns_lengths,
                       std::vector<std::vector<ck::index_t>> const& ds_ms_ns_strides,
                       std::vector<ck::index_t> const&              e_ms_ns_lengths,
                       std::vector<ck::index_t> const&              e_ms_ns_strides) {
            using Traits = MetaTraits<DeviceOp>;

            // Promote to derived class for necessary functions such as
            // MakeArgumentPointer and MakeInvokerPointer.
            auto* deviceOp = dynamic_cast<DeviceOp*>(launcher.mDeviceOp.get());

            // This specialization is for scale contraction
            launcher.mOpId = ContractionOpId_t::SCALE;

            // Initialize the argument pointer
            launcher.mArgPtr
                = std::move(deviceOp->MakeArgumentPointer(A,
                                                          B,
                                                          std::array<const void*, 0>{},
                                                          E,
                                                          a_ms_ns_lengths,
                                                          a_ms_ks_strides,
                                                          b_ns_ks_lengths,
                                                          b_ns_ks_strides,
                                                          std::array<std::vector<ck::index_t>, 0>{},
                                                          std::array<std::vector<ck::index_t>, 0>{},
                                                          e_ms_ns_lengths,
                                                          e_ms_ns_strides,
                                                          typename Traits::AOp{},
                                                          typename Traits::BOp{},
                                                          typename Traits::CDEOp{*(float*)alpha}));

            // Initialize the invoker
            launcher.mInvokerPtr = std::move(deviceOp->MakeInvokerPointer());

            // Get the kernel name
            launcher.mKernelName = deviceOp->GetTypeString();

            // Fill problem metrics
            launcher.mM = std::accumulate(e_ms_ns_lengths.begin(),
                                          e_ms_ns_lengths.begin() + Traits::DimsM,
                                          ck::index_t{1},
                                          std::multiplies<ck::index_t>{});

            launcher.mN = std::accumulate(e_ms_ns_lengths.begin() + Traits::DimsM,
                                          e_ms_ns_lengths.begin() + Traits::DimsM + Traits::DimsN,
                                          ck::index_t{1},
                                          std::multiplies<ck::index_t>{});

            launcher.mK = std::accumulate(a_ms_ns_lengths.begin() + Traits::DimsM,
                                          a_ms_ns_lengths.begin() + Traits::DimsM + Traits::DimsK,
                                          ck::index_t{1},
                                          std::multiplies<ck::index_t>{});

            // Byte count
            launcher.mBytes = sizeof(typename Traits::ADataT) * launcher.mM * launcher.mK
                              + sizeof(typename Traits::BDataT) * launcher.mK * launcher.mN
                              + sizeof(typename Traits::EDataT) * launcher.mM * launcher.mN;

            // Arg test
            launcher.mValid = deviceOp->IsSupportedArgument(launcher.mArgPtr.get());
        };
    }

    template <ck::index_t NumDimM,
              ck::index_t NumDimN,
              ck::index_t NumDimK,
              typename ADataType,
              typename BDataType,
              typename DsDataType,
              typename EDataType,
              typename AElementwiseOperation,
              typename BElementwiseOperation,
              typename CDEElementwiseOperation>
    std::vector<hiptensor::ContractionSolution> enumerateContractionSolutions()
    {
        using ContractionOp
            = ck::tensor_operation::device::DeviceContractionMultipleD<NumDimM,
                                                                       NumDimN,
                                                                       NumDimK,
                                                                       ADataType,
                                                                       BDataType,
                                                                       DsDataType,
                                                                       EDataType,
                                                                       AElementwiseOperation,
                                                                       BElementwiseOperation,
                                                                       CDEElementwiseOperation>;

        using Factory
            = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<ContractionOp>;

        std::vector<hiptensor::ContractionSolution> result;
        for(auto& opPtr : Factory::GetInstances())
        {
            result.push_back(hiptensor::ContractionSolution(std::move(opPtr)));
        }
        return result;
    }

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_SOLUTION_IMPL_HPP
