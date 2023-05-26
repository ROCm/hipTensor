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
#include "hash.hpp"

namespace std
{
    template <>
    struct std::hash<hiptensor::ContractionSolution>
    {
        std::size_t operator()(hiptensor::ContractionSolution const& s) const noexcept
        {
            return std::hash<hiptensor::ContractionSolutionParams>{}(*s.params());
        }
    };
}

namespace hiptensor
{
    template <typename DeviceOp, typename Enabler = void>
    class ContractionSolutionImpl;

    template <typename DeviceOp>
    class ContractionSolutionImpl<
        DeviceOp,
        std::enable_if_t<std::is_same_v<typename MetaTraits<DeviceOp>::CDEOp,
                                        ck::tensor_operation::element_wise::Bilinear>>>
        : public ContractionSolution
    {
    public:
        ContractionSolutionImpl(std::unique_ptr<DeviceOp>&& deviceOp)
            : ContractionSolution(std::move(deviceOp),
                                  std::make_unique<ContractionSolutionParamsImpl<DeviceOp>>())
        {
        }

        bool initArgs(void const*                     alpha,
                      void const*                     A,
                      void const*                     B,
                      void const*                     beta,
                      void const*                     D,
                      void*                           E,
                      std::vector<ck::index_t> const& a_ms_ks_lengths,
                      std::vector<ck::index_t> const& a_ms_ks_strides,
                      std::vector<ck::index_t> const& b_ns_ks_lengths,
                      std::vector<ck::index_t> const& b_ns_ks_strides,
                      std::vector<ck::index_t> const& ds_ms_ns_lengths,
                      std::vector<ck::index_t> const& ds_ms_ns_strides,
                      std::vector<ck::index_t> const& e_ms_ns_lengths,
                      std::vector<ck::index_t> const& e_ms_ns_strides,
                      void*                           workspacePtr) override
        {
            using Base   = ContractionSolution;
            using Traits = MetaTraits<DeviceOp>;

            // Clear out the previous arguments
            resetArgs();

            // Promote to derived class for necessary functions such as
            // MakeArgumentPointer and MakeInvokerPointer.
            auto* deviceOp = dynamic_cast<DeviceOp*>(Base::mDeviceOp.get());

            // Note: CK ALWAYS uses float for alpha / beta in contraction multipleD
            auto alphaF = hiptensor::readVal<float>(alpha, HipDataType_v<typename Traits::EDataT>);
            auto betaF = hiptensor::readVal<float>(beta, HipDataType_v<typename Traits::EDataT>);

            // Initialize the argument pointer
            Base::mArgPtr = std::move(deviceOp->MakeArgumentPointer(
                A,
                B,
                std::array<const void*, 1>{D},
                E,
                a_ms_ks_lengths,
                a_ms_ks_strides,
                b_ns_ks_lengths,
                b_ns_ks_strides,
                std::array<std::vector<ck::index_t>, 1>{ds_ms_ns_lengths},
                std::array<std::vector<ck::index_t>, 1>{ds_ms_ns_strides},
                e_ms_ns_lengths,
                e_ms_ns_strides,
                typename Traits::AOp{},
                typename Traits::BOp{},
                typename Traits::CDEOp{alphaF, betaF}));

            // Attach the workspace pointer
            deviceOp->SetWorkSpacePointer(Base::mArgPtr.get(), workspacePtr);

            // Initialize the invoker
            Base::mInvokerPtr = std::move(deviceOp->MakeInvokerPointer());

            // Fill problem metrics
            Base::mM = std::accumulate(e_ms_ns_lengths.begin(),
                                       e_ms_ns_lengths.begin() + Traits::DimsM,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            Base::mN = std::accumulate(e_ms_ns_lengths.begin() + Traits::DimsM,
                                       e_ms_ns_lengths.begin() + Traits::DimsM + Traits::DimsN,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            Base::mK = std::accumulate(a_ms_ks_lengths.begin() + Traits::DimsM,
                                       a_ms_ks_lengths.begin() + Traits::DimsM + Traits::DimsK,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            // Byte count
            Base::mBytes = sizeof(typename Traits::ADataT) * Base::mM * Base::mK
                           + sizeof(typename Traits::BDataT) * Base::mK * Base::mN
                           + sizeof(typename Traits::DDataT) * Base::mM * Base::mN
                           + sizeof(typename Traits::EDataT) * Base::mM * Base::mN;

            // Arg test
            Base::mValid = deviceOp->IsSupportedArgument(Base::mArgPtr.get());

            return mValid;
        }
    };

    template <typename DeviceOp>
    class ContractionSolutionImpl<
        DeviceOp,
        std::enable_if_t<std::is_same_v<typename MetaTraits<DeviceOp>::CDEOp,
                                        ck::tensor_operation::element_wise::Scale>>>
        : public ContractionSolution
    {
    public:
        ContractionSolutionImpl(std::unique_ptr<DeviceOp>&& deviceOp)
            : ContractionSolution(std::move(deviceOp),
                                  std::make_unique<ContractionSolutionParamsImpl<DeviceOp>>())
        {
        }

        bool initArgs(void const*                     alpha,
                      void const*                     A,
                      void const*                     B,
                      void const*                     beta,
                      void const*                     D,
                      void*                           E,
                      std::vector<ck::index_t> const& a_ms_ks_lengths,
                      std::vector<ck::index_t> const& a_ms_ks_strides,
                      std::vector<ck::index_t> const& b_ns_ks_lengths,
                      std::vector<ck::index_t> const& b_ns_ks_strides,
                      std::vector<ck::index_t> const& ds_ms_ns_lengths,
                      std::vector<ck::index_t> const& ds_ms_ns_strides,
                      std::vector<ck::index_t> const& e_ms_ns_lengths,
                      std::vector<ck::index_t> const& e_ms_ns_strides,
                      void*                           workspacePtr) override
        {
            using Base   = ContractionSolution;
            using Traits = MetaTraits<DeviceOp>;

            // Clear previous data
            resetArgs();

            // Promote to derived class for necessary functions such as
            // MakeArgumentPointer and MakeInvokerPointer.
            auto* deviceOp = dynamic_cast<DeviceOp*>(Base::mDeviceOp.get());

            // Note: CK ALWAYS uses float for alpha / beta in contraction multipleD
            auto alphaF = hiptensor::readVal<float>(alpha, HipDataType_v<typename Traits::EDataT>);

            // Initialize the argument pointer
            Base::mArgPtr
                = std::move(deviceOp->MakeArgumentPointer(A,
                                                          B,
                                                          std::array<const void*, 0>{},
                                                          E,
                                                          a_ms_ks_lengths,
                                                          a_ms_ks_strides,
                                                          b_ns_ks_lengths,
                                                          b_ns_ks_strides,
                                                          std::array<std::vector<ck::index_t>, 0>{},
                                                          std::array<std::vector<ck::index_t>, 0>{},
                                                          e_ms_ns_lengths,
                                                          e_ms_ns_strides,
                                                          typename Traits::AOp{},
                                                          typename Traits::BOp{},
                                                          typename Traits::CDEOp{alphaF}));

            // Attach the workspace pointer
            deviceOp->SetWorkSpacePointer(Base::mArgPtr.get(), workspacePtr);

            // Initialize the invoker
            Base::mInvokerPtr = std::move(deviceOp->MakeInvokerPointer());

            // Fill problem metrics
            Base::mM = std::accumulate(e_ms_ns_lengths.begin(),
                                       e_ms_ns_lengths.begin() + Traits::DimsM,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            Base::mN = std::accumulate(e_ms_ns_lengths.begin() + Traits::DimsM,
                                       e_ms_ns_lengths.begin() + Traits::DimsM + Traits::DimsN,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            Base::mK = std::accumulate(a_ms_ks_lengths.begin() + Traits::DimsM,
                                       a_ms_ks_lengths.begin() + Traits::DimsM + Traits::DimsK,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            // Byte count
            Base::mBytes = sizeof(typename Traits::ADataT) * Base::mM * Base::mK
                           + sizeof(typename Traits::BDataT) * Base::mK * Base::mN
                           + sizeof(typename Traits::EDataT) * Base::mM * Base::mN;

            // Arg test
            Base::mValid = deviceOp->IsSupportedArgument(Base::mArgPtr.get());

            return Base::mValid;
        }
    };

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
    std::vector<std::unique_ptr<hiptensor::ContractionSolution>> enumerateContractionSolutions()
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

        std::vector<std::unique_ptr<ContractionSolution>> result;
        for(auto& opPtr : Factory::GetInstances())
        {
            result.push_back(
                std::make_unique<ContractionSolutionImpl<ContractionOp>>(std::move(opPtr)));
        }
        return result;
    }

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_SOLUTION_IMPL_HPP
