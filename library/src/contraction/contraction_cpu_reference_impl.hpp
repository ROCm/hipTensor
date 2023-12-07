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

#ifndef HIPTENSOR_CONTRACTION_CPU_REFERENCE_IMPL_HPP
#define HIPTENSOR_CONTRACTION_CPU_REFERENCE_IMPL_HPP

// Std includes
#include <array>
#include <numeric>
#include <vector>

// CK includes
#include <contraction_bilinear.hpp>
#include <contraction_scale.hpp>
#include <device_contraction_multiple_d.hpp>
#include <element_wise_operation.hpp>
#include <host_tensor.hpp>

#include "contraction_meta_traits.hpp"
#include "contraction_solution.hpp"

namespace hiptensor
{
    // hardcoded for NumDimM == NumDimN == NumDimK == 2
    //
    // ck::bhalf_t is ushort, cannot perform bhalf_t * bhalf_t
    // CK does not use ck::bhalf_t as AccDataType. But we still
    // add this guard here
    template <
        ck::index_t NumDimM,
        ck::index_t NumDimN,
        ck::index_t NumDimK,
        typename ADataType,
        typename BDataType,
        typename AccDataType,
        typename DsDataType,
        typename EDataType,
        typename AElementwiseOperation,
        typename BElementwiseOperation,
        typename CDEElementwiseOperation,
        typename ComputeDataType = ADataType,
        ck::enable_if_t<NumDimM == 2 && NumDimN == 2 && NumDimK == 2 && DsDataType::Size() <= 1
                            && !std::is_same_v<AccDataType, ck::bhalf_t>,
                        bool>
        = false>
    struct ReferenceContraction_M2_N2_K2
        : public ck::tensor_operation::device::DeviceContractionMultipleD<NumDimM,
                                                                          NumDimN,
                                                                          NumDimK,
                                                                          ADataType,
                                                                          BDataType,
                                                                          DsDataType,
                                                                          EDataType,
                                                                          AElementwiseOperation,
                                                                          BElementwiseOperation,
                                                                          CDEElementwiseOperation,
                                                                          ComputeDataType>
    {
        using BaseArgument = ck::tensor_operation::device::BaseArgument;
        using BaseInvoker  = ck::tensor_operation::device::BaseInvoker;
        using index_t      = ck::index_t;

        static constexpr ck::index_t NumDTensor = DsDataType::Size();

        // Argument
        struct Argument : public BaseArgument
        {
            Argument(void const*                                             p_a,
                     void const*                                             p_b,
                     std::array<void const*, NumDTensor>                     p_d,
                     void*                                                   p_e,
                     std::vector<ck::index_t> const&                         a_ms_ks_lengths,
                     std::vector<ck::index_t> const&                         a_ms_ks_strides,
                     std::vector<ck::index_t> const&                         b_ns_ks_lengths,
                     std::vector<ck::index_t> const&                         b_ns_ks_strides,
                     std::array<std::vector<ck::index_t>, NumDTensor> const& d_ms_ns_lengths,
                     std::array<std::vector<ck::index_t>, NumDTensor> const& d_ms_ns_strides,
                     std::vector<ck::index_t> const&                         e_ms_ns_lengths,
                     std::vector<ck::index_t> const&                         e_ms_ns_strides,
                     AElementwiseOperation                                   a_element_op,
                     BElementwiseOperation                                   b_element_op,
                     CDEElementwiseOperation                                 cde_element_op)
                : BaseArgument()
                , mA(p_a)
                , mB(p_b)
                , mD(p_d)
                , mE(p_e)
                , mA_ms_ks_lengths(a_ms_ks_lengths)
                , mA_ms_ks_strides(a_ms_ks_strides)
                , mB_ns_ks_lengths(b_ns_ks_lengths)
                , mB_ns_ks_strides(b_ns_ks_strides)
                , mD_ms_ns_lengths(d_ms_ns_lengths)
                , mD_ms_ns_strides(d_ms_ns_strides)
                , mE_ms_ns_lengths(e_ms_ns_lengths)
                , mE_ms_ns_strides(e_ms_ns_strides)
                , mOpA(a_element_op)
                , mOpB(b_element_op)
                , mOpCDE(cde_element_op)
            {
            }

            Argument(Argument const&)            = default;
            Argument& operator=(Argument const&) = default;
            ~Argument()                          = default;

            void const*                                      mA;
            void const*                                      mB;
            std::array<void const*, NumDTensor>              mD;
            void*                                            mE;
            std::vector<ck::index_t>                         mA_ms_ks_lengths;
            std::vector<ck::index_t>                         mA_ms_ks_strides;
            std::vector<ck::index_t>                         mB_ns_ks_lengths;
            std::vector<ck::index_t>                         mB_ns_ks_strides;
            std::array<std::vector<ck::index_t>, NumDTensor> mD_ms_ns_lengths;
            std::array<std::vector<ck::index_t>, NumDTensor> mD_ms_ns_strides;
            std::vector<ck::index_t>                         mE_ms_ns_lengths;
            std::vector<ck::index_t>                         mE_ms_ns_strides;

            AElementwiseOperation   mOpA;
            BElementwiseOperation   mOpB;
            CDEElementwiseOperation mOpCDE;
        };

        // Invoker
        struct Invoker : public BaseInvoker
        {
            using Argument = ReferenceContraction_M2_N2_K2::Argument;

            float Run(const Argument& arg)
            {
                auto offset = [](auto const& indices, auto const& strides) {
                    return std::inner_product(
                        indices.begin(), indices.end(), strides.begin(), std::size_t{0});
                };

                auto f_ms_ns = [&](auto m0, auto m1, auto n0, auto n1) {
                    AccDataType accum = 0;

                    auto K0 = arg.mA_ms_ks_lengths[2];
                    auto K1 = arg.mA_ms_ks_lengths[3];

                    for(size_t k0 = 0; k0 < K0; k0++)
                    {
                        for(size_t k1 = 0; k1 < K1; k1++)
                        {
                            auto indexA
                                = offset(std::vector<size_t>{m0, m1, k0, k1}, arg.mA_ms_ks_strides);
                            auto indexB
                                = offset(std::vector<size_t>{n0, n1, k0, k1}, arg.mB_ns_ks_strides);

                            AccDataType valA;
                            AccDataType valB;

                            // Element-wise ops
                            arg.mOpA(
                                valA,
                                ck::type_convert<ComputeDataType>(((ADataType*)arg.mA)[indexA]));
                            arg.mOpB(
                                valB,
                                ck::type_convert<ComputeDataType>(((BDataType*)arg.mB)[indexB]));

                            // Mult / accum
                            accum += valA * valB;
                        }
                    }

                    auto indexE = offset(std::vector<size_t>{m0, m1, n0, n1}, arg.mE_ms_ns_strides);

                    if constexpr(std::is_same_v<CDEElementwiseOperation,
                                                ck::tensor_operation::element_wise::Scale>)
                    {
                        arg.mOpCDE(((EDataType*)arg.mE)[indexE],
                                   ck::type_convert<EDataType>(accum));
                    }
                    else // bilinear
                    {
                        // NumDTensor will be 1 due to SFINAE of this class
                        auto indexD
                            = offset(std::vector<size_t>{m0, m1, n0, n1}, arg.mD_ms_ns_strides[0]);
                        arg.mOpCDE(((EDataType*)arg.mE)[indexE],
                                   ck::type_convert<EDataType>(accum),
                                   ((EDataType*)(arg.mD[0]))[indexD]);
                    }
                };

                make_ParallelTensorFunctor(f_ms_ns,
                                           arg.mE_ms_ns_lengths[0],
                                           arg.mE_ms_ns_lengths[1],
                                           arg.mE_ms_ns_lengths[2],
                                           arg.mE_ms_ns_lengths[3])(
                    std::thread::hardware_concurrency());

                return 0;
            }

            float Run(const BaseArgument* p_arg,
                      const StreamConfig& /* stream_config */ = StreamConfig{}) override
            {
                return Run(*dynamic_cast<const Argument*>(p_arg));
            }
        };

        static constexpr bool IsValidCompilationParameter()
        {
            // TODO: properly implement this check
            return true;
        }

        bool IsSupportedArgument(const BaseArgument*) override
        {
            return true;
        }

        static auto
            MakeArgument(void const*                                             p_a,
                         void const*                                             p_b,
                         std::array<void const*, NumDTensor>                     p_d,
                         void*                                                   p_e,
                         std::vector<ck::index_t> const&                         a_ms_ks_lengths,
                         std::vector<ck::index_t> const&                         a_ms_ks_strides,
                         std::vector<ck::index_t> const&                         b_ns_ks_lengths,
                         std::vector<ck::index_t> const&                         b_ns_ks_strides,
                         std::array<std::vector<ck::index_t>, NumDTensor> const& d_ms_ns_lengths,
                         std::array<std::vector<ck::index_t>, NumDTensor> const& d_ms_ns_strides,
                         std::vector<ck::index_t> const&                         e_ms_ns_lengths,
                         std::vector<ck::index_t> const&                         e_ms_ns_strides,
                         AElementwiseOperation                                   a_element_op,
                         BElementwiseOperation                                   b_element_op,
                         CDEElementwiseOperation                                 cde_element_op)
        {
            return Argument{p_a,
                            p_b,
                            p_d,
                            p_e,
                            a_ms_ks_lengths,
                            a_ms_ks_strides,
                            b_ns_ks_lengths,
                            b_ns_ks_strides,
                            d_ms_ns_lengths,
                            d_ms_ns_strides,
                            e_ms_ns_lengths,
                            e_ms_ns_strides,
                            a_element_op,
                            b_element_op,
                            cde_element_op};
        }

        static auto MakeInvoker()
        {
            return Invoker{};
        }

        std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
        {
            return std::make_unique<Invoker>(Invoker{});
        }

        std::unique_ptr<BaseArgument> MakeArgumentPointer(
            const void*                                         p_a,
            const void*                                         p_b,
            std::array<const void*, NumDTensor>                 p_ds,
            void*                                               p_e,
            std::vector<index_t> const&                         a_ms_ks_lengths,
            std::vector<index_t> const&                         a_ms_ks_strides,
            std::vector<index_t> const&                         b_ns_ks_lengths,
            std::vector<index_t> const&                         b_ns_ks_strides,
            std::array<std::vector<index_t>, NumDTensor> const& ds_ms_ns_lengths,
            std::array<std::vector<index_t>, NumDTensor> const& ds_ms_ns_strides,
            std::vector<index_t> const&                         e_ms_ns_lengths,
            std::vector<index_t> const&                         e_ms_ns_strides,
            AElementwiseOperation                               a_element_op,
            BElementwiseOperation                               b_element_op,
            CDEElementwiseOperation                             cde_element_op) override
        {
            return std::make_unique<Argument>(Argument{p_a,
                                                       p_b,
                                                       p_ds,
                                                       p_e,
                                                       a_ms_ks_lengths,
                                                       a_ms_ks_strides,
                                                       b_ns_ks_lengths,
                                                       b_ns_ks_strides,
                                                       ds_ms_ns_lengths,
                                                       ds_ms_ns_strides,
                                                       e_ms_ns_lengths,
                                                       e_ms_ns_strides,
                                                       a_element_op,
                                                       b_element_op,
                                                       cde_element_op});
        }

        std::string GetTypeString() const override
        {
            auto str = std::stringstream();

            // clang-format off
            str << "ReferenceContraction_M2_N2_K2"
                << std::endl;
            // clang-format on

            return str.str();
        }
    };

    // Partial specialize for reference contraction
    template <ck::index_t NumDimsM,
              ck::index_t NumDimsN,
              ck::index_t NumDimsK,
              typename ADataType,
              typename BDataType,
              typename AccDataType,
              typename DsDataType,
              typename EDataType,
              typename AElementwiseOperation,
              typename BElementwiseOperation,
              typename CDEElementwiseOperation,
              typename ComputeDataType>
    struct MetaTraits<ReferenceContraction_M2_N2_K2<NumDimsM,
                                                    NumDimsN,
                                                    NumDimsK,
                                                    ADataType,
                                                    BDataType,
                                                    AccDataType,
                                                    DsDataType,
                                                    EDataType,
                                                    AElementwiseOperation,
                                                    BElementwiseOperation,
                                                    CDEElementwiseOperation,
                                                    ComputeDataType>>
        : public MetaTraits<
              ck::tensor_operation::device::DeviceContractionMultipleD<NumDimsM,
                                                                       NumDimsN,
                                                                       NumDimsK,
                                                                       ADataType,
                                                                       BDataType,
                                                                       DsDataType,
                                                                       EDataType,
                                                                       AElementwiseOperation,
                                                                       BElementwiseOperation,
                                                                       CDEElementwiseOperation,
                                                                       ComputeDataType>>
    {
    };

    template <ck::index_t NumDimM,
              ck::index_t NumDimN,
              ck::index_t NumDimK,
              typename ADataType,
              typename BDataType,
              typename AccDataType,
              typename DsDataType,
              typename EDataType,
              typename AElementwiseOperation,
              typename BElementwiseOperation,
              typename CDEElementwiseOperation,
              typename ComputeDataType = ADataType>
    auto enumerateReferenceSolutions()
    {
        using ReferenceOp = ReferenceContraction_M2_N2_K2<NumDimM,
                                                          NumDimN,
                                                          NumDimK,
                                                          ADataType,
                                                          BDataType,
                                                          AccDataType,
                                                          DsDataType,
                                                          EDataType,
                                                          AElementwiseOperation,
                                                          BElementwiseOperation,
                                                          CDEElementwiseOperation,
                                                          ComputeDataType>;

        auto solution = std::make_unique<ContractionSolutionImpl<ReferenceOp>>(
            std::make_unique<ReferenceOp>());
        auto result = std::vector<std::unique_ptr<ContractionSolution>>();
        result.push_back(std::move(solution));
        return result;
    }

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_CPU_REFERENCE_IMPL_HPP
