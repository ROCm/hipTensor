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

// std includes
#include <iomanip>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

// CK includes
#include <ck.hpp>
#include <contraction_bilinear.hpp>
#include <contraction_scale.hpp>
#include <device_contraction_multiple_d.hpp>
#include <element_wise_operation.hpp>
#include <tensor_layout.hpp>

// HipTensor includes
#include "hiptensor_contraction_ck.hpp"
#include "hiptensor_types.hpp"
#include "internal/hiptensor_utility.hpp"

using F32 = float;
using F64 = double;

template <typename T>
struct MetaTraits;

using ck::index_t;
template <index_t NumDimsM,
          index_t NumDimsN,
          index_t NumDimsK,
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
    constexpr static index_t DimsM = NumDimsM;
    constexpr static index_t DimsN = NumDimsN;
    constexpr static index_t DimsK = NumDimsK;
    using ADataT                   = ADataType;
    using BDataT                   = BDataType;
    using DDataT                   = DsDataType;
    using EDataT                   = EDataType;
    using AOp                      = AElementwiseOperation;
    using BOp                      = BElementwiseOperation;
    using CDEOp                    = ck::tensor_operation::element_wise::Bilinear;
};

template <index_t NumDimsM,
          index_t NumDimsN,
          index_t NumDimsK,
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
    constexpr static index_t DimsM = NumDimsM;
    constexpr static index_t DimsN = NumDimsN;
    constexpr static index_t DimsK = NumDimsK;
    using ADataT                   = ADataType;
    using BDataT                   = BDataType;
    using EDataT                   = EDataType;
    using AOp                      = AElementwiseOperation;
    using BOp                      = BElementwiseOperation;
    using CDEOp                    = ck::tensor_operation::element_wise::Scale;
};

struct KernelLauncher
{
    // Override for BilinearContraction
    template <
        typename DeviceOp,
        typename std::enable_if_t<std::is_same_v<typename MetaTraits<DeviceOp>::CDEOp,
                                                 ck::tensor_operation::element_wise::Bilinear>,
                                  void*>
        = nullptr>
    KernelLauncher(DeviceOp*                                deviceOp,
                   void const*                              alpha,
                   void const*                              A,
                   void const*                              B,
                   void const*                              beta,
                   void const*                              D,
                   void*                                    E,
                   std::vector<index_t> const&              a_ms_ns_lengths,
                   std::vector<index_t> const&              a_ms_ks_strides,
                   std::vector<index_t> const&              b_ns_ks_lengths,
                   std::vector<index_t> const&              b_ns_ks_strides,
                   std::vector<std::vector<index_t>> const& ds_ms_ns_lengths,
                   std::vector<std::vector<index_t>> const& ds_ms_ns_strides,
                   std::vector<index_t> const&              e_ms_ns_lengths,
                   std::vector<index_t> const&              e_ms_ns_strides)
        : a_ms_ns_lengths_(a_ms_ns_lengths)
        , a_ms_ks_strides_(a_ms_ks_strides)
        , b_ns_ks_lengths_(b_ns_ks_lengths)
        , b_ns_ks_strides_(b_ns_ks_strides)
        , ds_ms_ns_lengths_(ds_ms_ns_lengths)
        , ds_ms_ns_strides_(ds_ms_ns_strides)
        , e_ms_ns_lengths_(e_ms_ns_lengths)
        , e_ms_ns_strides_(e_ms_ns_strides)
    {
        using Traits = MetaTraits<DeviceOp>;

        // Initialize the argument pointer
        mArgPtr = std::move(deviceOp->MakeArgumentPointer(
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
            typename Traits::CDEOp{*(F32*)alpha, *(F32*)beta}));

        // Initialize the invoker
        mInvokerPtr = std::move(deviceOp->MakeInvokerPointer());

        // Get the kernel name
        mKernelName = deviceOp->GetTypeString();

        // Fill problem metrics
        mM = std::accumulate(e_ms_ns_lengths.begin(),
                             e_ms_ns_lengths.begin() + Traits::DimsM,
                             ck::index_t{1},
                             std::multiplies<ck::index_t>{});

        mN = std::accumulate(e_ms_ns_lengths.begin() + Traits::DimsM,
                             e_ms_ns_lengths.begin() + Traits::DimsM + Traits::DimsN,
                             ck::index_t{1},
                             std::multiplies<ck::index_t>{});

        mK = std::accumulate(e_ms_ns_lengths.begin() + Traits::DimsM,
                             e_ms_ns_lengths.begin() + Traits::DimsM + Traits::DimsK,
                             ck::index_t{1},
                             std::multiplies<ck::index_t>{});

        // Byte count
        mBytes = sizeof(typename Traits::ADataT) * mM * mK
                 + sizeof(typename Traits::BDataT) * mK * mN
                 + sizeof(typename Traits::DDataT) * mM * mN
                 + sizeof(typename Traits::EDataT) * mM * mN;

        mValid = deviceOp->IsSupportedArgument(mArgPtr.get());
    }

    template <typename DeviceOp,
              typename std::enable_if_t<std::is_same_v<typename MetaTraits<DeviceOp>::CDEOp,
                                                       ck::tensor_operation::element_wise::Scale>,
                                        void*>
              = nullptr>
    KernelLauncher(DeviceOp*                   deviceOp,
                   void const*                 alpha,
                   void const*                 A,
                   void const*                 B,
                   void*                       E,
                   std::vector<index_t> const& a_ms_ns_lengths,
                   std::vector<index_t> const& a_ms_ks_strides,
                   std::vector<index_t> const& b_ns_ks_lengths,
                   std::vector<index_t> const& b_ns_ks_strides,
                   std::vector<index_t> const& e_ms_ns_lengths,
                   std::vector<index_t> const& e_ms_ns_strides)
    {
        using Traits = MetaTraits<DeviceOp>;

        // Initialize the argument pointer
        mArgPtr = std::move(deviceOp->MakeArgumentPointer(A,
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
                                                          typename Traits::CDEOp{*(F32*)alpha}));

        // Initialize the invoker
        mInvokerPtr = std::move(deviceOp->MakeInvokerPointer());

        // Get the kernel name
        mKernelName = deviceOp->GetTypeString();

        // Fill problem metrics
        mM = std::accumulate(e_ms_ns_lengths.begin(),
                             e_ms_ns_lengths.begin() + Traits::DimsM,
                             ck::index_t{1},
                             std::multiplies<ck::index_t>{});

        mN = std::accumulate(e_ms_ns_lengths.begin() + Traits::DimsM,
                             e_ms_ns_lengths.begin() + Traits::DimsM + Traits::DimsN,
                             ck::index_t{1},
                             std::multiplies<ck::index_t>{});

        mK = std::accumulate(a_ms_ns_lengths.begin() + Traits::DimsM,
                             a_ms_ns_lengths.begin() + Traits::DimsM + Traits::DimsK,
                             ck::index_t{1},
                             std::multiplies<ck::index_t>{});

        // Byte count
        mBytes = sizeof(typename Traits::ADataT) * mM * mK
                 + sizeof(typename Traits::BDataT) * mK * mN
                 + sizeof(typename Traits::EDataT) * mM * mN;

        mValid = deviceOp->IsSupportedArgument(mArgPtr.get());
    }

    float operator()(StreamConfig const& streamConfig = StreamConfig{nullptr, true})
    {
        if(!mValid)
        {
#if !NDEBUG
            std::cout << op->mKernelName() << " does not support this problem" << std::endl;
#endif // !NDEBUG
            return -1.0f;
        }

        return mInvokerPtr->Run(mArgPtr.get(), streamConfig);
    }

    bool isValid() const
    {
        return mValid;
    }

    std::vector<index_t>              a_ms_ns_lengths_;
    std::vector<index_t>              a_ms_ks_strides_;
    std::vector<index_t>              b_ns_ks_lengths_;
    std::vector<index_t>              b_ns_ks_strides_;
    std::vector<std::vector<index_t>> ds_ms_ns_lengths_;
    std::vector<std::vector<index_t>> ds_ms_ns_strides_;
    std::vector<index_t>              e_ms_ns_lengths_;
    std::vector<index_t>              e_ms_ns_strides_;

    index_t mM, mN, mK;
    index_t mBytes;
    bool    mValid;

    std::unique_ptr<ck::tensor_operation::device::BaseArgument> mArgPtr;
    std::unique_ptr<ck::tensor_operation::device::BaseInvoker>  mInvokerPtr;
    std::string                                                 mKernelName;
};

hiptensorStatus_t hiptensorCKContraction(const hiptensorHandle_t*          handle,
                                         const hiptensorContractionPlan_t* plan,
                                         hiptensorContractionMetrics_t*    ht_contract_metrics,
                                         const void*                       alpha,
                                         const void*                       A,
                                         const void*                       B,
                                         const void*                       beta,
                                         const void*                       C,
                                         void*                             D,
                                         void*                             workspace,
                                         uint64_t                          workspaceSize,
                                         hipStream_t                       stream)
{
    if(!handle || !ht_contract_metrics || !A || !B || !D)
        return HIPTENSOR_STATUS_NOT_INITIALIZED;

    memset(ht_contract_metrics, 0, sizeof(hiptensorContractionMetrics_t));

    auto a_ms_ns_lengths
        = std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].lens.begin(),
                                   plan->ht_plan_desc.ht_contract_attr_desc[0].lens.end());

    auto a_ms_ks_strides
        = std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].strides.begin(),
                                   plan->ht_plan_desc.ht_contract_attr_desc[0].strides.end());

    auto b_ns_ks_lengths
        = std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].lens.begin(),
                                   plan->ht_plan_desc.ht_contract_attr_desc[1].lens.end());

    auto b_ns_ks_strides
        = std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].strides.begin(),
                                   plan->ht_plan_desc.ht_contract_attr_desc[1].strides.end());

    auto e_ms_ns_lengths
        = std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(),
                                   plan->ht_plan_desc.ht_contract_attr_desc[2].lens.end());

    auto e_ms_ns_strides
        = std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].strides.begin(),
                                   plan->ht_plan_desc.ht_contract_attr_desc[2].strides.end());

#if !NDEBUG
    std::cout << "Tensor A lengths: ";
    hiptensorPrintVectorElements<ck::index_t>(a_ms_ns_lengths);
    std::cout << ", strides: ";
    hiptensorPrintVectorElements<ck::index_t>(a_ms_ks_strides);
    std::cout << ", size: " << plan->ht_plan_desc.ht_contract_attr_desc[0].tensor_size << std::endl;

    std::cout << "Tensor B lengths: ";
    hiptensorPrintVectorElements<ck::index_t>(b_ns_ks_lengths);
    std::cout << ", strides: ";
    hiptensorPrintVectorElements<ck::index_t>(b_ns_ks_strides);
    std::cout << ", size: " << plan->ht_plan_desc.ht_contract_attr_desc[1].tensor_size << std::endl;

    std::cout << "Tensor C lengths: ";
    hiptensorPrintVectorElements<ck::index_t>(e_ms_ns_lengths);
    std::cout << ", strides: ";
    hiptensorPrintVectorElements<ck::index_t>(e_ms_ns_strides);
    std::cout << ", size: " << plan->ht_plan_desc.ht_contract_attr_desc[2].tensor_size << std::endl;
#endif // !NDEBUG

    void* output;

    hip_check_error(hipMalloc(static_cast<void**>(&output),
                              plan->ht_plan_desc.ht_contract_attr_desc[2].tensor_size));

    hip_check_error(hipMemset(output, 0, plan->ht_plan_desc.ht_contract_attr_desc[2].tensor_size));

    std::vector<KernelLauncher> solutions;

    if(plan->ht_plan_desc.ht_contract_op == HIPTENSOR_CONTRACTION_BILINEAR)
    {
        // Use this generic lambda to initialize the bilinear kernels.
        auto initBilinearSolutions = [&](auto const& v) {
            for(auto& opPtr : v)
            {
                solutions.push_back(
                    KernelLauncher(opPtr.get(),
                                   alpha,
                                   A,
                                   B,
                                   beta,
                                   C,
                                   output,
                                   a_ms_ns_lengths,
                                   a_ms_ks_strides,
                                   b_ns_ks_lengths,
                                   b_ns_ks_strides,
                                   std::vector<std::vector<ck::index_t>>{e_ms_ns_lengths},
                                   std::vector<std::vector<ck::index_t>>{e_ms_ns_strides},
                                   e_ms_ns_lengths,
                                   e_ms_ns_strides));
            }
        };

        auto ADataType = plan->ht_plan_desc.ht_contract_attr_desc[0].ht_type;
        auto BDataType = plan->ht_plan_desc.ht_contract_attr_desc[1].ht_type;
        auto CDataType = plan->ht_plan_desc.ht_contract_attr_desc[2].ht_type;
        auto DDataType = plan->ht_plan_desc.ht_contract_attr_desc[3].ht_type;

        if(ADataType == HIPTENSOR_R_32F && BDataType == HIPTENSOR_R_32F &&
           CDataType == HIPTENSOR_R_32F && DDataType == HIPTENSOR_R_32F)
        {
            using ContractionBilinearOp = ck::tensor_operation::device::DeviceContractionMultipleD<
                2,
                2,
                2,
                F32,
                F32,
                ck::Tuple<F32>,
                F32,
                ck::tensor_operation::element_wise::PassThrough,
                ck::tensor_operation::element_wise::PassThrough,
                ck::tensor_operation::element_wise::Bilinear>;

            initBilinearSolutions(
                ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
                    ContractionBilinearOp>::GetInstances());
        }
        else if(ADataType == HIPTENSOR_R_64F && BDataType == HIPTENSOR_R_64F &&
                CDataType == HIPTENSOR_R_64F && DDataType == HIPTENSOR_R_64F)
        {
            using ContractionBilinearOp =
                ck::tensor_operation::device::DeviceContractionMultipleD<
                2, 2, 2, F64, F64, ck::Tuple<F64>, F64,
                ck::tensor_operation::element_wise::PassThrough,
                ck::tensor_operation::element_wise::PassThrough,
                ck::tensor_operation::element_wise::Bilinear>;

            initBilinearSolutions(
                ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
                ContractionBilinearOp>::GetInstances());
        }
    }
    else if(plan->ht_plan_desc.ht_contract_op == HIPTENSOR_CONTRACTION_SCALE)
    {
        // Use this generic lambda to initialize the bilinear kernels.
        auto initScaleSolutions = [&](auto const& v) {
            for(auto& opPtr : v)
            {
                solutions.push_back(KernelLauncher(opPtr.get(),
                                                   alpha,
                                                   A,
                                                   B,
                                                   D,
                                                   a_ms_ns_lengths,
                                                   a_ms_ks_strides,
                                                   b_ns_ks_lengths,
                                                   b_ns_ks_strides,
                                                   e_ms_ns_lengths,
                                                   e_ms_ns_strides));
            }
        };

        auto ADataType = plan->ht_plan_desc.ht_contract_attr_desc[0].ht_type;
        auto BDataType = plan->ht_plan_desc.ht_contract_attr_desc[1].ht_type;
        auto CDataType = plan->ht_plan_desc.ht_contract_attr_desc[2].ht_type;
        auto DDataType = plan->ht_plan_desc.ht_contract_attr_desc[3].ht_type;

        if(ADataType == HIPTENSOR_R_32F && BDataType == HIPTENSOR_R_32F &&
           CDataType == HIPTENSOR_R_32F && DDataType == HIPTENSOR_R_32F)
        {
            using ContractionScaleOp = ck::tensor_operation::device::DeviceContractionMultipleD<
                2,
                2,
                2,
                F32,
                F32,
                ck::Tuple<>,
                F32,
                ck::tensor_operation::element_wise::PassThrough,
                ck::tensor_operation::element_wise::PassThrough,
                ck::tensor_operation::element_wise::Scale>;

            initScaleSolutions(
                ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
                    ContractionScaleOp>::GetInstances());
        }
        else if(ADataType == HIPTENSOR_R_64F && BDataType == HIPTENSOR_R_64F &&
                CDataType == HIPTENSOR_R_64F && DDataType == HIPTENSOR_R_64F)
        {
            using ContractionScaleOp =
                ck::tensor_operation::device::DeviceContractionMultipleD<
                2, 2, 2, F64, F64, ck::Tuple<>, F64,
                ck::tensor_operation::element_wise::PassThrough,
                ck::tensor_operation::element_wise::PassThrough,
                ck::tensor_operation::element_wise::Scale>;

            initScaleSolutions(
                ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
                ContractionScaleOp>::GetInstances());
        }
    }

    /// Dispatching end

    // Now we can launch the kernels and get the metrics.
    std::cout << "Run all instances and do timing" << std::endl;

    std::string                   best_op_name;
    bool                          found     = false;
    hiptensorContractionMetrics_t bestFound = {0, 0, 0, ""};

    for(auto& solution : solutions)
    {

        if(solution.isValid())
        {
            auto flops = std::size_t(2) * solution.mM * solution.mN * solution.mK;
            auto bytes = solution.mBytes;

            auto time = solution();

            hiptensorContractionMetrics_t metrics = {
                time, // avg time
                static_cast<float>(flops) / static_cast<float>(1.E9) / time, // tflops
                static_cast<float>(solution.mBytes) / static_cast<float>(1.E6) / time, //
                solution.mKernelName // name
            };

            if(metrics.tflops > bestFound.tflops)
            {
                found     = true;
                bestFound = metrics;
            }
        }
    }

    if(found)
    {
        *ht_contract_metrics = bestFound;
    }

    if(output)
    {
        hip_check_error(hipMemcpy(D,
                                  output,
                                  plan->ht_plan_desc.ht_contract_attr_desc[2].tensor_size,
                                  hipMemcpyDeviceToDevice));

        hip_check_error(hipFree(output));
    }

    return HIPTENSOR_STATUS_SUCCESS;
}
