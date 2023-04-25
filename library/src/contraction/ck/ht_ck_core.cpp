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
#include <vector>

// CK includes
#include <ck.hpp>
#include <contraction_bilinear.hpp>
#include <contraction_scale.hpp>
#include <device_contraction_multiple_d.hpp>
#include <element_wise_operation.hpp>
#include <tensor_layout.hpp>

// HipTensor includes
#include "ht_ck_core.hpp"
#include "ht_types.hpp"
#include "ht_utility.hpp"

using F32       = float;
using ADataType = F32;
using BDataType = F32;
using CDataType = F32;
using DDataType = F32;

static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 2;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Scale       = ck::tensor_operation::element_wise::Scale;
using Bilinear    = ck::tensor_operation::element_wise::Bilinear;

using AElementOp           = PassThrough;
using BElementOp           = PassThrough;
using CDEScaleElementOp    = Scale;
using CDEBilinearElementOp = Bilinear;

using ContractionScaleOp = ck::tensor_operation::device::DeviceContractionMultipleD<
    NumDimM,
    NumDimN,
    NumDimK,
    ADataType,
    BDataType,
    ck::Tuple<>,
    DDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::Scale>;

using ContractionBilinearOp = ck::tensor_operation::device::DeviceContractionMultipleD<
    NumDimM,
    NumDimN,
    NumDimK,
    ADataType,
    BDataType,
    ck::Tuple<CDataType>,
    DDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::Bilinear>;

hiptensorStatus_t
    hiptensorFillCKContractionMetrics(const hiptensorContractionPlan_t*     plan,
                                      hiptensorContractionMetrics_t*        ht_contract_metrics,
                                      const hiptensorContractionOperation_t contractionOp)
{
    ck::index_t M
        = std::accumulate(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(),
                          plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin() + NumDimM,
                          ck::index_t{1},
                          std::multiplies<ck::index_t>{});

    ck::index_t N = std::accumulate(
        plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin() + NumDimM,
        plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin() + NumDimM + NumDimN,
        ck::index_t{1},
        std::multiplies<ck::index_t>{});

    ck::index_t K = std::accumulate(
        plan->ht_plan_desc.ht_contract_attr_desc[0].lens.begin() + NumDimM,
        plan->ht_plan_desc.ht_contract_attr_desc[0].lens.begin() + NumDimM + NumDimK,
        ck::index_t{1},
        std::multiplies<ck::index_t>{});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype;

    if(contractionOp == hiptensor_CONTRACTION_BILINEAR)
    {
        num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N
                    + sizeof(CDataType) * M * N + sizeof(DDataType) * M * N;
    }
    else if(contractionOp == hiptensor_CONTRACTION_SCALE)
    {
        num_btype
            = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(DDataType) * M * N;
    }
    else
    {
        std::cout << "Input Contraction operation not supported by CK" << std::endl;
        return HIPTENSOR_STATUS_CK_ERROR;
    }

    ht_contract_metrics->tflops = static_cast<float>(flop) / 1.E9 / ht_contract_metrics->avg_time;
    ht_contract_metrics->transfer_speed = num_btype / 1.E6 / ht_contract_metrics->avg_time;
    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorCKScaleContraction(const hiptensorHandle_t*          handle,
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

    std::string best_op_name;
    bool        found           = false;
    int         best_op_id      = -1;
    float       best_ave_time   = 0;
    float       best_tflops     = 0;
    float       best_gb_per_sec = 0;

    memset(ht_contract_metrics, 0, sizeof(hiptensorContractionMetrics_t));

#ifdef HT_DEBUG_MODE
    std::cout << "Tensor A lengths: ";
    hiptensorPrintVectorElements<ck::index_t>(
        std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].lens.begin(),
                                 plan->ht_plan_desc.ht_contract_attr_desc[0].lens.end()));
    std::cout << ", strides: ";
    hiptensorPrintVectorElements<ck::index_t>(
        std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].strides.begin(),
                                 plan->ht_plan_desc.ht_contract_attr_desc[0].strides.end()));
    std::cout << ", size: " << plan->ht_plan_desc.ht_contract_attr_desc[0].tensor_size << std::endl;

    std::cout << "Tensor B lengths: ";
    hiptensorPrintVectorElements<ck::index_t>(
        std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].lens.begin(),
                                 plan->ht_plan_desc.ht_contract_attr_desc[1].lens.end()));
    std::cout << ", strides: ";
    hiptensorPrintVectorElements<ck::index_t>(
        std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].strides.begin(),
                                 plan->ht_plan_desc.ht_contract_attr_desc[1].strides.end()));
    std::cout << ", size: " << plan->ht_plan_desc.ht_contract_attr_desc[1].tensor_size << std::endl;

    std::cout << "Tensor C lengths: ";
    hiptensorPrintVectorElements<ck::index_t>(
        std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(),
                                 plan->ht_plan_desc.ht_contract_attr_desc[2].lens.end()));
    std::cout << ", strides: ";
    hiptensorPrintVectorElements<ck::index_t>(
        std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].strides.begin(),
                                 plan->ht_plan_desc.ht_contract_attr_desc[2].strides.end()));
    std::cout << ", size: " << plan->ht_plan_desc.ht_contract_attr_desc[2].tensor_size << std::endl;
#endif

    auto contraction_scale = [&](auto& op_layout) {
        if(!op_layout)
            return HIPTENSOR_STATUS_NOT_INITIALIZED;

        using ContractionInstance = decltype(op_layout);
        ContractionInstance op    = std::move(op_layout);

        const auto a_element_op = AElementOp{};
        const auto b_element_op = BElementOp{};

        const auto cde_element_op = CDEScaleElementOp{*(F32*)alpha};
        auto       argument_ptr   = op->MakeArgumentPointer(
            A,
            B,
            std::array<const void*, 0>{},
            D,
            std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].lens.begin(),
                                     plan->ht_plan_desc.ht_contract_attr_desc[0].lens.end()),
            std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].strides.begin(),
                                     plan->ht_plan_desc.ht_contract_attr_desc[0].strides.end()),
            std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].lens.begin(),
                                     plan->ht_plan_desc.ht_contract_attr_desc[1].lens.end()),
            std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].strides.begin(),
                                     plan->ht_plan_desc.ht_contract_attr_desc[1].strides.end()),
            std::array<std::vector<ck::index_t>, 0>{},
            std::array<std::vector<ck::index_t>, 0>{},
            std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(),
                                     plan->ht_plan_desc.ht_contract_attr_desc[2].lens.end()),
            std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].strides.begin(),
                                     plan->ht_plan_desc.ht_contract_attr_desc[2].strides.end()),
            a_element_op,
            b_element_op,
            cde_element_op);

        auto        invoker_ptr = op->MakeInvokerPointer();
        std::string op_name     = op->GetTypeString();

        if(!op->IsSupportedArgument(argument_ptr.get()))
        {
#ifdef HT_DEBUG_MODE
            std::cout << op->GetTypeString() << " does not support this problem" << std::endl;
#endif
            return HIPTENSOR_STATUS_CK_ERROR;
        }

        ht_contract_metrics->avg_time
            = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});
        hiptensorFillCKContractionMetrics(
            plan, ht_contract_metrics, plan->ht_plan_desc.ht_contract_op);
        return HIPTENSOR_STATUS_SUCCESS;
    };

    const auto op_scale_ptrs
        = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
            ContractionScaleOp>::GetInstances();

    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_scale_ptrs.size(); ++i)
    {
        auto& op_ptr = op_scale_ptrs[i];
        contraction_scale(op_ptr);
        if(ht_contract_metrics->tflops > best_tflops)
        {
            found           = true;
            best_op_id      = i;
            best_op_name    = op_ptr->GetTypeString();
            best_tflops     = ht_contract_metrics->tflops;
            best_ave_time   = ht_contract_metrics->avg_time;
            best_gb_per_sec = ht_contract_metrics->transfer_speed;
        }
    }

#ifdef HT_DEBUG_MODE
    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;
#endif

    auto& contract_op_ptr            = op_scale_ptrs[best_op_id];
    ht_contract_metrics->ht_instance = contract_op_ptr->GetTypeString();
    contraction_scale(contract_op_ptr);
    return HIPTENSOR_STATUS_SUCCESS;
}
hiptensorStatus_t hiptensorCKBilinearContraction(const hiptensorHandle_t*          handle,
                                                 const hiptensorContractionPlan_t* plan,
                                                 hiptensorContractionMetrics_t* ht_contract_metrics,
                                                 const void*                    alpha,
                                                 const void*                    A,
                                                 const void*                    B,
                                                 const void*                    beta,
                                                 const void*                    C,
                                                 void*                          D,
                                                 void*                          workspace,
                                                 uint64_t                       workspaceSize,
                                                 hipStream_t                    stream)
{
    if(!handle || !ht_contract_metrics || !A || !B || !D)
        return HIPTENSOR_STATUS_NOT_INITIALIZED;

    std::string best_op_name;
    bool        found           = false;
    int         best_op_id      = -1;
    float       best_ave_time   = 0;
    float       best_tflops     = 0;
    float       best_gb_per_sec = 0;
    void*       output;

    memset(ht_contract_metrics, 0, sizeof(hiptensorContractionMetrics_t));

#ifdef HT_DEBUG_MODE
    std::cout << "Tensor A lengths: ";
    hiptensorPrintVectorElements<ck::index_t>(
        std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].lens.begin(),
                                 plan->ht_plan_desc.ht_contract_attr_desc[0].lens.end()));
    std::cout << ", strides: ";
    hiptensorPrintVectorElements<ck::index_t>(
        std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].strides.begin(),
                                 plan->ht_plan_desc.ht_contract_attr_desc[0].strides.end()));
    std::cout << ", size: " << plan->ht_plan_desc.ht_contract_attr_desc[0].tensor_size << std::endl;

    std::cout << "Tensor B lengths: ";
    hiptensorPrintVectorElements<ck::index_t>(
        std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].lens.begin(),
                                 plan->ht_plan_desc.ht_contract_attr_desc[1].lens.end()));
    std::cout << ", strides: ";
    hiptensorPrintVectorElements<ck::index_t>(
        std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].strides.begin(),
                                 plan->ht_plan_desc.ht_contract_attr_desc[1].strides.end()));
    std::cout << ", size: " << plan->ht_plan_desc.ht_contract_attr_desc[1].tensor_size << std::endl;

    std::cout << "Tensor C lengths: ";
    hiptensorPrintVectorElements<ck::index_t>(
        std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(),
                                 plan->ht_plan_desc.ht_contract_attr_desc[2].lens.end()));
    std::cout << ", strides: ";
    hiptensorPrintVectorElements<ck::index_t>(
        std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].strides.begin(),
                                 plan->ht_plan_desc.ht_contract_attr_desc[2].strides.end()));
    std::cout << ", size: " << plan->ht_plan_desc.ht_contract_attr_desc[2].tensor_size << std::endl;
#endif

    hip_check_error(hipMalloc(static_cast<void**>(&output),
                              plan->ht_plan_desc.ht_contract_attr_desc[2].tensor_size));
    hip_check_error(hipMemset(output, 0, plan->ht_plan_desc.ht_contract_attr_desc[2].tensor_size));

    auto contraction_bilinear = [&](auto& op_layout) {
        if(!op_layout)
            return HIPTENSOR_STATUS_NOT_INITIALIZED;

        using ContractionInstance = decltype(op_layout);
        ContractionInstance op    = std::move(op_layout);

        const auto a_element_op = AElementOp{};
        const auto b_element_op = BElementOp{};

        const auto cde_element_op = CDEBilinearElementOp{*(F32*)alpha, *(F32*)beta};

        auto argument_ptr = op->MakeArgumentPointer(
            A,
            B,
            std::array<const void*, 1>{C},
            output,
            std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].lens.begin(),
                                     plan->ht_plan_desc.ht_contract_attr_desc[0].lens.end()),
            std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].strides.begin(),
                                     plan->ht_plan_desc.ht_contract_attr_desc[0].strides.end()),
            std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].lens.begin(),
                                     plan->ht_plan_desc.ht_contract_attr_desc[1].lens.end()),
            std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].strides.begin(),
                                     plan->ht_plan_desc.ht_contract_attr_desc[1].strides.end()),
            std::array<std::vector<ck::index_t>, 1>{
                std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(),
                                         plan->ht_plan_desc.ht_contract_attr_desc[2].lens.end())},
            std::array<std::vector<ck::index_t>, 1>{std::vector<ck::index_t>(
                plan->ht_plan_desc.ht_contract_attr_desc[2].strides.begin(),
                plan->ht_plan_desc.ht_contract_attr_desc[2].strides.end())},
            std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(),
                                     plan->ht_plan_desc.ht_contract_attr_desc[2].lens.end()),
            std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].strides.begin(),
                                     plan->ht_plan_desc.ht_contract_attr_desc[2].strides.end()),
            a_element_op,
            b_element_op,
            cde_element_op);

        auto        invoker_ptr = op->MakeInvokerPointer();
        std::string op_name     = op->GetTypeString();

        if(!op->IsSupportedArgument(argument_ptr.get()))
        {
#ifdef HT_DEBUG_MODE
            std::cout << op->GetTypeString() << " does not support this problem" << std::endl;
#endif
            return HIPTENSOR_STATUS_CK_ERROR;
        }

        ht_contract_metrics->avg_time
            = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});
        hiptensorFillCKContractionMetrics(
            plan, ht_contract_metrics, plan->ht_plan_desc.ht_contract_op);
        return HIPTENSOR_STATUS_SUCCESS;
    };

    const auto op_bilinear_ptrs
        = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
            ContractionBilinearOp>::GetInstances();

#ifdef HT_DEBUG_MODE
    std::cout << "Run all instances and do timing" << std::endl;
#endif

    for(int i = 0; i < op_bilinear_ptrs.size(); ++i)
    {
        auto& op_ptr = op_bilinear_ptrs[i];
        contraction_bilinear(op_ptr);
        if(ht_contract_metrics->tflops > best_tflops)
        {
            found           = true;
            best_op_id      = i;
            best_op_name    = op_ptr->GetTypeString();
            best_tflops     = ht_contract_metrics->tflops;
            best_ave_time   = ht_contract_metrics->avg_time;
            best_gb_per_sec = ht_contract_metrics->transfer_speed;
        }
    }
#ifdef HT_DEBUG_MODE
    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;
#endif
    auto& contract_op_ptr            = op_bilinear_ptrs[best_op_id];
    ht_contract_metrics->ht_instance = contract_op_ptr->GetTypeString();
    contraction_bilinear(contract_op_ptr);

    hip_check_error(hipMemcpy(D,
                              output,
                              plan->ht_plan_desc.ht_contract_attr_desc[2].tensor_size,
                              hipMemcpyDeviceToDevice));

    if(output)
        hip_check_error(hipFree(output));
    return HIPTENSOR_STATUS_SUCCESS;
}
