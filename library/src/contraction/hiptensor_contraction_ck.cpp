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
#include "contraction_meta_traits.hpp"
#include "contraction_solution.hpp"
#include "contraction_solution_registry.hpp"
#include "hiptensor_contraction_ck.hpp"
#include "hiptensor_types.hpp"
#include "internal/hiptensor_utility.hpp"

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
    if(handle == nullptr || plan == nullptr || ht_contract_metrics == nullptr || alpha == nullptr
       || A == nullptr || B == nullptr
       || ((beta == nullptr || C == nullptr)
           && plan->ht_plan_desc.ht_contract_op == (int)hiptensor::ContractionOpId_t::BILINEAR)
       || D == nullptr)
    {
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    memset(ht_contract_metrics, 0, sizeof(hiptensorContractionMetrics_t));

    // NOTE: Here, ck::index_t is int, NOT same as std::index_t = long uint
    // Therefore the conversion to ck::index_t is required.
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
        = std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[3].lens.begin(),
                                   plan->ht_plan_desc.ht_contract_attr_desc[3].lens.end());

    auto e_ms_ns_strides
        = std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[3].strides.begin(),
                                   plan->ht_plan_desc.ht_contract_attr_desc[3].strides.end());

#if !NDEBUG
    std::cout << "Tensor A lengths: ";
    hiptensorPrintVectorElements(a_ms_ns_lengths);
    std::cout << ", strides: ";
    hiptensorPrintVectorElements(a_ms_ks_strides);
    std::cout << ", size: " << plan->ht_plan_desc.ht_contract_attr_desc[0].tensor_size << std::endl;

    std::cout << "Tensor B lengths: ";
    hiptensorPrintVectorElements(b_ns_ks_lengths);
    std::cout << ", strides: ";
    hiptensorPrintVectorElements(b_ns_ks_strides);
    std::cout << ", size: " << plan->ht_plan_desc.ht_contract_attr_desc[1].tensor_size << std::endl;

    std::cout << "Tensor C lengths: ";
    hiptensorPrintVectorElements(e_ms_ns_lengths);
    std::cout << ", strides: ";
    hiptensorPrintVectorElements(e_ms_ns_strides);
    std::cout << ", size: " << plan->ht_plan_desc.ht_contract_attr_desc[2].tensor_size << std::endl;
#endif // !NDEBUG

    hiptensor::ContractionSolutionRegistry registry;

    auto ADataType = plan->ht_plan_desc.ht_contract_attr_desc[0].ht_type;
    auto BDataType = plan->ht_plan_desc.ht_contract_attr_desc[1].ht_type;
    auto CDataType = plan->ht_plan_desc.ht_contract_attr_desc[2].ht_type;
    auto DDataType = plan->ht_plan_desc.ht_contract_attr_desc[3].ht_type;

    if(plan->ht_plan_desc.ht_contract_op == (int)hiptensor::ContractionOpId_t::BILINEAR)
    {
        if(ADataType == HIP_R_32F && BDataType == HIP_R_32F && CDataType == HIP_R_32F
           && DDataType == HIP_R_32F)
        {
            auto bilinearSolutions = hiptensor::enumerateContractionSolutions<
                2,
                2,
                2,
                float,
                float,
                ck::Tuple<float>,
                float,
                ck::tensor_operation::element_wise::PassThrough,
                ck::tensor_operation::element_wise::PassThrough,
                ck::tensor_operation::element_wise::Bilinear>();

            registry.registerSolutions(std::move(bilinearSolutions));
        }
        else if(ADataType == HIP_R_64F && BDataType == HIP_R_64F && CDataType == HIP_R_64F
                && DDataType == HIP_R_64F)
        {
            auto bilinearSolutions = hiptensor::enumerateContractionSolutions<
                2,
                2,
                2,
                double,
                double,
                ck::Tuple<double>,
                double,
                ck::tensor_operation::element_wise::PassThrough,
                ck::tensor_operation::element_wise::PassThrough,
                ck::tensor_operation::element_wise::Bilinear>();

            registry.registerSolutions(std::move(bilinearSolutions));
        }
    }
    else if(plan->ht_plan_desc.ht_contract_op == (int)hiptensor::ContractionOpId_t::SCALE)
    {
        if(ADataType == HIP_R_32F && BDataType == HIP_R_32F && DDataType == HIP_R_32F)
        {
            auto scaleSolutions = hiptensor::enumerateContractionSolutions<
                2,
                2,
                2,
                float,
                float,
                ck::Tuple<>,
                float,
                ck::tensor_operation::element_wise::PassThrough,
                ck::tensor_operation::element_wise::PassThrough,
                ck::tensor_operation::element_wise::Scale>();

            registry.registerSolutions(std::move(scaleSolutions));
        }
        else if(ADataType == HIP_R_64F && BDataType == HIP_R_64F && DDataType == HIP_R_64F)
        {
            auto scaleSolutions = hiptensor::enumerateContractionSolutions<
                2,
                2,
                2,
                double,
                double,
                ck::Tuple<>,
                double,
                ck::tensor_operation::element_wise::PassThrough,
                ck::tensor_operation::element_wise::PassThrough,
                ck::tensor_operation::element_wise::Scale>();

            registry.registerSolutions(std::move(scaleSolutions));
        }
    }

    /// Dispatching end
    auto solutions
        = registry.querySolutions(2,
                                  2,
                                  2,
                                  ADataType,
                                  BDataType,
                                  CDataType,
                                  DDataType,
                                  hiptensorOperator_t::HIPTENSOR_OP_IDENTITY,
                                  hiptensorOperator_t::HIPTENSOR_OP_IDENTITY,
                                  (hiptensor::ContractionOpId_t)plan->ht_plan_desc.ht_contract_op);

    // Now we can launch the kernels and get the metrics.
    std::cout << "Run all instances and do timing: " << solutions.size() << std::endl;

    std::string                   best_op_name;
    bool                          found     = false;
    hiptensorContractionMetrics_t bestFound = {0, 0, 0, ""};

    for(auto& solution : solutions)
    {
        if(solution->initArgs(alpha,
                              A,
                              B,
                              beta,
                              C,
                              D,
                              a_ms_ns_lengths,
                              a_ms_ks_strides,
                              b_ns_ks_lengths,
                              b_ns_ks_strides,
                              std::vector<std::vector<ck::index_t>>{e_ms_ns_lengths},
                              std::vector<std::vector<ck::index_t>>{e_ms_ns_strides},
                              e_ms_ns_lengths,
                              e_ms_ns_strides))
        {
            // Make sure to time the kernels
            auto    time = (*solution)(StreamConfig{stream, true});
            int32_t m, n, k;
            std::tie(m, n, k) = solution->problemDims();
            auto flops        = std::size_t(2) * m * n * k;
            auto bytes        = solution->problemBytes();

            hiptensorContractionMetrics_t metrics = {
                time, // avg time
                static_cast<float>(flops) / static_cast<float>(1.E9) / time, // tflops
                static_cast<float>(bytes) / static_cast<float>(1.E6) / time, //
                solution->kernelName() // name
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

    return HIPTENSOR_STATUS_SUCCESS;
}
