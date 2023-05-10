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
#include "contraction_heuristics.hpp"
#include "contraction_solution.hpp"
#include "contraction_solution_registry.hpp"
#include "hip_device.hpp"
#include "hiptensor.hpp"

hiptensorContractionMetrics_t ht_contract_metrics;

hiptensorStatus_t hiptensorInitContractionDescriptor(const hiptensorHandle_t*           handle,
                                                     hiptensorContractionDescriptor_t*  desc,
                                                     const hiptensorTensorDescriptor_t* descA,
                                                     const int32_t                      modeA[],
                                                     const uint32_t alignmentRequirementA,
                                                     const hiptensorTensorDescriptor_t* descB,
                                                     const int32_t                      modeB[],
                                                     const uint32_t alignmentRequirementB,
                                                     const hiptensorTensorDescriptor_t* descC,
                                                     const int32_t                      modeC[],
                                                     const uint32_t alignmentRequirementC,
                                                     const hiptensorTensorDescriptor_t* descD,
                                                     const int32_t                      modeD[],
                                                     const uint32_t         alignmentRequirementD,
                                                     hiptensorComputeType_t typeCompute)

{
    if(!handle || !desc || !descA || !descB || !descD)
    {
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    const hiptensorTensorDescriptor_t* ht_input_descs[]           = {descA, descB, descC, descD};
    const uint32_t                     alignmentRequirement_arr[] = {
        alignmentRequirementA, alignmentRequirementB, alignmentRequirementC, alignmentRequirementD};
    desc->hiptensorContractionAttrUpdate(ht_input_descs, alignmentRequirement_arr, 4);

    if(descC == nullptr || modeC == nullptr)
    {
        desc->ht_contract_op = (int32_t)hiptensor::ContractionOpId_t::SCALE;
    }
    else
    {
        desc->ht_contract_op = (int32_t)hiptensor::ContractionOpId_t::BILINEAR;
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorInitContractionFind(const hiptensorHandle_t*    handle,
                                               hiptensorContractionFind_t* find,
                                               const hiptensorAlgo_t       algo)
{
    if(handle == nullptr || find == nullptr)
    {
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    // Ensure current HIP device is same as the handle.
    hiptensor::HipDevice currentDevice;
    if((int)currentDevice.getDeviceHandle() != handle->mHipDevice)
    {
        return HIPTENSOR_STATUS_INTERNAL_ERROR;
    }

    if(algo == HIPTENSOR_ALGO_DEFAULT)
    {
        // Update the stored selection algorithm
        find->mSelectionAlgorithm = HIPTENSOR_ALGO_DEFAULT;

        // For now, enumerate all known contraction kernels.
        // Using the hipDevice, determine if the device supports F64
        auto& registry = hiptensor::ContractionSolutionRegistry::instance();
        auto  query    = registry->allSolutions();

        // Check if the current device supports F64
        if(!currentDevice.supportsF64())
        {
            // Allow only supported f32 combos
            query = query.query(HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F)
                    || query.query(HIP_R_32F, HIP_R_32F, hipDataType(-1), HIP_R_32F);
        }

        // Can do more checking for scale / bilinear, etc. if we need to.

        if(query.solutionCount() == 0)
        {
            // No kernels found!
            return HIPTENSOR_STATUS_INTERNAL_ERROR;
        }

        // Retrieve the solution map
        auto& solutions = query.solutions();

        // Extract the solutions to the candidates vector.
        find->mCandidates.resize(solutions.size());
        transform(solutions.begin(), solutions.end(), find->mCandidates.begin(), [](auto p) {
            return (void*)p.second;
        });

        return HIPTENSOR_STATUS_SUCCESS;
    }
    else
    {
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }
}

hiptensorStatus_t hiptensorContractionGetWorkspaceSize(const hiptensorHandle_t* handle,
                                                       const hiptensorContractionDescriptor_t* desc,
                                                       const hiptensorContractionFind_t*       find,
                                                       const hiptensorWorksizePreference_t     pref,
                                                       uint64_t* workspaceSize)
{
    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorInitContractionPlan(const hiptensorHandle_t*                handle,
                                               hiptensorContractionPlan_t*             plan,
                                               const hiptensorContractionDescriptor_t* desc,
                                               const hiptensorContractionFind_t*       find,
                                               const uint64_t workspaceSize)
{
    if(handle == nullptr || plan == nullptr || desc == nullptr || find == nullptr)
    {
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    hiptensor::HipDevice currentDevice;
    if((int)currentDevice.getDeviceHandle() != handle->mHipDevice)
    {
        return HIPTENSOR_STATUS_INTERNAL_ERROR;
    }

    // At this point, we need to format inputs for kernels as they will be tested via heuristic.
    // Brute force method currently uses CK kernel format, so we will adjust inputs to that style.

    // Convert to concrete contraction solutions
    auto candidates = std::vector<hiptensor::ContractionSolution*>(find->mCandidates.size());
    transform(find->mCandidates.begin(), find->mCandidates.end(), candidates.begin(), [](auto* p) {
        return (hiptensor::ContractionSolution*)p;
    });

    // Query contraction solutions for the correct contraction operation
    auto solutionMap = hiptensor::ContractionSolutionRegistry::Query{candidates}
                           .query((hiptensor::ContractionOpId_t)desc->ht_contract_op)
                           .solutions();
    candidates.resize(solutionMap.size());
    transform(solutionMap.begin(), solutionMap.end(), candidates.begin(), [](auto p) {
        return (hiptensor::ContractionSolution*)p.second;
    });

    // NOTE: Here, ck::index_t is int, NOT same as std::index_t = long uint
    // Therefore the conversion to ck::index_t is required.
    auto toCKVec
        = [](auto& inputVec) { return std::vector<ck::index_t>(inputVec.begin(), inputVec.end()); };

    auto ADataType = desc->ht_contract_attr_desc[0].ht_type;
    auto BDataType = desc->ht_contract_attr_desc[1].ht_type;
    auto DDataType = desc->ht_contract_attr_desc[2].ht_type;
    auto EDataType = desc->ht_contract_attr_desc[3].ht_type;

    auto a_ms_ks_lengths = toCKVec(desc->ht_contract_attr_desc[0].lens);
    auto a_ms_ks_strides = toCKVec(desc->ht_contract_attr_desc[0].strides);
    auto b_ns_ks_lengths = toCKVec(desc->ht_contract_attr_desc[1].lens);
    auto b_ns_ks_strides = toCKVec(desc->ht_contract_attr_desc[1].strides);
    auto d_ms_ns_lengths = toCKVec(desc->ht_contract_attr_desc[2].lens);
    auto d_ms_ns_strides = toCKVec(desc->ht_contract_attr_desc[2].strides);
    auto e_ms_ns_lengths = toCKVec(desc->ht_contract_attr_desc[3].lens);
    auto e_ms_ns_strides = toCKVec(desc->ht_contract_attr_desc[3].strides);

    // Launch heuristic
    hiptensor::ContractionSolution* winner = nullptr;
    auto                            result = hiptensor::bruteForceHeuristic(&winner,
                                                 candidates,
                                                 ADataType,
                                                 a_ms_ks_lengths,
                                                 a_ms_ks_strides,
                                                 BDataType,
                                                 b_ns_ks_lengths,
                                                 b_ns_ks_strides,
                                                 DDataType,
                                                 d_ms_ns_lengths,
                                                 d_ms_ns_strides,
                                                 EDataType,
                                                 e_ms_ns_lengths,
                                                 e_ms_ns_strides,
                                                 workspaceSize);

    if(result != HIPTENSOR_STATUS_SUCCESS)
    {
        return result;
    }

    // Assign the contraction descriptor
    plan->ht_plan_desc = *desc;
    plan->mSolution    = winner;

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorContraction(const hiptensorHandle_t*          handle,
                                       const hiptensorContractionPlan_t* plan,
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
    if(handle == nullptr || plan == nullptr)
    {
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    if(alpha == nullptr || A == nullptr || B == nullptr || D == nullptr)
    {
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    if(plan->mSolution == nullptr)
    {
        return HIPTENSOR_STATUS_INTERNAL_ERROR;
    }

    auto* cSolution = (hiptensor::ContractionSolution*)(plan->mSolution);

    // NOTE: Here, ck::index_t is int, NOT same as std::index_t = long uint
    // Therefore the conversion to ck::index_t is required.
    auto toCKVec
        = [](auto& inputVec) { return std::vector<ck::index_t>(inputVec.begin(), inputVec.end()); };

    auto a_ms_ks_lengths = toCKVec(plan->ht_plan_desc.ht_contract_attr_desc[0].lens);
    auto a_ms_ks_strides = toCKVec(plan->ht_plan_desc.ht_contract_attr_desc[0].strides);

    auto b_ns_ks_lengths = toCKVec(plan->ht_plan_desc.ht_contract_attr_desc[1].lens);
    auto b_ns_ks_strides = toCKVec(plan->ht_plan_desc.ht_contract_attr_desc[1].strides);

    auto d_ms_ns_lengths = toCKVec(plan->ht_plan_desc.ht_contract_attr_desc[2].lens);
    auto d_ms_ns_strides = toCKVec(plan->ht_plan_desc.ht_contract_attr_desc[2].strides);

    auto e_ms_ns_lengths = toCKVec(plan->ht_plan_desc.ht_contract_attr_desc[3].lens);
    auto e_ms_ns_strides = toCKVec(plan->ht_plan_desc.ht_contract_attr_desc[3].strides);

    auto canRun = cSolution->initArgs(alpha,
                                      A,
                                      B,
                                      beta,
                                      C,
                                      D,
                                      a_ms_ks_lengths,
                                      a_ms_ks_strides,
                                      b_ns_ks_lengths,
                                      b_ns_ks_strides,
                                      std::vector<std::vector<ck::index_t>>{d_ms_ns_lengths},
                                      std::vector<std::vector<ck::index_t>>{d_ms_ns_strides},
                                      e_ms_ns_lengths,
                                      e_ms_ns_strides);

    if(cSolution->params()->opCDE() == hiptensor::ContractionOpId_t::SCALE && canRun)
    {
        std::cout << "Running a scale!!" << std::endl;
    }

    if(canRun)
    {
        (*cSolution)(StreamConfig{stream, false});
        return HIPTENSOR_STATUS_SUCCESS;
    }
    else
    {
        return HIPTENSOR_STATUS_INTERNAL_ERROR;
    }
}

void hiptensorContractionPlan_t::hiptensorPrintContractionMetrics()
{
    std::cout << "Perf: " << ht_contract_metrics.avg_time << " ms, " << ht_contract_metrics.tflops
              << " TFlops, " << ht_contract_metrics.transfer_speed << " GB/s, "
              << ht_contract_metrics.ht_instance << std::endl;
}
