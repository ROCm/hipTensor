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
#include <hiptensor/hiptensor.hpp>

#include "contraction_selection.hpp"
#include "contraction_solution.hpp"
#include "contraction_solution_instances.hpp"
#include "contraction_solution_registry.hpp"
#include "handle.hpp"
#include "hip_device.hpp"
#include "logger.hpp"

// Convert between vectors of void ptrs stored in opaque API objects
// to vectors of ContractionSolution ptrs with simple cast.
inline auto toContractionSolutionVec(std::vector<void*> const& v)
{
    auto result = std::vector<hiptensor::ContractionSolution*>(v.size());
    std::transform(v.begin(), v.end(), result.begin(), [](auto* p) {
        return (hiptensor::ContractionSolution*)p;
    });
    return result;
}

inline auto toContractionSolutionVec(
    std::unordered_map<std::size_t, hiptensor::ContractionSolution*> const& map)
{
    auto result = std::vector<hiptensor::ContractionSolution*>(map.size());
    transform(map.begin(), map.end(), result.begin(), [](auto p) { return p.second; });
    return result;
}

inline auto toVoidVec(std::vector<hiptensor::ContractionSolution*> const& v)
{
    auto result = std::vector<void*>(v.size());
    transform(v.begin(), v.end(), result.begin(), [](auto* p) { return (void*)p; });
    return result;
}

inline auto toVoidVec(std::unordered_map<std::size_t, hiptensor::ContractionSolution*> const& map)
{
    auto result = std::vector<void*>(map.size());
    transform(map.begin(), map.end(), result.begin(), [](auto p) { return (void*)p.second; });
    return result;
}

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
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[1024];
    snprintf(
        msg,
        sizeof(msg),
        "handle=0x%0*llX, desc=0x%llX, descA=0x%llX, modeA=0x%llX, alignmentRequirementA=0x%02X, "
        "descB=0x%llX, modeB=0x%llX, alignmentRequirementB=0x%02X, descC=0x%llX, modeC=0x%llX, "
        "alignmentRequirementC=0x%02X, descD=0x%llX, modeD=0x%llX, alignmentRequirementD=0x%02X, "
        "typeCompute=0x%02X",
        2 * (int)sizeof(void*),
        (unsigned long long)handle,
        (unsigned long long)desc,
        (unsigned long long)descA,
        (unsigned long long)modeA,
        (unsigned int)alignmentRequirementA,
        (unsigned long long)descB,
        (unsigned long long)modeB,
        (unsigned int)alignmentRequirementB,
        (unsigned long long)descC,
        (unsigned long long)modeC,
        (unsigned int)alignmentRequirementC,
        (unsigned long long)descD,
        (unsigned long long)modeD,
        (unsigned int)alignmentRequirementD,
        (unsigned int)typeCompute);

    logger->logAPITrace("hiptensorInitContractionDescriptor", msg);

    if(!handle || !desc || !descA || !descB || !descD)
    {
        auto errorCode = HIPTENSOR_STATUS_NOT_INITIALIZED;
        if(!handle)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : handle = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        else if(!desc)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : contraction descriptor = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        else
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : Tensor descriptors = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        logger->logError("hiptensorInitContractionDescriptor", msg);
        return errorCode;
    }

    if(descC == nullptr || modeC == nullptr)
    {
        // Use a scale contraction due to
        // tensor C-descriptor is empty

        *desc = {(int32_t)hiptensor::ContractionOpId_t::SCALE,
                 typeCompute,
                 {*descA,
                  *descB,
                  {hiptensor::NONE_TYPE,
                   std::vector<std::size_t>(descD->mLengths.size(), 0),
                   std::vector<std::size_t>(descD->mStrides.size(), 0)},
                  *descD},
                 {alignmentRequirementA, alignmentRequirementB, 0, alignmentRequirementD}};
    }
    else
    {
        // Use a bilinear contraction due to
        // tensor C-descriptor is not empty
        *desc = {(int32_t)hiptensor::ContractionOpId_t::BILINEAR,
                 typeCompute,
                 {*descA, *descB, *descC, *descD},
                 {alignmentRequirementA,
                  alignmentRequirementB,
                  alignmentRequirementC,
                  alignmentRequirementD}};
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorInitContractionFind(const hiptensorHandle_t*    handle,
                                               hiptensorContractionFind_t* find,
                                               const hiptensorAlgo_t       algo)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[256];
    snprintf(msg,
             sizeof(msg),
             "handle=0x%0*llX, find=0x%llX, algo=0x%02X",
             2 * (int)sizeof(void*),
             (unsigned long long)handle,
             (unsigned long long)find,
             (int)algo);

    logger->logAPITrace("hiptensorInitContractionFind", msg);

    if(handle == nullptr || find == nullptr)
    {
        auto errorCode = HIPTENSOR_STATUS_NOT_INITIALIZED;
        if(handle == nullptr)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : handle = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        else
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : contraction find = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        logger->logError("hiptensorInitContractionFind", msg);
        return errorCode;
    }

    auto realHandle = hiptensor::Handle::toHandle((int64_t*)handle->fields);

    // Ensure current HIP device is same as the handle.
    hiptensor::HipDevice currentDevice;
    if((int)currentDevice.getDeviceId() != realHandle->getDevice().getDeviceId())
    {
        auto errorCode = HIPTENSOR_STATUS_ARCH_MISMATCH;
        snprintf(msg,
                 sizeof(msg),
                 "Device mismatch error: current device id: %d, handle device id: %d (%s)",
                 (int)currentDevice.getDeviceId(),
                 (int)realHandle->getDevice().getDeviceId(),
                 hiptensorGetErrorString(errorCode));

        logger->logError("hiptensorInitContractionFind", msg);
        return errorCode;
    }

    if(algo == HIPTENSOR_ALGO_DEFAULT || algo == HIPTENSOR_ALGO_DEFAULT_PATIENT
       || algo == HIPTENSOR_ALGO_ACTOR_CRITIC)
    {
        // Update the stored selection algorithm
        find->mSelectionAlgorithm = algo;

        // For now, enumerate all known contraction kernels.
        // Using the hipDevice, determine if the device supports F64
        auto& instances = hiptensor::ContractionSolutionInstances::instance();
        auto  solnQ     = instances->allSolutions();

        // Check if the current device supports F64
        if(!currentDevice.supportsF64())
        {
            // Allow only supported f32 combos
            solnQ = solnQ.query(HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F) || // Bilinear F32
                    solnQ.query(HIP_R_32F,
                                HIP_R_32F,
                                hipDataType(hiptensor::NONE_TYPE),
                                HIP_R_32F); // Scale F32 (no C)
        }

        // Can do more checking for scale / bilinear, etc. if we need to.

        if(solnQ.solutionCount() == 0)
        {
            // No kernels found!
            auto errorCode = HIPTENSOR_STATUS_INTERNAL_ERROR;
            snprintf(msg,
                     sizeof(msg),
                     "Internal Error : No Kernels Found (%s)",
                     hiptensorGetErrorString(errorCode));
            logger->logError("hiptensorInitContractionFind", msg);
            return errorCode;
        }

        // Extract the solutions to the candidates vector.
        find->mCandidates = toVoidVec(solnQ.solutions());

        return HIPTENSOR_STATUS_SUCCESS;
    }
    else
    {
        auto errorCode = HIPTENSOR_STATUS_INVALID_VALUE;
        snprintf(msg, sizeof(msg), "Invalid Algo Value (%s)", hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorInitContractionFind", msg);
        return errorCode;
    }
}

hiptensorStatus_t hiptensorContractionGetWorkspaceSize(const hiptensorHandle_t* handle,
                                                       const hiptensorContractionDescriptor_t* desc,
                                                       const hiptensorContractionFind_t*       find,
                                                       const hiptensorWorksizePreference_t     pref,
                                                       uint64_t* workspaceSize)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[512];
    snprintf(msg,
             sizeof(msg),
             "handle=0x%0*llX, desc=0x%llX, find=0x%llX, pref=0x%02X, workspaceSize=0x%04lX",
             2 * (int)sizeof(void*),
             (unsigned long long)handle,
             (unsigned long long)desc,
             (unsigned long long)find,
             (unsigned int)pref,
             (unsigned long)*workspaceSize);
    logger->logAPITrace("hiptensorContractionGetWorkspaceSize", msg);

    if(handle == nullptr || desc == nullptr || find == nullptr || workspaceSize == nullptr)
    {
        auto errorCode = HIPTENSOR_STATUS_NOT_INITIALIZED;
        if(handle == nullptr)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : handle = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        else if(desc == nullptr)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : contraction descriptor = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        else if(find == nullptr)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : contraction find = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        else if(workspaceSize == nullptr)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : workspace size = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        logger->logError("hiptensorContractionGetWorkspaceSize", msg);
        return errorCode;
    }

    *workspaceSize = 0u;

    for(auto* candidate : find->mCandidates)
    {
        auto* solution = (hiptensor::ContractionSolution*)candidate;
        if(solution->initArgs(nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              desc->mTensorDesc[0].mLengths,
                              desc->mTensorDesc[0].mStrides,
                              desc->mTensorDesc[1].mLengths,
                              desc->mTensorDesc[1].mStrides,
                              desc->mTensorDesc[2].mLengths,
                              desc->mTensorDesc[2].mStrides,
                              desc->mTensorDesc[3].mLengths,
                              desc->mTensorDesc[3].mStrides,
                              nullptr))
        {
            if(*workspaceSize == 0)
            {
                *workspaceSize = solution->workspaceSize();
            }
            else
            {
                if(pref == HIPTENSOR_WORKSPACE_MIN)
                {
                    *workspaceSize = std::min(*workspaceSize, solution->workspaceSize());
                }
                else
                {
                    *workspaceSize = std::max(*workspaceSize, solution->workspaceSize());
                }
            }
        }
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorInitContractionPlan(const hiptensorHandle_t*                handle,
                                               hiptensorContractionPlan_t*             plan,
                                               const hiptensorContractionDescriptor_t* desc,
                                               const hiptensorContractionFind_t*       find,
                                               const uint64_t workspaceSize)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access

    char msg[256];
    snprintf(msg,
             sizeof(msg),
             "handle=0x%0*llX, plan=0x%llX, desc=0x%llX, find=0x%llX, workspaceSize=0x%04lX",
             2 * (int)sizeof(void*),
             (unsigned long long)handle,
             (unsigned long long)plan,
             (unsigned long long)desc,
             (unsigned long long)find,
             (unsigned long)workspaceSize);
    logger->logAPITrace("hiptensorInitContractionPlan", msg);

    if(handle == nullptr || plan == nullptr || desc == nullptr || find == nullptr)
    {
        auto errorCode = HIPTENSOR_STATUS_NOT_INITIALIZED;
        if(handle == nullptr)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : handle = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        else if(plan == nullptr)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : plan = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        else if(desc == nullptr)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : contraction descriptor = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        else
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : contraction find = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        logger->logError("hiptensorInitContractionPlan", msg);
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    auto realHandle = hiptensor::Handle::toHandle((int64_t*)handle->fields);

    // Ensure current HIP device is same as the handle.
    hiptensor::HipDevice currentDevice;
    if((int)currentDevice.getDeviceId() != realHandle->getDevice().getDeviceId())
    {
        auto errorCode = HIPTENSOR_STATUS_ARCH_MISMATCH;
        snprintf(msg,
                 sizeof(msg),
                 "Device mismatch error: current device id: %d, handle device id: %d (%s)",
                 (int)currentDevice.getDeviceId(),
                 (int)realHandle->getDevice().getDeviceId(),
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorInitContractionPlan", msg);
        return HIPTENSOR_STATUS_ARCH_MISMATCH;
    }

    // At this point, we need to format inputs for kernels as they will be tested via selection model.
    // Brute force method currently uses CK kernel format, so we will adjust inputs to that style.

    // Convert to concrete contraction solutions
    auto candidates = toContractionSolutionVec(find->mCandidates);

    auto ADataType = desc->mTensorDesc[0].mType;
    auto BDataType = desc->mTensorDesc[1].mType;
    auto DDataType = desc->mTensorDesc[2].mType;
    auto EDataType = desc->mTensorDesc[3].mType;

    // Query contraction solutions for the correct contraction operation and type
    auto solutionQ = hiptensor::ContractionSolutionRegistry::Query{candidates}
                         .query((hiptensor::ContractionOpId_t)desc->mContractionOpId)
                         .query(ADataType, BDataType, DDataType, EDataType);

    candidates = toContractionSolutionVec(solutionQ.solutions());

    // Measure timing for solution selection
    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    CHECK_HIP_ERROR(hipEventRecord(startEvent));

    // Launch selection algorithm
    hiptensor::ContractionSolution* winner = nullptr;
    auto                            result = HIPTENSOR_STATUS_INTERNAL_ERROR;
    if(find->mSelectionAlgorithm == HIPTENSOR_ALGO_DEFAULT
       || find->mSelectionAlgorithm == HIPTENSOR_ALGO_DEFAULT_PATIENT)
    {
        result = hiptensor::bruteForceModel(&winner,
                                            candidates,
                                            ADataType,
                                            desc->mTensorDesc[0].mLengths,
                                            desc->mTensorDesc[0].mStrides,
                                            BDataType,
                                            desc->mTensorDesc[1].mLengths,
                                            desc->mTensorDesc[1].mStrides,
                                            DDataType,
                                            desc->mTensorDesc[2].mLengths,
                                            desc->mTensorDesc[2].mStrides,
                                            EDataType,
                                            desc->mTensorDesc[3].mLengths,
                                            desc->mTensorDesc[3].mStrides,
                                            workspaceSize);
    }
    else if(find->mSelectionAlgorithm == HIPTENSOR_ALGO_ACTOR_CRITIC)
    {
        result = hiptensor::actorCriticModel(&winner,
                                             solutionQ.solutions(),
                                             ADataType,
                                             desc->mTensorDesc[0].mLengths,
                                             desc->mTensorDesc[0].mStrides,
                                             BDataType,
                                             desc->mTensorDesc[1].mLengths,
                                             desc->mTensorDesc[1].mStrides,
                                             DDataType,
                                             desc->mTensorDesc[2].mLengths,
                                             desc->mTensorDesc[2].mStrides,
                                             EDataType,
                                             desc->mTensorDesc[3].mLengths,
                                             desc->mTensorDesc[3].mStrides,
                                             workspaceSize);
    }

    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));

    if(result != HIPTENSOR_STATUS_SUCCESS)
    {
        snprintf(msg,
                 sizeof(msg),
                 "Init contraction plan not successful (%s)",
                 hiptensorGetErrorString(result));
        logger->logError("hiptensorInitContractionPlan", msg);
        return result;
    }

    // Log the selected contraction solution and selection timing
    snprintf(msg,
             sizeof(msg),
             "Algo: %d, KernelId: %lu, KernelName: %s, SelectionTime: %0.3f ms",
             find->mSelectionAlgorithm,
             winner->uid(),
             winner->kernelName().c_str(),
             elapsedTimeMs);
    logger->logPerformanceTrace("hiptensorInitContractionPlan", msg);

    // Assign the contraction descriptor
    plan->mContractionDesc = *desc;
    plan->mSolution        = winner;

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
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[512];
    char alphaMsg[32];
    char betaMsg[32];

    if(plan != nullptr)
    {
        if(alpha == nullptr)
        {
            snprintf(alphaMsg, sizeof(alphaMsg), "alpha=NULL");
        }
        else
        {
            if(plan->mContractionDesc.mComputeType == HIPTENSOR_COMPUTE_32F)
            {
                snprintf(
                    alphaMsg, sizeof(alphaMsg), "alpha=%.6f", *(static_cast<const float*>(alpha)));
            }
            else if(plan->mContractionDesc.mComputeType == HIPTENSOR_COMPUTE_64F)
            {
                snprintf(alphaMsg,
                         sizeof(alphaMsg),
                         "alpha=%.6lf",
                         *(static_cast<const double*>(alpha)));
            }
        }

        if(beta == nullptr)
        {
            snprintf(betaMsg, sizeof(betaMsg), "beta=NULL");
        }
        else
        {
            if(plan->mContractionDesc.mComputeType == HIPTENSOR_COMPUTE_32F)
            {
                snprintf(betaMsg, sizeof(betaMsg), "beta=%.6f", *(static_cast<const float*>(beta)));
            }
            else if(plan->mContractionDesc.mComputeType == HIPTENSOR_COMPUTE_64F)
            {
                snprintf(
                    betaMsg, sizeof(betaMsg), "beta=%.6lf", *(static_cast<const double*>(beta)));
            }
        }
    }
    else
    {
        snprintf(alphaMsg, sizeof(alphaMsg), "alpha=NULL");
        snprintf(betaMsg, sizeof(betaMsg), "beta=NULL");
    }

    snprintf(msg,
             sizeof(msg),
             "handle=0x%0*llX, plan=0x%llX, %s, A=0x%llX, B=0x%llX, %s, "
             "C=0x%llX, D=0x%llX, workspace=0x%llX, workspaceSize=0x%04lX, stream=0x%llX",
             2 * (int)sizeof(void*),
             (unsigned long long)handle,
             (unsigned long long)plan,
             alphaMsg,
             (unsigned long long)A,
             (unsigned long long)B,
             betaMsg,
             (unsigned long long)C,
             (unsigned long long)D,
             (unsigned long long)workspace,
             (unsigned long)workspaceSize,
             (unsigned long long)stream);

    logger->logAPITrace("hiptensorContraction", msg);

    if(handle == nullptr || plan == nullptr)
    {
        auto errorCode = HIPTENSOR_STATUS_NOT_INITIALIZED;
        if(handle == nullptr)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : handle = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        else
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : plan = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        logger->logError("hiptensorContraction", msg);
        return errorCode;
    }

    if(alpha == nullptr || A == nullptr || B == nullptr || D == nullptr)
    {
        auto errorCode = HIPTENSOR_STATUS_INVALID_VALUE;
        if(alpha == nullptr)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Input Parameter Error : alpha = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        else
        {
            snprintf(msg,
                     sizeof(msg),
                     "Input Parameter Error : A/B/D = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        logger->logError("hiptensorContraction", msg);
        return errorCode;
    }

    if(plan->mSolution == nullptr)
    {
        auto errorCode = HIPTENSOR_STATUS_INTERNAL_ERROR;
        snprintf(msg,
                 sizeof(msg),
                 "Internal Error : solution = nullptr (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorContraction", msg);
        return errorCode;
    }

    auto realHandle = hiptensor::Handle::toHandle((int64_t*)handle->fields);

    // Ensure current HIP device is same as the handle.
    hiptensor::HipDevice currentDevice;
    if((int)currentDevice.getDeviceId() != realHandle->getDevice().getDeviceId())
    {
        auto errorCode = HIPTENSOR_STATUS_ARCH_MISMATCH;
        snprintf(msg,
                 sizeof(msg),
                 "Device mismatch error: current device id: %d, handle device id: %d (%s)",
                 (int)currentDevice.getDeviceId(),
                 (int)realHandle->getDevice().getDeviceId(),
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorContraction", msg);
        return errorCode;
    }

    if(plan->mContractionDesc.mComputeType != plan->mContractionDesc.mTensorDesc[3].mType)
    {
        auto errorCode = HIPTENSOR_STATUS_INVALID_VALUE;
        snprintf(msg,
                 sizeof(msg),
                 "Internal Error : compute type != D type (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorContraction", msg);
        return errorCode;
    }

    auto* cSolution = (hiptensor::ContractionSolution*)(plan->mSolution);

    auto canRun = cSolution->initArgs(alpha,
                                      A,
                                      B,
                                      beta,
                                      C,
                                      D,
                                      plan->mContractionDesc.mTensorDesc[0].mLengths,
                                      plan->mContractionDesc.mTensorDesc[0].mStrides,
                                      plan->mContractionDesc.mTensorDesc[1].mLengths,
                                      plan->mContractionDesc.mTensorDesc[1].mStrides,
                                      plan->mContractionDesc.mTensorDesc[2].mLengths,
                                      plan->mContractionDesc.mTensorDesc[2].mStrides,
                                      plan->mContractionDesc.mTensorDesc[3].mLengths,
                                      plan->mContractionDesc.mTensorDesc[3].mStrides,
                                      workspace);

    if(canRun)
    {
        if(cSolution->workspaceSize() > workspaceSize)
        {
            auto errorCode = HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE;
            snprintf(msg,
                     sizeof(msg),
                     "Insufficient workspace: req: %lu alloc: %lu (%s)",
                     cSolution->workspaceSize(),
                     workspaceSize,
                     hiptensorGetErrorString(errorCode));
            logger->logError("hiptensorContraction", msg);
            return errorCode;
        }

        // Perform contraction with timing if LOG_LEVEL_PERF_TRACE
        if(logger->getLogMask() & HIPTENSOR_LOG_LEVEL_PERF_TRACE)
        {
            auto time = (*cSolution)(StreamConfig{stream, true});

            int32_t m, n, k;
            std::tie(m, n, k) = cSolution->problemDims();
            auto flops        = std::size_t(2) * m * n * k;
            auto bytes        = cSolution->problemBytes();

            hiptensor::PerfMetrics metrics = {
                cSolution->uid(), // id
                cSolution->kernelName(), // name
                time, // avg time
                static_cast<float>(flops) / static_cast<float>(1.E9) / time, // tflops
                static_cast<float>(bytes) / static_cast<float>(1.E6) / time // BW
            };

            // log perf metrics (not name/id)
            snprintf(msg,
                     sizeof(msg),
                     "KernelId: %lu KernelName: %s, %0.3f ms, %0.3f TFlops, %0.3f GB/s",
                     metrics.mKernelUid,
                     metrics.mKernelName.c_str(),
                     metrics.mAvgTimeMs,
                     metrics.mTflops,
                     metrics.mBandwidth);
            logger->logPerformanceTrace("hiptensorContraction", msg);
        }
        // Perform contraction without timing
        else
        {
            (*cSolution)(StreamConfig{stream, false});
        }

        return HIPTENSOR_STATUS_SUCCESS;
    }
    else
    {
        auto errorCode = HIPTENSOR_STATUS_INTERNAL_ERROR;
        snprintf(msg,
                 sizeof(msg),
                 "Selected kernel is unable to solve the problem (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorContraction", msg);
        return errorCode;
    }
}
