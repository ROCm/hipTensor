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
#include "contraction_selection.hpp"
#include "contraction_solution.hpp"
#include "contraction_solution_registry.hpp"
#include "handle.hpp"
#include "hip_device.hpp"
#include "hiptensor.hpp"

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
    char msg[512];
    sprintf(
        msg,
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
        sprintf(
            msg,
            "handle=0x%0*llX, desc=0x%llX, descA=0x%llX, modeA=0x%llX, "
            "alignmentRequirementA=0x%02X, "
            "descB=0x%llX, modeB=0x%llX, alignmentRequirementB=0x%02X, descC=0x%llX, modeC=0x%llX, "
            "alignmentRequirementC=0x%02X, descD=0x%llX, modeD=0x%llX, "
            "alignmentRequirementD=0x%02X, "
            "typeCompute=0x%02X (HIPTENSOR_STATUS_NOT_INITIALIZED)",
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

        logger->logError("hiptensorInitContractionDescriptor", msg);
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    int32_t contractionOp;
    if(descC == nullptr || modeC == nullptr)
    {
        // C-descriptor is empty
        *desc = {(int32_t)hiptensor::ContractionOpId_t::SCALE,
                 {*descA, *descB, {(hipDataType)-1, {}, {}}, *descD},
                 {alignmentRequirementA, alignmentRequirementB, 0, alignmentRequirementD}};
    }
    else
    {
        // C-descriptor is not empty
        *desc = {(int32_t)hiptensor::ContractionOpId_t::BILINEAR,
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
    char msg[128];
    sprintf(msg,
            "handle=0x%0*llX, find=0x%llX, algo=0x%02X",
            2 * (int)sizeof(void*),
            (unsigned long long)handle,
            (unsigned long long)find,
            (int)algo);

    logger->logAPITrace("hiptensorInitContractionFind", msg);

    if(handle == nullptr || find == nullptr)
    {
        sprintf(msg,
                "handle=0x%0*llX, find=0x%llX, algo=0x%02X (HIPTENSOR_STATUS_NOT_INITIALIZED)",
                2 * (int)sizeof(void*),
                (unsigned long long)handle,
                (unsigned long long)find,
                algo);
        logger->logError("hiptensorInitContractionFind", msg);
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    auto realHandle = hiptensor::Handle::toHandle((int64_t*)handle->fields);

    // Ensure current HIP device is same as the handle.
    hiptensor::HipDevice currentDevice;
    if((int)currentDevice.getDeviceId() != realHandle->getDevice().getDeviceId())
    {
        sprintf(msg,
                "handle=0x%0*llX, find=0x%llX, algo=0x%02X (HIPTENSOR_STATUS_ARCH_MISMATCH)",
                2 * (int)sizeof(void*),
                (unsigned long long)handle,
                (unsigned long long)find,
                (int)algo);
        logger->logError("hiptensorInitContractionFind", msg);
        return HIPTENSOR_STATUS_ARCH_MISMATCH;
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
            query = query.query(HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F) // Bilinear F32
                    || query.query(
                        HIP_R_32F, HIP_R_32F, hipDataType(-1), HIP_R_32F); // Scale F32 (no C)
        }

        // Can do more checking for scale / bilinear, etc. if we need to.

        if(query.solutionCount() == 0)
        {
            // No kernels found!
            sprintf(msg,
                    "handle=0x%0*llX, find=0x%llX, algo=0x%02X (HIPTENSOR_STATUS_INTERNAL_ERROR)",
                    2 * (int)sizeof(void*),
                    (unsigned long long)handle,
                    (unsigned long long)find,
                    (int)algo);
            logger->logError("hiptensorInitContractionFind", msg);
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
        sprintf(msg,
                "handle=0x%0*llX, find=0x%llX, algo=0x%02X (HIPTENSOR_STATUS_INVALID_VALUE)",
                2 * (int)sizeof(void*),
                (unsigned long long)handle,
                (unsigned long long)find,
                (int)algo);
        logger->logError("hiptensorInitContractionFind", msg);
        return HIPTENSOR_STATUS_INVALID_VALUE;
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
    char msg[256];
    sprintf(msg,
            "handle=0x%0*llX, desc=0x%llX, find=0x%llX, pref=0x%02X, workspaceSize=0x%04lX",
            2 * (int)sizeof(void*),
            (unsigned long long)handle,
            (unsigned long long)desc,
            (unsigned long long)find,
            (unsigned int)pref,
            (unsigned long)*workspaceSize);

    logger->logAPITrace("hiptensorContractionGetWorkspaceSize", msg);

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
    sprintf(msg,
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
        sprintf(msg,
                "handle=0x%0*llX, plan=0x%llX, desc=0x%llX, find=0x%llX, workspaceSize=0x%04lX "
                "(HIPTENSOR_STATUS_NOT_INITIALIZED)",
                2 * (int)sizeof(void*),
                (unsigned long long)handle,
                (unsigned long long)plan,
                (unsigned long long)desc,
                (unsigned long long)find,
                (unsigned long)workspaceSize);
        logger->logError("hiptensorInitContractionPlan", msg);
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    auto realHandle = hiptensor::Handle::toHandle((int64_t*)handle->fields);

    // Ensure current HIP device is same as the handle.
    hiptensor::HipDevice currentDevice;
    if((int)currentDevice.getDeviceId() != realHandle->getDevice().getDeviceId())
    {
        sprintf(msg,
                "handle=0x%0*llX, plan=0x%llX, desc=0x%llX, find=0x%llX, workspaceSize=0x%04lX "
                "(HIPTENSOR_STATUS_ARCH_MISMATCH)",
                2 * (int)sizeof(void*),
                (unsigned long long)handle,
                (unsigned long long)plan,
                (unsigned long long)desc,
                (unsigned long long)find,
                (unsigned long)workspaceSize);
        logger->logError("hiptensorInitContractionFind", msg);
        return HIPTENSOR_STATUS_ARCH_MISMATCH;
    }

    // At this point, we need to format inputs for kernels as they will be tested via selection model.
    // Brute force method currently uses CK kernel format, so we will adjust inputs to that style.

    // Convert to concrete contraction solutions
    auto candidates = std::vector<hiptensor::ContractionSolution*>(find->mCandidates.size());
    transform(find->mCandidates.begin(), find->mCandidates.end(), candidates.begin(), [](auto* p) {
        return (hiptensor::ContractionSolution*)p;
    });

    // Query contraction solutions for the correct contraction operation
    auto solutionMap = hiptensor::ContractionSolutionRegistry::Query{candidates}
                           .query((hiptensor::ContractionOpId_t)desc->mContractionOpId)
                           .solutions();
    candidates.resize(solutionMap.size());
    transform(solutionMap.begin(), solutionMap.end(), candidates.begin(), [](auto p) {
        return (hiptensor::ContractionSolution*)p.second;
    });

    // NOTE: Here, ck::index_t is int, NOT same as std::index_t = long uint
    // Therefore the conversion to ck::index_t is required.
    auto toCKVec
        = [](auto& inputVec) { return std::vector<ck::index_t>(inputVec.begin(), inputVec.end()); };

    auto ADataType = desc->mTensorDesc[0].mType;
    auto BDataType = desc->mTensorDesc[1].mType;
    auto DDataType = desc->mTensorDesc[2].mType;
    auto EDataType = desc->mTensorDesc[3].mType;

    auto a_ms_ks_lengths = toCKVec(desc->mTensorDesc[0].mLengths);
    auto a_ms_ks_strides = toCKVec(desc->mTensorDesc[0].mStrides);
    auto b_ns_ks_lengths = toCKVec(desc->mTensorDesc[1].mLengths);
    auto b_ns_ks_strides = toCKVec(desc->mTensorDesc[1].mStrides);
    auto d_ms_ns_lengths = toCKVec(desc->mTensorDesc[2].mLengths);
    auto d_ms_ns_strides = toCKVec(desc->mTensorDesc[2].mStrides);
    auto e_ms_ns_lengths = toCKVec(desc->mTensorDesc[3].mLengths);
    auto e_ms_ns_strides = toCKVec(desc->mTensorDesc[3].mStrides);

    // Launch selection algorithm
    hiptensor::ContractionSolution* winner = nullptr;
    auto                            result = hiptensor::bruteForceModel(&winner,
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
        sprintf(msg,
                "handle=0x%0*llX (%s)",
                2 * (int)sizeof(void*),
                (unsigned long long)handle,
                hiptensorGetErrorString(result));
        logger->logError("hiptensorInitContractionFind", msg);
        return result;
    }

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
    char  msg[512];
    float alphaVal = 0.0f;
    float betaVal  = 0.0f;

    if(plan != nullptr)
    {
        if(plan->mContractionDesc.mTensorDesc[3].mType == HIP_R_32F)
        {
            if(alpha != nullptr)
                alphaVal = *(static_cast<const float*>(alpha));

            if(beta != nullptr)
                betaVal = *(static_cast<const float*>(beta));

            sprintf(msg,
                    "handle=0x%0*llX, plan=0x%llX, alpha=%.6f, A=0x%llX, B=0x%llX, beta=%.6f, "
                    "C=0x%llX, D=0x%llX, workspace=0x%llX, workspaceSize=0x%04lX, stream=0x%llX",
                    2 * (int)sizeof(void*),
                    (unsigned long long)handle,
                    (unsigned long long)plan,
                    (float)alphaVal,
                    (unsigned long long)A,
                    (unsigned long long)B,
                    (float)betaVal,
                    (unsigned long long)C,
                    (unsigned long long)D,
                    (unsigned long long)workspace,
                    (unsigned long)workspaceSize,
                    (unsigned long long)stream);
            logger->logAPITrace("hiptensorContraction", msg);
        }
        else if(plan->mContractionDesc.mTensorDesc[3].mType == HIP_R_64F)
        {
            double alphaVal = 0.0;
            if(alpha != nullptr)
                alphaVal = *(static_cast<const double*>(alpha));

            double betaVal = 0.0;
            if(beta != nullptr)
                betaVal = *(static_cast<const double*>(beta));

            sprintf(msg,
                    "handle=0x%0*llX, plan=0x%llX, alpha=%.6lf, A=0x%llX, B=0x%llX, beta=%.6lf, "
                    "C=0x%llX, D=0x%llX, workspace=0x%llX, workspaceSize=0x%04lX, stream=0x%llX",
                    2 * (int)sizeof(void*),
                    (unsigned long long)handle,
                    (unsigned long long)plan,
                    (double)alphaVal,
                    (unsigned long long)A,
                    (unsigned long long)B,
                    (double)betaVal,
                    (unsigned long long)C,
                    (unsigned long long)D,
                    (unsigned long long)workspace,
                    (unsigned long)workspaceSize,
                    (unsigned long long)stream);
            logger->logAPITrace("hiptensorContraction", msg);
        }
    }
    else
    {
        sprintf(msg,
                "handle=0x%0*llX, plan=0x%llX, alpha=%.6f, A=0x%llX, B=0x%llX, beta=%.6f, "
                "C=0x%llX, D=0x%llX, workspace=0x%llX, workspaceSize=0x%04lX, stream=0x%llX",
                2 * (int)sizeof(void*),
                (unsigned long long)handle,
                (unsigned long long)plan,
                (float)alphaVal,
                (unsigned long long)A,
                (unsigned long long)B,
                (float)betaVal,
                (unsigned long long)C,
                (unsigned long long)D,
                (unsigned long long)workspace,
                (unsigned long)workspaceSize,
                (unsigned long long)stream);
        logger->logAPITrace("hiptensorContraction", msg);
    }

    if(handle == nullptr || plan == nullptr)
    {
        sprintf(msg,
                "handle=0x%0*llX, plan=0x%llX, alpha=%.6f, A=0x%llX, B=0x%llX, beta=%.6f, "
                "C=0x%llX, D=0x%llX, workspace=0x%llX, workspaceSize=0x%04lX, stream=0x%llX "
                "(HIPTENSOR_STATUS_NOT_INITIALIZED)",
                2 * (int)sizeof(void*),
                (unsigned long long)handle,
                (unsigned long long)plan,
                (float)alphaVal,
                (unsigned long long)A,
                (unsigned long long)B,
                (float)betaVal,
                (unsigned long long)C,
                (unsigned long long)D,
                (unsigned long long)workspace,
                (unsigned long)workspaceSize,
                (unsigned long long)stream);

        logger->logError("hiptensorContraction", msg);
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    if(alpha == nullptr || A == nullptr || B == nullptr || D == nullptr)
    {
        sprintf(msg,
                "handle=0x%0*llX, plan=0x%llX, alpha=%.6f, A=0x%llX, B=0x%llX, beta=%.6f, "
                "C=0x%llX, D=0x%llX, workspace=0x%llX, workspaceSize=0x%04lX, stream=0x%llX "
                "(HIPTENSOR_STATUS_INVALID_VALUE)",
                2 * (int)sizeof(void*),
                (unsigned long long)handle,
                (unsigned long long)plan,
                (float)alphaVal,
                (unsigned long long)A,
                (unsigned long long)B,
                (float)betaVal,
                (unsigned long long)C,
                (unsigned long long)D,
                (unsigned long long)workspace,
                (unsigned long)workspaceSize,
                (unsigned long long)stream);
        logger->logError("hiptensorContraction", msg);
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    if(plan->mSolution == nullptr)
    {
        if(plan->mContractionDesc.mTensorDesc[3].mType == HIP_R_32F)
        {
            if(alpha != nullptr)
                alphaVal = *(static_cast<const float*>(alpha));

            if(beta != nullptr)
                betaVal = *(static_cast<const float*>(beta));

            sprintf(msg,
                    "handle=0x%0*llX, plan=0x%llX, alpha=%.6f, A=0x%llX, B=0x%llX, beta=%.6f, "
                    "C=0x%llX, D=0x%llX, workspace=0x%llX, workspaceSize=0x%04lX, stream=0x%llX "
                    "(HIPTENSOR_STATUS_INTERNAL_ERROR)",
                    2 * (int)sizeof(void*),
                    (unsigned long long)handle,
                    (unsigned long long)plan,
                    (float)alphaVal,
                    (unsigned long long)A,
                    (unsigned long long)B,
                    (float)betaVal,
                    (unsigned long long)C,
                    (unsigned long long)D,
                    (unsigned long long)workspace,
                    (unsigned long)workspaceSize,
                    (unsigned long long)stream);

            logger->logError("hiptensorContraction", msg);
            return HIPTENSOR_STATUS_INTERNAL_ERROR;
        }
        else if(plan->mContractionDesc.mTensorDesc[3].mType == HIP_R_64F)
        {
            double alphaVal = 0.0;
            if(alpha != nullptr)
                alphaVal = *(static_cast<const double*>(alpha));

            double betaVal = 0.0;
            if(beta != nullptr)
                betaVal = *(static_cast<const double*>(beta));

            sprintf(msg,
                    "handle=0x%0*llX, plan=0x%llX, alpha=%.6lf, A=0x%llX, B=0x%llX, beta=%.6lf, "
                    "C=0x%llX, D=0x%llX, workspace=0x%llX, workspaceSize=0x%04lX, stream=0x%llX, "
                    "(HIPTENSOR_STATUS_INTERNAL_ERROR)",
                    2 * (int)sizeof(void*),
                    (unsigned long long)handle,
                    (unsigned long long)plan,
                    (double)alphaVal,
                    (unsigned long long)A,
                    (unsigned long long)B,
                    (double)betaVal,
                    (unsigned long long)C,
                    (unsigned long long)D,
                    (unsigned long long)workspace,
                    (unsigned long)workspaceSize,
                    (unsigned long long)stream);

            logger->logError("hiptensorContraction", msg);
            return HIPTENSOR_STATUS_INTERNAL_ERROR;
        }
    }

    auto realHandle = hiptensor::Handle::toHandle((int64_t*)handle->fields);

    // Ensure current HIP device is same as the handle.
    hiptensor::HipDevice currentDevice;
    if((int)currentDevice.getDeviceId() != realHandle->getDevice().getDeviceId())
    {
        return HIPTENSOR_STATUS_ARCH_MISMATCH;
    }

    auto* cSolution = (hiptensor::ContractionSolution*)(plan->mSolution);

    // NOTE: Here, ck::index_t is int, NOT same as std::index_t = long uint
    // Therefore the conversion to ck::index_t is required.
    auto toCKVec
        = [](auto& inputVec) { return std::vector<ck::index_t>(inputVec.begin(), inputVec.end()); };

    auto a_ms_ks_lengths = toCKVec(plan->mContractionDesc.mTensorDesc[0].mLengths);
    auto a_ms_ks_strides = toCKVec(plan->mContractionDesc.mTensorDesc[0].mStrides);

    auto b_ns_ks_lengths = toCKVec(plan->mContractionDesc.mTensorDesc[1].mLengths);
    auto b_ns_ks_strides = toCKVec(plan->mContractionDesc.mTensorDesc[1].mStrides);

    auto d_ms_ns_lengths = toCKVec(plan->mContractionDesc.mTensorDesc[2].mLengths);
    auto d_ms_ns_strides = toCKVec(plan->mContractionDesc.mTensorDesc[2].mStrides);

    auto e_ms_ns_lengths = toCKVec(plan->mContractionDesc.mTensorDesc[3].mLengths);
    auto e_ms_ns_strides = toCKVec(plan->mContractionDesc.mTensorDesc[3].mStrides);

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

    if(canRun)
    {
        (*cSolution)(StreamConfig{stream, false});
        return HIPTENSOR_STATUS_SUCCESS;
    }
    else
    {
        if(plan->mContractionDesc.mTensorDesc[3].mType == HIP_R_32F)
        {
            if(alpha != nullptr)
                alphaVal = *(static_cast<const float*>(alpha));

            if(beta != nullptr)
                betaVal = *(static_cast<const float*>(beta));

            sprintf(msg,
                    "handle=0x%0*llX, plan=0x%llX, alpha=%.6f, A=0x%llX, B=0x%llX, beta=%.6f,"
                    "C=0x%llX, D=0x%llX, workspace=0x%llX, workspaceSize=0x%04lX, stream=0x%llX "
                    "(HIPTENSOR_STATUS_INTERNAL_ERROR)",
                    2 * (int)sizeof(void*),
                    (unsigned long long)handle,
                    (unsigned long long)plan,
                    (float)alphaVal,
                    (unsigned long long)A,
                    (unsigned long long)B,
                    (float)betaVal,
                    (unsigned long long)C,
                    (unsigned long long)D,
                    (unsigned long long)workspace,
                    (unsigned long)workspaceSize,
                    (unsigned long long)stream);
            logger->logError("hiptensorContraction", msg);
            return HIPTENSOR_STATUS_INTERNAL_ERROR;
        }
        else if(plan->mContractionDesc.mTensorDesc[3].mType == HIP_R_64F)
        {
            if(alpha != nullptr)
                alphaVal = *(static_cast<const double*>(alpha));

            if(beta != nullptr)
                betaVal = *(static_cast<const double*>(beta));

            sprintf(msg,
                    "handle=0x%0*llX, plan=0x%llX, alpha=%.6lf, A=0x%llX, B=0x%llX, beta=%.6lf,"
                    "C=0x%llX, D=0x%llX, workspace=0x%llX, workspaceSize=0x%04lX, stream=0x%llX "
                    "(HIPTENSOR_STATUS_INTERNAL_ERROR)",
                    2 * (int)sizeof(void*),
                    (unsigned long long)handle,
                    (unsigned long long)plan,
                    (double)alphaVal,
                    (unsigned long long)A,
                    (unsigned long long)B,
                    (double)betaVal,
                    (unsigned long long)C,
                    (unsigned long long)D,
                    (unsigned long long)workspace,
                    (unsigned long)workspaceSize,
                    (unsigned long long)stream);
            logger->logError("hiptensorContraction", msg);
            return HIPTENSOR_STATUS_INTERNAL_ERROR;
        }
    }
}
