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

#include "permutation_solution.hpp"
#include "permutation_solution_instances.hpp"
#include "permutation_solution_registry.hpp"
#include "logger.hpp"

inline auto toPermutationSolutionVec(
    std::unordered_map<std::size_t, hiptensor::PermutationSolution*> const& map)
{
    auto result = std::vector<hiptensor::PermutationSolution*>(map.size());
    transform(map.begin(), map.end(), result.begin(), [](auto p) { return p.second; });
    return result;
}

hiptensorStatus_t hiptensorPermutation(const hiptensorHandle_t*           handle,
                                       const void*                        alpha,
                                       const void*                        A,
                                       const hiptensorTensorDescriptor_t* descA,
                                       const int32_t                      modeA[],
                                       void*                              B,
                                       const hiptensorTensorDescriptor_t* descB,
                                       const int32_t                      modeB[],
                                       const hipDataType                  typeScalar,
                                       const hipStream_t                  stream)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[2048];
    snprintf(msg,
             sizeof(msg),
             "handle=%p, alpha=%p, A=%p, descA=%p, modeA=%p, B=%p, descB=%p, modeB=%p, "
             "typeScalar=0x%02X, stream=%p",
             handle,
             alpha,
             A,
             descA,
             modeA,
             B,
             descB,
             modeB,
             (unsigned int)typeScalar,
             stream);

    logger->logAPITrace("hiptensorPermutation", msg);

    if(!handle || !alpha || !A || !descA || !modeA || !B || !descB || !modeB)
    {
        auto errorCode         = HIPTENSOR_STATUS_NOT_INITIALIZED;
        auto printErrorMessage = [&logger, errorCode](const std::string& paramName) {
            char msg[512];
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : %s = nullptr (%s)",
                     paramName.c_str(),
                     hiptensorGetErrorString(errorCode));
            logger->logError("hiptensorPermutation", msg);
        };
        if(!handle)
        {
            printErrorMessage("handle");
        }
        if(!alpha)
        {
            printErrorMessage("alpha");
        }
        if(!A)
        {
            printErrorMessage("A");
        }
        if(!descA)
        {
            printErrorMessage("descA");
        }
        if(!modeA)
        {
            printErrorMessage("modeA");
        }
        if(!B)
        {
            printErrorMessage("B");
        }
        if(!descB)
        {
            printErrorMessage("descB");
        }
        if(!modeB)
        {
            printErrorMessage("modeB");
        }
        return errorCode;
    }

    if(descA->mType != HIP_R_16F && descA->mType != HIP_R_32F)
    {
        auto errorCode = HIPTENSOR_STATUS_NOT_SUPPORTED;
        snprintf(msg,
                 sizeof(msg),
                 "Unsupported Data Type Error : The supported data types of A and B are HIP_R_16F "
                 "and HIP_R_32F (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorPermutation", msg);
        return errorCode;
    }

    if(descA->mType != descB->mType)
    {
        auto errorCode = HIPTENSOR_STATUS_INVALID_VALUE;
        snprintf(msg,
                 sizeof(msg),
                 "Mismatched Data Type Error : Data types of A and B are not the same. (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorPermutation", msg);
        return errorCode;
    }

    if(typeScalar != HIP_R_16F && typeScalar != HIP_R_32F)
    {
        auto errorCode = HIPTENSOR_STATUS_NOT_SUPPORTED;
        snprintf(msg,
                 sizeof(msg),
                 "Unsupported Data Type Error : The supported data types of alpha are HIP_R_16F "
                 "and HIP_R_32F (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorPermutation", msg);
        return errorCode;
    }

    // ADDED
    // For now, enumerate all known permutation kernels.
    auto& instances = hiptensor::PermutationSolutionInstances::instance();
    auto  solnQ     = instances->allSolutions();

    if(solnQ.solutionCount() == 0)
    {
        // No kernels found!
        auto errorCode = HIPTENSOR_STATUS_INTERNAL_ERROR;
        snprintf(msg,
                 sizeof(msg),
                 "Internal Error : No Kernels Found (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorPermutation", msg);
        return errorCode;
    }

    // Extract the solutions to the candidates vector.
    auto candidates = toPermutationSolutionVec(solnQ.solutions());

    int  nDims              = descA->mLengths.size();
    auto ADataType          = descA->mType;
    auto BDataType          = descB->mType;
    auto AOp                = descA->mUnaryOp;
    auto BOp                = descB->mUnaryOp;

    // Query permutation solutions for the correct permutation operation and type
    auto solutionQ = hiptensor::PermutationSolutionRegistry::Query{candidates}
                         .query(nDims)
                         .query(ADataType, BDataType)
                         .query(AOp, BOp);

    if(solutionQ.solutionCount() == 0)
    {
        // No kernels found!
        auto errorCode = HIPTENSOR_STATUS_INTERNAL_ERROR;
        snprintf(msg,
                 sizeof(msg),
                 "Internal Error : No Kernels Found (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorPermutation", msg);
        return errorCode;
    }

    candidates = toPermutationSolutionVec(solutionQ.solutions());

    hiptensor::PermutationSolution *pSolution = candidates[0];

    auto canRun = pSolution->initArgs(alpha,
                                      A,
                                      B,
                                      descA->mLengths,
                                      descA->mStrides,
                                      modeA,
                                      descB->mLengths,
                                      descB->mStrides,
                                      modeB,
                                      typeScalar);

    if(canRun)
    {
        // Perform permutation with timing if LOG_LEVEL_PERF_TRACE
        if(logger->getLogMask() & HIPTENSOR_LOG_LEVEL_PERF_TRACE)
        {
            auto time = (*pSolution)(StreamConfig{
                stream, // stream id
                true, // time_kernel
                0, // log_level
                0, // cold_niters
                1, // nrepeat
            });
            if(time < 0)
            {
                return HIPTENSOR_STATUS_CK_ERROR;
            }

            int n             = pSolution->problemDim();
            auto flops        = std::size_t(2) * n;
            auto bytes        = pSolution->problemBytes();

            hiptensor::PerfMetrics metrics = {
                pSolution->uid(), // id
                pSolution->kernelName(), // name
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
            logger->logPerformanceTrace("hiptensorPermutation", msg);
        }
        // Perform permutation without timing
        else
        {
            if((*pSolution)(StreamConfig{stream, false}) < 0)
            {
                return HIPTENSOR_STATUS_CK_ERROR;
            }
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
        logger->logError("hiptensorPermutation", msg);
        return errorCode;
    }

            // if(descA->mType == HIP_R_16F)
            // {
            //     return hiptensor::detail::permuteByCk(alpha,
            //                                         static_cast<const _Float16*>(A),
            //                                         descA,
            //                                         modeA,
            //                                         static_cast<_Float16*>(B),
            //                                         descB,
            //                                         modeB,
            //                                         typeScalar,
            //                                         stream);
            // }
            // else if(descA->mType == HIP_R_32F)
            // {
            //     return hiptensor::detail::permuteByCk(alpha,
            //                                         static_cast<const float*>(A),
            //                                         descA,
            //                                         modeA,
            //                                         static_cast<float*>(B),
            //                                         descB,
            //                                         modeB,
            //                                         typeScalar,
            //                                         stream);
            // }
}
