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

#include "logger.hpp"
#include "permutation_ck.hpp"

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

    if(descA->mType == HIP_R_16F)
    {
        return hiptensor::detail::permuteByCk(alpha,
                                              static_cast<const _Float16*>(A),
                                              descA,
                                              modeA,
                                              static_cast<_Float16*>(B),
                                              descB,
                                              modeB,
                                              typeScalar,
                                              stream);
    }
    else if(descA->mType == HIP_R_32F)
    {
        return hiptensor::detail::permuteByCk(alpha,
                                              static_cast<const float*>(A),
                                              descA,
                                              modeA,
                                              static_cast<float*>(B),
                                              descB,
                                              modeB,
                                              typeScalar,
                                              stream);
    }
    return HIPTENSOR_STATUS_NOT_SUPPORTED;
}
