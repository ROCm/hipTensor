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
#include "ht_tensor.hpp"
#include "ht_tensor_generator_utility.hpp"

#include "../include/logger.hpp"

hiptensorStatus_t hiptensorInit(hiptensorHandle_t* handle)
{
    if(!handle)
        return HIPTENSOR_STATUS_NOT_INITIALIZED;

    else if(hipInit(0) == hipErrorInvalidDevice)
        return HIPTENSOR_STATUS_HIP_ERROR;

    else if(hipInit(0) == hipErrorInvalidValue)
        return HIPTENSOR_STATUS_INVALID_VALUE;

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorInitTensorDescriptor(const hiptensorHandle_t*     handle,
                                                hiptensorTensorDescriptor_t* desc,
                                                const uint32_t               numModes,
                                                const int64_t                lens[],
                                                const int64_t                strides[],
                                                hiptensorDataType_t          dataType,
                                                hiptensorOperator_t          unaryOp)
{
    if(!handle || !desc)
        return HIPTENSOR_STATUS_NOT_INITIALIZED;

    if(((!lens) && (!strides)) || dataType != HIPTENSOR_R_32F || unaryOp != HIPTENSOR_OP_IDENTITY)
        return HIPTENSOR_STATUS_INVALID_VALUE;

    using descType = float;
    int ht_index   = 0;

    std::vector<std::int64_t> ht_lens;
    std::vector<std::int64_t> ht_strides;

    for(int index = 0; index < numModes; index++)
    {
        ht_lens.push_back(lens[index]);
        if(strides)
            ht_strides.push_back(strides[index]);
    }
    if(!strides)
        *desc
            = hiptensorTensorDescriptor_t(std::vector<std::size_t>(ht_lens.begin(), ht_lens.end()));
    else
        *desc = hiptensorTensorDescriptor_t(
            std::vector<std::size_t>(ht_lens.begin(), ht_lens.end()),
            std::vector<std::size_t>(ht_strides.begin(), ht_strides.end()));
    desc->ht_type = dataType;

    return HIPTENSOR_STATUS_SUCCESS;
}

const char* hiptensorGetErrorString(const hiptensorStatus_t error)
{
    if(error == HIPTENSOR_STATUS_SUCCESS)
        return "HIPTENSOR_STATUS_SUCCESS";
    else if(error == HIPTENSOR_STATUS_NOT_INITIALIZED)
        return "HIPTENSOR_STATUS_NOT_INITIALIZED";
    else if(error == HIPTENSOR_STATUS_ALLOC_FAILED)
        return "HIPTENSOR_STATUS_ALLOC_FAILED";
    else if(error == HIPTENSOR_STATUS_INVALID_VALUE)
        return "HIPTENSOR_STATUS_INVALID_VALUE";
    else if(error == HIPTENSOR_STATUS_ARCH_MISMATCH)
        return "HIPTENSOR_STATUS_ARCH_MISMATCH";
    else if(error == HIPTENSOR_STATUS_EXECUTION_FAILED)
        return "HIPTENSOR_STATUS_EXECUTION_FAILED";
    else if(error == HIPTENSOR_STATUS_INTERNAL_ERROR)
        return "HIPTENSOR_STATUS_INTERNAL_ERROR";
    else if(error == HIPTENSOR_STATUS_NOT_SUPPORTED)
        return "HIPTENSOR_STATUS_NOT_SUPPORTED";
    else if(error == HIPTENSOR_STATUS_CK_ERROR)
        return "HIPTENSOR_STATUS_CK_ERROR";
    else if(error == HIPTENSOR_STATUS_HIP_ERROR)
        return "HIPTENSOR_STATUS_HIP_ERROR";
    else if(error == HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE)
        return "HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE";
    else if(error == HIPTENSOR_STATUS_INSUFFICIENT_DRIVER)
        return "HIPTENSOR_STATUS_INSUFFICIENT_DRIVER";
    else if(error == HIPTENSOR_STATUS_IO_ERROR)
        return "HIPTENSOR_STATUS_IO_ERROR";
    else
        return "HIPTENSOR_STATUS_UNKNOWN";
}

hiptensorStatus_t hiptensorGetAlignmentRequirement(const hiptensorHandle_t*           handle,
                                                   const void*                        ptr,
                                                   const hiptensorTensorDescriptor_t* desc,
                                                   uint32_t* alignmentRequirement)
{
    if(!handle || !desc)
        return HIPTENSOR_STATUS_NOT_INITIALIZED;

    if(desc->ht_type != HIPTENSOR_R_32F)
        return HIPTENSOR_STATUS_INVALID_VALUE;

    using descType = float;

    *alignmentRequirement = sizeof(descType) * desc->hiptensorGetElementSpace();

    return HIPTENSOR_STATUS_SUCCESS;
}

void hiptensorContractionDescriptor_t::hiptensorContractionAttrUpdate(
    const hiptensorTensorDescriptor_t* desc[],
    const uint32_t                     tensor_size[],
    const int                          tensor_desc_num)
{
    for(int index = 0; index < tensor_desc_num; index++)
    {
        ht_contract_attr_desc.push_back({desc[index]->hiptensorGetLengths(),
                                         desc[index]->hiptensorGetStrides(),
                                         tensor_size[index]});
    }
    return;
}

void hiptensorTensorDescriptor_t::hiptensorCalculateStrides()
{
    mStrides.clear();
    mStrides.resize(mLens.size(), 0);
    if(mStrides.empty())
        return;

    mStrides.back() = 1;
    std::partial_sum(
        mLens.rbegin(), mLens.rend() - 1, mStrides.rbegin() + 1, std::multiplies<std::size_t>());
}

std::size_t hiptensorTensorDescriptor_t::hiptensorGetNumOfDimension() const
{
    return mLens.size();
}

std::size_t hiptensorTensorDescriptor_t::hiptensorGetElementSize() const
{
    assert(mLens.size() == mStrides.size());
    return std::accumulate(
        mLens.begin(), mLens.end(), std::size_t{1}, std::multiplies<std::size_t>());
}

std::size_t hiptensorTensorDescriptor_t::hiptensorGetElementSpace() const
{
    std::size_t space = 1;
    for(std::size_t i = 0; i < mLens.size(); ++i)
    {
        space += (mLens[i] - 1) * mStrides[i];
    }
    return space;
}

const std::vector<std::size_t>& hiptensorTensorDescriptor_t::hiptensorGetLengths() const
{
    return mLens;
}

const std::vector<std::size_t>& hiptensorTensorDescriptor_t::hiptensorGetStrides() const
{
    return mStrides;
}

std::ostream& operator<<(std::ostream& os, const hiptensorTensorDescriptor_t& desc)
{
    os << "dim " << desc.hiptensorGetNumOfDimension() << ", ";

    os << "lengths {";
    hiptensorPrintVectorElements(desc.hiptensorGetLengths(), ", ");
    os << "}, ";

    os << "strides {";
    hiptensorPrintVectorElements(desc.hiptensorGetStrides(), ", ");
    os << "}";

    return os;

    void printHexAddress(char* str, void const* obj)
    {
        // Format string as hex
        // Width in hex = 8 Byte * 2 = 16
        // Cast obj to
        sprintf(str, "0x%0*llX", 2 * (int)sizeof(void*), (unsigned long long)obj);
    }

    hiptensorStatus_t hiptensorLoggerSetCallback(hiptensorLoggerCallback_t callback)
    {
        using hiptensor::Logger;
        auto& logger = Logger::instance();

        // Log API access
        char msg[64];
        sprintf(msg, "callback=0x%0*llX", 2 * (int)sizeof(void*), (unsigned long long)callback);
        logger->logMessage(HIPTENSOR_LOG_LEVEL_API_TRACE, "hiptensorLoggerSetCallback", msg);

        // Check logger callback result
        auto loggerResult = logger->setCallback(callback);
        if(loggerResult != Logger::Status_t::SUCCESS)
        {
            sprintf(msg,
                    "callback=0x%0*llX (%s)",
                    2 * (int)sizeof(void*),
                    (unsigned long long)callback,
                    logger->statusString(loggerResult));
            logger->logMessage(HIPTENSOR_LOG_LEVEL_ERROR, "hiptensorLoggerSetCallback", msg);
            return HIPTENSOR_STATUS_INVALID_VALUE;
        }

        return HIPTENSOR_STATUS_SUCCESS;
    }

    hiptensorStatus_t hiptensorLoggerSetFile(FILE * file)
    {
        using hiptensor::Logger;
        auto& logger = Logger::instance();

        // Log API access
        char msg[64];
        sprintf(msg, "file=0x%0*llX", 2 * (int)sizeof(void*), (unsigned long long)file);
        logger->logMessage(HIPTENSOR_LOG_LEVEL_API_TRACE, "hiptensorLoggerSetFile", msg);

        // Check logger callback result
        auto loggerResult = logger->writeToStream(file);
        if(loggerResult != Logger::Status_t::SUCCESS)
        {
            sprintf(msg,
                    "file=0x%0*llX (%s)",
                    2 * (int)sizeof(void*),
                    (unsigned long long)file,
                    logger->statusString(loggerResult));
            logger->logMessage(HIPTENSOR_LOG_LEVEL_ERROR, "hiptensorLoggerSetFile", msg);
            return HIPTENSOR_STATUS_IO_ERROR;
        }

        return HIPTENSOR_STATUS_SUCCESS;
    }

    hiptensorStatus_t hiptensorLoggerOpenFile(const char* logFile)
    {
        using hiptensor::Logger;
        auto& logger = Logger::instance();

        // Log API trace
        char msg[2048];
        sprintf(msg, "logFile=%s", logFile);
        logger->logMessage(HIPTENSOR_LOG_LEVEL_API_TRACE, "hiptensorLoggerOpenFile", msg);

        // Check logger open file result
        auto loggerResult = logger->openFileStream(logFile);
        if(loggerResult != Logger::Status_t::SUCCESS)
        {
            sprintf(msg, "fileName=%s (%s)", logFile, logger->statusString(loggerResult));
            logger->logMessage(HIPTENSOR_LOG_LEVEL_ERROR, "hiptensorLoggerOpenFile", msg);
            return HIPTENSOR_STATUS_IO_ERROR;
        }

        return HIPTENSOR_STATUS_SUCCESS;
    }

    hiptensorStatus_t hiptensorLoggerSetLevel(hiptensorLogLevel_t level)
    {
        using hiptensor::Logger;
        auto& logger = Logger::instance();

        // Log API trace
        char msg[64];
        sprintf(msg, "log level=0x%02X", (unsigned int)level);
        logger->logMessage(HIPTENSOR_LOG_LEVEL_API_TRACE, "hiptensorLoggerSetLevel", msg);

        // Check logger level
        auto loggerResult = logger->setLogLevel(Logger::LogLevel_t(level));
        if(loggerResult != Logger::Status_t::SUCCESS)
        {
            sprintf(
                msg, "level=0x%02X (%s)", (unsigned int)level, logger->statusString(loggerResult));
            logger->logMessage(HIPTENSOR_LOG_LEVEL_ERROR, "hiptensorLoggerSetLevel", msg);
            return HIPTENSOR_STATUS_INVALID_VALUE;
        }

        return HIPTENSOR_STATUS_SUCCESS;
    }

    hiptensorStatus_t hiptensorLoggerSetMask(int32_t mask)
    {
        using hiptensor::Logger;
        auto& logger = Logger::instance();

        // Log API trace
        char msg[64];
        sprintf(msg, "mask=0x%02X", (unsigned int)mask);
        logger->logMessage(HIPTENSOR_LOG_LEVEL_API_TRACE, "hiptensorLoggerSetMask", msg);

        // Check for logger error
        auto loggerResult = logger->setLogMask(mask);
        if(loggerResult != Logger::Status_t::SUCCESS)
        {
            sprintf(
                msg, "mask=0x%02X (%s)", (unsigned int)mask, logger->statusString(loggerResult));
            logger->logMessage(HIPTENSOR_LOG_LEVEL_ERROR, "hiptensorLoggerSetMask", msg);
            return HIPTENSOR_STATUS_INVALID_VALUE;
        }

        return HIPTENSOR_STATUS_SUCCESS;
    }

    hiptensorStatus_t hiptensorLoggerForceDisable()
    {
        // Log API trace
        auto& logger = hiptensor::Logger::instance();
        logger->logMessage(
            HIPTENSOR_LOG_LEVEL_API_TRACE, "hiptensorLoggerForceDisable", "Logging Disabled");
        logger->disable();
        return HIPTENSOR_STATUS_SUCCESS;
    }
