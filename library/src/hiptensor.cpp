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
#include <hip/hip_runtime_api.h>

#include <hiptensor/hiptensor.hpp>

#include "handle.hpp"
#include "logger.hpp"
#include "types.hpp"
#include "util.hpp"

hiptensorStatus_t hiptensorCreate(hiptensorHandle_t** handle)
{
    (*handle) = new hiptensorHandle_t;

    if(*handle == nullptr)
    {
        return HIPTENSOR_STATUS_ALLOC_FAILED;
    }

    auto hip_status = hipInit(0);

    if(hip_status == hipErrorInvalidDevice)
    {
        return HIPTENSOR_STATUS_HIP_ERROR;
    }

    else if(hip_status == hipErrorInvalidValue)
    {
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    // Get the current device (handled by the Handle class)
    auto realHandle = hiptensor::Handle::createHandle((*handle)->fields);

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorDestroy(hiptensorHandle_t* handle)
{
    hiptensor::Handle::destroyHandle(handle->fields);

    delete handle;
    handle = nullptr;

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorInitTensorDescriptor(const hiptensorHandle_t*     handle,
                                                hiptensorTensorDescriptor_t* desc,
                                                const uint32_t               numModes,
                                                const int64_t                lens[],
                                                const int64_t                strides[],
                                                hipDataType                  dataType,
                                                hiptensorOperator_t          unaryOp)
{
    if(handle == nullptr || desc == nullptr)
    {
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    if((lens == nullptr) || ((dataType != HIP_R_32F) && (dataType != HIP_R_64F))
       || unaryOp != HIPTENSOR_OP_IDENTITY)
    {
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    // todo: check for f64 support (dataType == HIP_R_64F && isSupported)
    // return INVALID_ARCH if not supported

    if(strides)
    {
        // Construct with both given lengths and strides
        *desc = {dataType,
                 std::vector<std::size_t>(lens, lens + numModes),
                 std::vector<std::size_t>(strides, strides + numModes)};
    }
    else
    {
        // Re-construct strides from lengths, assuming packed.
        std::vector<std::size_t> l(lens, lens + numModes);
        std::vector<std::size_t> s = hiptensor::stridesFromLengths(l);

        *desc = {dataType, l, s};
    }

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
    {
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    *alignmentRequirement = 0u;
    for(auto i = hiptensor::hipDataTypeSize(desc->mType); i <= 16u; i *= 2)
    {
        if((std::size_t)ptr % (std::size_t)i == 0)
        {
            *alignmentRequirement = i;
        }
    }

    if(*alignmentRequirement == 0)
    {
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }
    else
    {
        return HIPTENSOR_STATUS_SUCCESS;
    }
}

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
    logger->logAPITrace("hiptensorLoggerSetCallback", msg);

    // Check logger callback result
    auto loggerResult = logger->setCallback(callback);
    if(loggerResult != Logger::Status_t::SUCCESS)
    {
        sprintf(msg,
                "callback=0x%0*llX (%s)",
                2 * (int)sizeof(void*),
                (unsigned long long)callback,
                logger->statusString(loggerResult));
        logger->logError("hiptensorLoggerSetCallback", msg);
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorLoggerSetFile(FILE* file)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[64];
    sprintf(msg, "file=0x%0*llX", 2 * (int)sizeof(void*), (unsigned long long)file);
    logger->logAPITrace("hiptensorLoggerSetFile", msg);

    // Check logger callback result
    auto loggerResult = logger->writeToStream(file);
    if(loggerResult != Logger::Status_t::SUCCESS)
    {
        sprintf(msg,
                "file=0x%0*llX (%s)",
                2 * (int)sizeof(void*),
                (unsigned long long)file,
                logger->statusString(loggerResult));
        logger->logError("hiptensorLoggerSetFile", msg);
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
    logger->logAPITrace("hiptensorLoggerOpenFile", msg);

    // Check logger open file result
    auto loggerResult = logger->openFileStream(logFile);
    if(loggerResult != Logger::Status_t::SUCCESS)
    {
        sprintf(msg, "fileName=%s (%s)", logFile, logger->statusString(loggerResult));
        logger->logError("hiptensorLoggerOpenFile", msg);
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
    logger->logAPITrace("hiptensorLoggerSetLevel", msg);

    // Check logger level
    auto loggerResult = logger->setLogLevel(Logger::LogLevel_t(level));
    if(loggerResult != Logger::Status_t::SUCCESS)
    {
        sprintf(msg, "level=0x%02X (%s)", (unsigned int)level, logger->statusString(loggerResult));
        logger->logError("hiptensorLoggerSetLevel", msg);
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
    logger->logAPITrace("hiptensorLoggerSetMask", msg);

    // Check for logger error
    auto loggerResult = logger->setLogMask(mask);
    if(loggerResult != Logger::Status_t::SUCCESS)
    {
        sprintf(msg, "mask=0x%02X (%s)", (unsigned int)mask, logger->statusString(loggerResult));
        logger->logError("hiptensorLoggerSetMask", msg);
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorLoggerForceDisable()
{
    // Log API trace
    auto& logger = hiptensor::Logger::instance();
    logger->logAPITrace("hiptensorLoggerForceDisable", "Logging Disabled");
    logger->disable();
    return HIPTENSOR_STATUS_SUCCESS;
}

int hiptensorGetHiprtVersion()
{
    // Log API trace
    auto& logger = hiptensor::Logger::instance();
    logger->logAPITrace("hiptensorGetHiprtVersion", "");

    int  version   = 0;
    auto hipResult = hipRuntimeGetVersion(&version);
    if(hipResult != hipError_t::hipSuccess)
    {
        char msg[256];
        sprintf(msg, "Hip error: (%s)", hipGetErrorString(hipResult));
        logger->logError("hiptensorGetHiprtVersion", msg);
        return -1;
    }

    return version;
}
