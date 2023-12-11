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

#include "data_types.hpp"

namespace hiptensor
{
    // Get data size in bytes from id
    uint32_t hipDataTypeSize(hipDataType id)
    {
        if(id == HIP_R_16BF)
        {
            return sizeof(hip_bfloat16);
        }
        else if(id == HIP_R_16F)
        {
            return sizeof(_Float16);
        }
        else if(id == HIP_R_32F)
        {
            return sizeof(float);
        }
        else if(id == HIP_R_64F)
        {
            return sizeof(double);
        }
        else if(id == HIP_R_8I)
        {
            return sizeof(int8_t);
        }
        else if(id == HIP_R_8U)
        {
            return sizeof(uint8_t);
        }
        else if(id == HIP_R_16I)
        {
            return sizeof(int16_t);
        }
        else if(id == HIP_R_16U)
        {
            return sizeof(uint16_t);
        }
        else if(id == HIP_R_32I)
        {
            return sizeof(int32_t);
        }
        else if(id == HIP_R_32U)
        {
            return sizeof(uint32_t);
        }
        else if(id == HIP_R_64I)
        {
            return sizeof(int64_t);
        }
        else if(id == HIP_R_64U)
        {
            return sizeof(uint64_t);
        }
        else if(id == HIP_C_32F)
        {
            return sizeof(hipFloatComplex);
        }
        else if(id == HIP_C_64F)
        {
            return sizeof(hipDoubleComplex);
        }
        else if(id == NONE_TYPE)
        {
            return 0;
        }
        else
        {
#if !NDEBUG
            std::cout << "Unhandled hip datatype: " << id << std::endl;
#endif // !NDEBUG
            return 0;
        }
    }

    hiptensorComputeType_t convertToComputeType(hipDataType hipType)
    {
        if(hipType == HIP_R_16BF)
        {
            return HIPTENSOR_COMPUTE_16BF;
        }
        else if(hipType == HIP_R_16F)
        {
            return HIPTENSOR_COMPUTE_16F;
        }
        else if(hipType == HIP_R_32F || hipType == HIP_C_32F)
        {
            return HIPTENSOR_COMPUTE_32F;
        }
        else if(hipType == HIP_R_64F || hipType == HIP_C_64F)
        {
            return HIPTENSOR_COMPUTE_64F;
        }
        else if(hipType == HIP_R_8I)
        {
            return HIPTENSOR_COMPUTE_8I;
        }
        else if(hipType == HIP_R_8U)
        {
            return HIPTENSOR_COMPUTE_8U;
        }
        else if(hipType == HIP_R_32I)
        {
            return HIPTENSOR_COMPUTE_32I;
        }
        else if(hipType == HIP_R_32U)
        {
            return HIPTENSOR_COMPUTE_32U;
        }
        else
        {
            return HIPTENSOR_COMPUTE_NONE;
        }
    }

    void writeVal(void const* addr, hiptensorComputeType_t id, double value)
    {
        if(id == HIPTENSOR_COMPUTE_16F)
        {
            *(_Float16*)addr = value;
        }
        else if(id == HIPTENSOR_COMPUTE_16BF)
        {
            *(hip_bfloat16*)addr = value;
        }
        else if(id == HIPTENSOR_COMPUTE_32F)
        {
            *(float*)addr = value;
        }
        else if(id == HIPTENSOR_COMPUTE_64F)
        {
            *(double*)addr = value;
        }
        else if(id == HIPTENSOR_COMPUTE_8U)
        {
            *(uint8_t*)addr = value;
        }
        else if(id == HIPTENSOR_COMPUTE_8I)
        {
            *(int8_t*)addr = value;
        }
        else if(id == HIPTENSOR_COMPUTE_32U)
        {
            *(uint32_t*)addr = value;
        }
        else if(id == HIPTENSOR_COMPUTE_32I)
        {
            *(int32_t*)addr = value;
        }
        else
        {
#if !NDEBUG
            std::cout << "Unhandled hiptensorComputeType_t: " << id << std::endl;
#endif // !NDEBUG
            return;
        }
    }

} // namespace hiptensor

bool operator==(hipDataType hipType, hiptensorComputeType_t computeType)
{
    if(hipType == HIP_R_16BF)
    {
        return (computeType == HIPTENSOR_COMPUTE_16BF);
    }
    else if(hipType == HIP_R_16F)
    {
        return (computeType == HIPTENSOR_COMPUTE_16F);
    }
    else if(hipType == HIP_R_32F || hipType == HIP_C_32F)
    {
        return (computeType == HIPTENSOR_COMPUTE_32F);
    }
    else if(hipType == HIP_R_64F || hipType == HIP_C_64F)
    {
        return (computeType == HIPTENSOR_COMPUTE_64F);
    }
    else if(hipType == HIP_R_8I)
    {
        return (computeType == HIPTENSOR_COMPUTE_8I);
    }
    else if(hipType == HIP_R_8U)
    {
        return (computeType == HIPTENSOR_COMPUTE_8U);
    }
    else if(hipType == HIP_R_16I)
    {
        return false;
    }
    else if(hipType == HIP_R_16U)
    {
        return false;
    }
    else if(hipType == HIP_R_32I)
    {
        return (computeType == HIPTENSOR_COMPUTE_32I);
    }
    else if(hipType == HIP_R_32U)
    {
        return (computeType == HIPTENSOR_COMPUTE_32U);
    }
    else if(hipType == HIP_R_64I)
    {
        return false;
    }
    else if(hipType == HIP_R_64U)
    {
        return false;
    }
    else
    {
#if !NDEBUG
        std::cout << "Unhandled hip datatype: " << hipType << std::endl;
#endif // !NDEBUG
        return false;
    }
}

bool operator==(hiptensorComputeType_t computeType, hipDataType hipType)
{
    return hipType == computeType;
}

bool operator!=(hipDataType hipType, hiptensorComputeType_t computeType)
{
    return !(hipType == computeType);
}

bool operator!=(hiptensorComputeType_t computeType, hipDataType hipType)
{
    return !(computeType == hipType);
}
