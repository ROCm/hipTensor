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

#include "types.hpp"

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

} // namespace hiptensor
