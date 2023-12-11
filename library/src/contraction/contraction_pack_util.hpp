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

#ifndef HIPTENSOR_CONTRACTION_PACK_UTIL_HPP
#define HIPTENSOR_CONTRACTION_PACK_UTIL_HPP

#include "data_types.hpp"
#include "util.hpp"
#include <hiptensor/hiptensor.hpp>

namespace hiptensor
{
    /**
     * \brief This function unpacks structured data (hipFloatComplex / hipDoubleComplex)
     *        into non-structured data (float / double).
     */
    template<typename InputType, typename OutputType>
    __global__ void unpack(const InputType* in, OutputType* out_real, OutputType *out_img, int length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if(idx < length)
        {
            if constexpr(std::is_same_v<InputType,hipFloatComplex>)
            {
                out_real[idx] = hipCrealf(in[idx]);
                out_img[idx] = hipCimagf(in[idx]);
            }
            else if constexpr(std::is_same_v<InputType,hipDoubleComplex>)
            {
                out_real[idx] = hipCreal(in[idx]);
                out_img[idx] = hipCimag(in[idx]);
            }
        }
    }

    /**
     * \brief This function packs non-structured data (float / double)
     *        into structured data (hipFloatComplex / hipDoubleComplex).
     */
    template<typename InputType, typename OutputType>
    __global__ void pack(const InputType* in_real, InputType* in_img, OutputType *out, int length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if(idx < length)
        {
            if constexpr(std::is_same_v<OutputType, hipFloatComplex>)
            {
                out[idx] = make_hipFloatComplex((float)in_real[idx], (float)in_img[idx]);
            }
            else if constexpr(std::is_same_v<OutputType, hipDoubleComplex>)
            {
                out[idx] = make_hipDoubleComplex((double)in_real[idx], (double)in_img[idx]);
            }
        }
    }
 
    struct DeviceDeleter
    {
        void operator()(void* ptr)
        {
            CHECK_HIP_ERROR(hipFree(ptr));
        }
    };

    template<typename T>
    auto allocDevice(int64_t numElements)
    {
        T* data;
        CHECK_HIP_ERROR(hipMalloc(&data, numElements));
        return std::unique_ptr<T, DeviceDeleter>(data, DeviceDeleter());
    }

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_PACK_UTIL_HPP

