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

#ifndef HIPTENSOR_LIBRARY_TYPES_HPP
#define HIPTENSOR_LIBRARY_TYPES_HPP

// clang-format off
// Include order needs to be preserved
#include <hip/library_types.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <iostream>

#include <hiptensor/hiptensor_types.hpp>

// clang-format on

namespace hiptensor
{
    // Used to map to empty tensors
    struct NoneType;

    static constexpr hipDataType NONE_TYPE = (hipDataType)31;

    // Map type to runtime HipDataType
    template <typename T>
    struct HipDataType;

    template <typename T>
    static constexpr auto HipDataType_v = HipDataType<T>::value;

    // Get data size in bytes from id
    uint32_t hipDataTypeSize(hipDataType id);

    // Read a single value from void pointer, casted to T
    template <typename T>
    T readVal(void const* value, hipDataType id);

    template <typename T>
    T readVal(void const* value, hiptensorComputeType_t id);

} // namespace hiptensor

bool operator==(hipDataType hipType, hiptensorComputeType_t computeType);
bool operator==(hiptensorComputeType_t computeType, hipDataType hipType);

bool operator!=(hipDataType hipType, hiptensorComputeType_t computeType);
bool operator!=(hiptensorComputeType_t computeType, hipDataType hipType);

#include "types_impl.hpp"

#endif // HIPTENSOR_LIBRARY_TYPES_HPP
