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
#ifndef HIPTENSOR_UTILITY_INTERNAL_HPP
#define HIPTENSOR_UTILITY_INTERNAL_HPP

#include <fstream>
#include <hip/hip_runtime.h>
#include <iostream>

#include "../hiptensor_types.hpp"

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(status)                   \
    if(status != hipSuccess)                      \
    {                                             \
        fprintf(stderr,                           \
                "hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(status),        \
                status,                           \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPTENSOR_ERROR
#define CHECK_HIPTENSOR_ERROR(status)                   \
    if(status != HIPTENSOR_STATUS_SUCCESS)              \
    {                                                   \
        fprintf(stderr,                                 \
                "hipTensor error: '%s'(%d) at %s:%d\n", \
                hiptensorGetErrorString(status),        \
                status,                                 \
                __FILE__,                               \
                __LINE__);                              \
        exit(EXIT_FAILURE);                             \
    }
#endif

template <typename T>
void hiptensorPrintArrayElements(T* vec, size_t size)
{
    int index = 0;
    while(index != size)
    {
        if(index == size - 1)
        {
            std::cout << vec[index];
        }
        else
            std::cout << vec[index] << ", ";

        index++;
    }
}

template <typename S>
void hiptensorPrintVectorElements(const std::vector<S>& vec, std::string sep = " ")
{
    for(auto& elem : vec)
    {
        std::cout << elem;
        if(&elem != &vec.back())
        {
            std::cout << sep;
        }
    }
}

template <typename F>
void hiptensorPrintElementsToFile(std::ofstream& fs, F* output, size_t size, std::string sep = " ")
{
    if(!fs.is_open())
    {
        std::cout << "File not found!\n";
        return;
    }

    for(int i = 0; i < size; i++)
    {
        if(i == size - 1)
        {
            fs << static_cast<F>(output[i]);
        }
        else
            fs << static_cast<F>(output[i]) << sep;
    }
    return;
}

namespace std
{
    static ostream& operator<<(ostream& os, const hiptensorTensorDescriptor_t& desc)
    {
        os << "dim " << desc.mLengths.size() << ", ";

        os << "lengths {";
        hiptensorPrintVectorElements(desc.mLengths, ", ");
        os << "}, ";

        os << "strides {";
        hiptensorPrintVectorElements(desc.mStrides, ", ");
        os << "}";

        return os;
    }
}

#endif // HIPTENSOR_UTILITY_INTERNAL_HPP
