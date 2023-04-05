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
#pragma once
#include <fstream>
#if 0
template <typename T>
struct GeneratorTensor_cuTensor
{
    template <typename... Is>
    T operator()(Is...)
    {
        float tmp = ((float(std::rand()))/float(RAND_MAX))*2;
        return static_cast<T>(tmp);
    }
};


template <typename T, typename Range>
void LogRangeToFile(std::ofstream& fs, Range&& range, std::string delim)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            fs << delim;
        fs << static_cast<T>(v);
    }
    return;
}
#endif
