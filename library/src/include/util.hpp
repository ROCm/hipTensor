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

#ifndef HIPTENSOR_SRC_UTIL_HPP
#define HIPTENSOR_SRC_UTIL_HPP

#include <type_traits>
#include <vector>

namespace hiptensor
{
    template <typename intT1,
              class = typename std::enable_if<std::is_integral<intT1>::value>::type,
              typename intT2,
              class = typename std::enable_if<std::is_integral<intT2>::value>::type>
    static constexpr intT1 ceilDiv(const intT1 numerator, const intT2 divisor)
    {
        return (numerator + divisor - 1) / divisor;
    }

    template <typename T>
    static inline std::vector<T> stridesFromLengths(std::vector<T> const& lengths,
                                                    bool                  col_major = false)
    {
        if(lengths.empty())
        {
            return lengths;
        }

        // Re-construct strides from lengths, assuming packed.
        std::vector<T> strides(lengths.size(), 1);
        if(!col_major)
        {
            strides.back() = 1;
            std::partial_sum(
                lengths.rbegin(), lengths.rend() - 1, strides.rbegin() + 1, std::multiplies<T>());
        }
        else
        {
            strides.front() = 1;
            std::partial_sum(
                lengths.begin(), lengths.end() - 1, strides.begin() + 1, std::multiplies<T>());
        }

        return strides;
    }

    // Get count of element of a tensor. Note that the count is 1 if the rank of tensor is 0.
    template <typename T>
    static inline T elementsFromLengths(std::vector<T> const& lengths)
    {
        return std::accumulate(lengths.begin(), lengths.end(), T{1}, std::multiplies<T>());
    }
} // namespace hiptensor

#endif // HIPTENSOR_SRC_UTIL_HPP
