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

#ifndef HIPTENSOR_SRC_UTIL_HPP
#define HIPTENSOR_SRC_UTIL_HPP

namespace hiptensor
{
    template <typename T>
    static inline std::vector<T> stridesFromLengths(std::vector<T> const& lengths)
    {
        // Re-construct strides from lengths, assuming packed.
        std::vector<std::size_t> strides(lengths.size());
        strides.back() = 1;
        std::partial_sum(
            lengths.rbegin(), lengths.rend() - 1, strides.rbegin() + 1, std::multiplies<T>());
        return strides;
    }

    template <typename T>
    static inline T elementsFromLengths(std::vector<T> const& lengths)
    {
        return std::accumulate(lengths.begin(), lengths.end(), T{1}, std::multiplies<T>());
    }

    template <typename T>
    static inline T elementSpaceFromLengthsAndStrides(std::vector<T> const& lengths,
                                                      std::vector<T> const& strides)
    {
        auto accum = T{1};
        for(int i = 0; i < lengths.size(); i++)
        {
            accum += (lengths[i] - 1) * strides[i];
        }
        return accum;
    }

} // namespace hiptensor

#endif // HIPTENSOR_SRC_UTIL_HPP
