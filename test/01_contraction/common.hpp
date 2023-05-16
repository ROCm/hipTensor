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

#ifndef HIPTENSOR_TEST_CONTRACTION_COMMON_HPP
#define HIPTENSOR_TEST_CONTRACTION_COMMON_HPP

#define NDIM 4

template < typename ADataType,
           typename BDataType,
           typename DDataType,
           typename floatTypeCompute>
void hiptensorScaleContractionReference(ADataType *A,
                                        BDataType *B,
                                        DDataType *D,
                                        floatTypeCompute alpha,
                                        std::vector<int64_t> a_ms_ks_lengths,
                                        std::vector<int64_t> b_ks_ns_lengths,
                                        std::vector<int64_t> d_ms_ns_lengths,
                                        std::vector<size_t> a_ms_ks_strides,
                                        std::vector<size_t> b_ks_ns_strides,
                                        std::vector<size_t> d_ms_ns_strides,
                                        int elementsD)
{
    auto d_ms_ns = [&](auto m0, auto m1, auto n0, auto n1)
    {
        floatTypeCompute valA, valB, valC, valAcc = 0;
        size_t indexA, indexB, indexD;

        auto K0 = d_ms_ns_lengths[2];
        auto K1 = d_ms_ns_lengths[3];

        auto offset = [](std::vector<size_t> curIndices, std::vector<size_t> strides) {
                        return std::inner_product(curIndices.begin(), curIndices.end(), strides.begin(), std::size_t{0}); };

        for(size_t k0 = 0; k0 < K0; k0++)
        {
            for(size_t k1 = 0; k1 < K1; k1++)
            {

                indexA = offset(std::vector<size_t>{m0, m1, k0, k1}, a_ms_ks_strides);
                valA = static_cast<floatTypeCompute> (A[indexA]);

                indexB = offset(std::vector<size_t>{n0, n1, k0, k1}, b_ks_ns_strides);
                valB = static_cast<floatTypeCompute> (B[indexB]);

                valAcc += valA * valB;
            }
        }

        valC = alpha * valAcc;

        indexD = offset(std::vector<size_t>{m0, m1, n0, n1}, d_ms_ns_strides);
        D[indexD] = static_cast<DDataType>(valC);
    };

    auto GetNdIndices = [&](size_t index) {
        std::array<std::size_t, NDIM> indices;

        for(std::size_t idim = 0;  idim < NDIM; ++idim)
        {
            indices[idim] = index / d_ms_ns_strides[idim];
            index -= indices[idim] * d_ms_ns_strides[idim];
        }

        return indices;
    };

    for(std::size_t i = 0; i < elementsD; ++i)
    {
        std::array<std::size_t, NDIM> indices = GetNdIndices(i);
        d_ms_ns(indices[0], indices[1], indices[2], indices[3]);
    }
}

template <typename DDataType>
std::pair<bool, double> compareEqual(DDataType const* deviceD,
                                     DDataType const* hostD,
                                     int elementsD,
                                     double       tolerance = 100.0)
{
    bool   retval             = true;
    double max_relative_error = 0.0;

    auto toDouble
        = [](DDataType const& val) { return static_cast<double>(static_cast<float>(val)); };

    bool       isInf = false;
    bool       isNaN = false;
    std::mutex writeMutex;

#pragma omp parallel for
    for(int i = 0; i < elementsD; ++i)
    {
        auto valDevice = deviceD[i];
        auto valHost = hostD[i];

        auto numerator = fabs(toDouble(valDevice) - toDouble(valHost));
        auto divisor   = fabs(toDouble(valDevice)) + fabs(toDouble(valHost)) + 1.0;

        if(std::isinf(numerator) || std::isinf(divisor))
        {
#pragma omp atomic
            isInf |= true;
        }
        else
        {
            auto relative_error = numerator / divisor;
            if(std::isnan(relative_error))
            {
#pragma omp atomic
                isNaN |= true;
            }
            else if(relative_error > max_relative_error)
            {
                const std::lock_guard<std::mutex> guard(writeMutex);
                // Double check in case of stall
                if(relative_error > max_relative_error)
                {
                    max_relative_error = relative_error;
                }
            }
        }

        if(isInf || isNaN)
        {
            i = elementsD;
        }
    }

    auto eps = toDouble(std::numeric_limits<DDataType>::epsilon());
    if(isInf)
    {
        retval             = false;
        max_relative_error = std::numeric_limits<DDataType>::infinity();
    }
    else if(isNaN)
    {
        retval             = false;
        max_relative_error = std::numeric_limits<DDataType>::signaling_NaN();
    }
    else if(max_relative_error > (eps * tolerance))
    {
        retval = false;
    }

    return std::make_pair(retval, max_relative_error);
}

#endif // HIPTENSOR_TEST_CONTRACTION_COMMON_HPP
