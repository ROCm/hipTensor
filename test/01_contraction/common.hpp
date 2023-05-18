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

#include <algorithm>
#include <fstream>
#include <iterator>
#include <math.h>
#include <mutex>
#include <numeric>
#include <unordered_map>

// hiptensor includes
#include <hiptensor/hiptensor.hpp>
#include <hiptensor/hiptensor_types.hpp>
#include <hiptensor/internal/hiptensor_utility.hpp>

#include "device/common.hpp"

#define NDIM 4

template <typename intT1,
          class = typename std::enable_if<std::is_integral<intT1>::value>::type,
          typename intT2,
          class = typename std::enable_if<std::is_integral<intT2>::value>::type>
static constexpr intT1 ceilDiv(const intT1 numerator, const intT2 divisor)
{
    return (numerator + divisor - 1) / divisor;
}

struct joinable_thread : std::thread
{
    template <typename... Xs>
    joinable_thread(Xs&&... xs)
        : std::thread(std::forward<Xs>(xs)...)
    {
    }

    joinable_thread(joinable_thread&&)            = default;
    joinable_thread& operator=(joinable_thread&&) = default;

    ~joinable_thread()
    {
        if(this->joinable())
            this->join();
    }
};

template <typename ADataType, typename BDataType, typename DDataType, typename floatTypeCompute>
void hiptensorScaleContractionReference(ADataType*          A,
                                        BDataType*          B,
                                        DDataType*          D,
                                        floatTypeCompute    alpha,
                                        std::vector<size_t> a_ms_ks_lengths,
                                        std::vector<size_t> b_ks_ns_lengths,
                                        std::vector<size_t> d_ms_ns_lengths,
                                        std::vector<size_t> a_ms_ks_strides,
                                        std::vector<size_t> b_ks_ns_strides,
                                        std::vector<size_t> d_ms_ns_strides,
                                        std::size_t         elementsD,
                                        std::size_t         num_thread = 1)
{
    auto d_ms_ns = [&](auto m0, auto m1, auto n0, auto n1) {
        floatTypeCompute valA, valB, valC, valAcc = 0;
        size_t           indexA, indexB, indexD;

        auto K0 = a_ms_ks_lengths[2];
        auto K1 = a_ms_ks_lengths[3];

        auto offset = [&](std::vector<size_t> curIndices, std::vector<size_t> strides) {
            return std::inner_product(
                curIndices.begin(), curIndices.end(), strides.begin(), std::size_t{0});
        };

        for(size_t k0 = 0; k0 < K0; k0++)
        {
            for(size_t k1 = 0; k1 < K1; k1++)
            {
                indexA = offset(std::vector<size_t>{m0, m1, k0, k1}, a_ms_ks_strides);
                valA   = static_cast<floatTypeCompute>(A[indexA]);

                indexB = offset(std::vector<size_t>{n0, n1, k0, k1}, b_ks_ns_strides);
                valB   = static_cast<floatTypeCompute>(B[indexB]);

                valAcc += valA * valB;
            }
        }

        valC = alpha * valAcc;

        indexD    = offset(std::vector<size_t>{m0, m1, n0, n1}, d_ms_ns_strides);
        D[indexD] = static_cast<DDataType>(valC);
    };

    auto GetNdIndices = [&](size_t index) {
        std::array<std::size_t, NDIM> indices;

        for(std::size_t idim = 0; idim < NDIM; ++idim)
        {
            indices[idim] = index / d_ms_ns_strides[idim];
            index -= indices[idim] * d_ms_ns_strides[idim];
        }

        return indices;
    };

    std::size_t                  work_per_thread = (elementsD + num_thread - 1) / num_thread;
    std::vector<joinable_thread> threads(num_thread);

    for(std::size_t i = 0; i < num_thread; ++i)
    {
        std::size_t it_begin = i * work_per_thread;
        std::size_t it_end   = std::min((i + 1) * work_per_thread, elementsD);

        auto f = [=] {
            for(std::size_t it = it_begin; it < it_end; ++it)
            {
                std::array<std::size_t, NDIM> indices = GetNdIndices(it);
                d_ms_ns(indices[0], indices[1], indices[2], indices[3]);
            }
        };

        threads[i] = joinable_thread(f);
    }
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename DDataType,
          typename floatTypeCompute>
void hiptensorBilinearContractionReference(ADataType*          A,
                                           BDataType*          B,
                                           CDataType*          C,
                                           DDataType*          D,
                                           floatTypeCompute    alpha,
                                           floatTypeCompute    beta,
                                           std::vector<size_t> a_ms_ks_lengths,
                                           std::vector<size_t> b_ks_ns_lengths,
                                           std::vector<size_t> c_ms_ns_lengths,
                                           std::vector<size_t> d_ms_ns_lengths,
                                           std::vector<size_t> a_ms_ks_strides,
                                           std::vector<size_t> b_ks_ns_strides,
                                           std::vector<size_t> c_ms_ns_strides,
                                           std::vector<size_t> d_ms_ns_strides,
                                           std::size_t         elementsD,
                                           std::size_t         num_thread = 1)
{
    auto d_ms_ns = [&](auto m0, auto m1, auto n0, auto n1) {
        floatTypeCompute valA, valB, valAcc = 0, valD1, valD2;
        size_t           indexA, indexB, indexC, indexD;

        auto K0 = a_ms_ks_lengths[2];
        auto K1 = a_ms_ks_lengths[3];

        auto offset = [&](std::vector<size_t> curIndices, std::vector<size_t> strides) {
            return std::inner_product(
                curIndices.begin(), curIndices.end(), strides.begin(), std::size_t{0});
        };

        for(size_t k0 = 0; k0 < K0; k0++)
        {
            for(size_t k1 = 0; k1 < K1; k1++)
            {
                indexA = offset(std::vector<size_t>{m0, m1, k0, k1}, a_ms_ks_strides);
                valA   = static_cast<floatTypeCompute>(A[indexA]);

                indexB = offset(std::vector<size_t>{n0, n1, k0, k1}, b_ks_ns_strides);
                valB   = static_cast<floatTypeCompute>(B[indexB]);

                valAcc += valA * valB;
            }
        }

        valD1 = valAcc * alpha;

        indexC = offset(std::vector<size_t>{m0, m1, n0, n1}, c_ms_ns_strides);
        valD2  = static_cast<floatTypeCompute>(C[indexC]) * beta;

        indexD    = offset(std::vector<size_t>{m0, m1, n0, n1}, d_ms_ns_strides);
        D[indexD] = static_cast<DDataType>(valD1 + valD2);
    };

    auto GetNdIndices = [&](size_t index) {
        std::array<std::size_t, NDIM> indices;

        for(std::size_t idim = 0; idim < NDIM; ++idim)
        {
            indices[idim] = index / d_ms_ns_strides[idim];
            index -= indices[idim] * d_ms_ns_strides[idim];
        }

        return indices;
    };

    std::size_t                  work_per_thread = (elementsD + num_thread - 1) / num_thread;
    std::vector<joinable_thread> threads(num_thread);

    for(std::size_t i = 0; i < num_thread; ++i)
    {
        std::size_t it_begin = i * work_per_thread;
        std::size_t it_end   = std::min((i + 1) * work_per_thread, elementsD);

        auto f = [=] {
            for(std::size_t it = it_begin; it < it_end; ++it)
            {
                std::array<std::size_t, NDIM> indices = GetNdIndices(it);
                d_ms_ns(indices[0], indices[1], indices[2], indices[3]);
            }
        };

        threads[i] = joinable_thread(f);
    }
}

template <typename DDataType>
std::pair<bool, double> compareEqual(DDataType const* deviceD,
                                     DDataType const* hostD,
                                     std::size_t      elementsD,
                                     double           tolerance = 100.0)
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
        auto valHost   = hostD[i];

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

template <typename DDataType>
std::pair<bool, double> compareEqualLaunchKernel(DDataType*  deviceD,
                                                 DDataType*  hostD,
                                                 std::size_t elementsD,
                                                 double      tolerance = 100.0)
{
    auto blockDim = dim3(1024, 1, 1);
    auto gridDim  = dim3(ceilDiv(elementsD, blockDim.x), 1, 1);

    double* d_relativeError;
    double  maxRelativeError;
    CHECK_HIP_ERROR(hipMalloc(&d_relativeError, elementsD * sizeof(double)));

    hipEvent_t syncEvent;
    CHECK_HIP_ERROR(hipEventCreate(&syncEvent));

    // Calculate the relative error for each element of Tensor D
    hipLaunchKernelGGL((compareEqualKernel<DDataType>),
                       gridDim,
                       blockDim,
                       0,
                       0,
                       deviceD,
                       hostD,
                       d_relativeError,
                       elementsD);
    CHECK_HIP_ERROR(hipEventRecord(syncEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(syncEvent));

    // Determine the maximum relative error
    blockDim             = dim3(512, 1, 1);
    uint32_t maxElements = 1024;
    uint32_t offset      = 1;

    for(uint32_t i = elementsD; i > 1; i = ceilDiv(i, maxElements))
    {
        gridDim       = dim3(ceilDiv(i, maxElements), 1, 1);
        auto elements = i > maxElements ? maxElements : i;

        hipLaunchKernelGGL((maxReduceKernel),
                           gridDim,
                           blockDim,
                           0,
                           0,
                           d_relativeError,
                           elements,
                           offset,
                           elementsD);

        CHECK_HIP_ERROR(hipEventRecord(syncEvent));
        CHECK_HIP_ERROR(hipEventSynchronize(syncEvent));
        offset = offset * maxElements;
    }

    CHECK_HIP_ERROR(
        hipMemcpy(&maxRelativeError, d_relativeError, sizeof(double), hipMemcpyDeviceToHost));

    // Free allocated device memory
    CHECK_HIP_ERROR(hipFree(d_relativeError));

    bool retval = true;
    bool isNaN  = std::isnan(maxRelativeError);

    auto toDouble
        = [](DDataType const& val) { return static_cast<double>(static_cast<float>(val)); };

    auto eps = toDouble(std::numeric_limits<DDataType>::epsilon());
    if(isNaN)
    {
        retval           = false;
        maxRelativeError = std::numeric_limits<DDataType>::signaling_NaN();
    }
    else if(maxRelativeError > (eps * tolerance))
    {
        retval = false;
    }

    return std::make_pair(retval, maxRelativeError);
}

#endif // HIPTENSOR_TEST_CONTRACTION_COMMON_HPP
