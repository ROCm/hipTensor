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

#ifndef CHECK_HIP_ALLOC
#define CHECK_HIP_ALLOC(status)               \
    if(status != hipSuccess)                  \
    {                                         \
        return HIPTENSOR_STATUS_ALLOC_FAILED; \
    }
#endif

#include "contraction_selection.hpp"
#include "performance.hpp"
#include "util.hpp"

#include "contraction_cpu_reference.hpp"

namespace hiptensor
{
    hiptensorStatus_t bruteForceModel(ContractionSolution**                    winner,
                                      PerfMetrics*                             winnerMetrics,
                                      std::vector<ContractionSolution*> const& candidates,
                                      hipDataType                              typeA,
                                      std::vector<ck::index_t> const&          a_ms_ks_lengths,
                                      std::vector<ck::index_t> const&          a_ms_ks_strides,
                                      hipDataType                              typeB,
                                      std::vector<ck::index_t> const&          b_ns_ks_lengths,
                                      std::vector<ck::index_t> const&          b_ns_ks_strides,
                                      hipDataType                              typeD,
                                      std::vector<ck::index_t> const&          d_ms_ns_lengths,
                                      std::vector<ck::index_t> const&          d_ms_ns_strides,
                                      hipDataType                              typeE,
                                      std::vector<ck::index_t> const&          e_ms_ns_lengths,
                                      std::vector<ck::index_t> const&          e_ms_ns_strides,
                                      const uint64_t                           workspaceSize)
    {
        // Make sure that we calculate full element space incase strides are not packed.
        auto sizeA = elementSpaceFromLengthsAndStrides(a_ms_ks_lengths, a_ms_ks_strides)
                     * hipDataTypeSize(typeA);
        auto sizeB = elementSpaceFromLengthsAndStrides(b_ns_ks_lengths, b_ns_ks_strides)
                     * hipDataTypeSize(typeB);
        auto sizeD = 0;
        if(typeD != NONE_TYPE)
        {
            sizeD = elementSpaceFromLengthsAndStrides(d_ms_ns_lengths, d_ms_ns_strides)
                    * hipDataTypeSize(typeD);
        }
        auto sizeE = elementSpaceFromLengthsAndStrides(e_ms_ns_lengths, e_ms_ns_strides)
                     * hipDataTypeSize(typeE);

        void *A_d, *B_d, *D_d, *E_d, *wspace;
        float alpha = 1.02f;
        float beta  = 1.03f;

        CHECK_HIP_ALLOC(hipMalloc(&A_d, sizeA));
        CHECK_HIP_ALLOC(hipMalloc(&B_d, sizeB));
        CHECK_HIP_ALLOC(hipMalloc(&D_d, sizeD));
        CHECK_HIP_ALLOC(hipMalloc(&E_d, sizeE));
        CHECK_HIP_ALLOC(hipMalloc(&wspace, workspaceSize));

        std::string          best_op_name;
        ContractionSolution* bestSolution = nullptr;
        PerfMetrics          bestMetrics  = {0, 0, 0, ""};

        for(auto* solution : candidates)
        {
            if(solution->initArgs(&alpha,
                                  A_d,
                                  B_d,
                                  &beta,
                                  D_d,
                                  E_d,
                                  a_ms_ks_lengths,
                                  a_ms_ks_strides,
                                  b_ns_ks_lengths,
                                  b_ns_ks_strides,
                                  d_ms_ns_lengths,
                                  d_ms_ns_strides,
                                  e_ms_ns_lengths,
                                  e_ms_ns_strides,
                                  wspace)
               && solution->workspaceSize() <= workspaceSize)
            {
                // Make sure to time the kernels
                auto    time = (*solution)(StreamConfig{nullptr, true});
                int32_t m, n, k;
                std::tie(m, n, k) = solution->problemDims();
                auto flops        = std::size_t(2) * m * n * k;
                auto bytes        = solution->problemBytes();

                PerfMetrics metrics = {
                    time, // avg time
                    static_cast<float>(flops) / static_cast<float>(1.E9) / time, // tflops
                    static_cast<float>(bytes) / static_cast<float>(1.E6) / time, // BW
                    solution->kernelName() // name
                };

                if(metrics > bestMetrics)
                {
                    bestSolution = solution;
                    bestMetrics  = metrics;
                }
            }
        }

        CHECK_HIP_ALLOC(hipFree(A_d));
        CHECK_HIP_ALLOC(hipFree(B_d));
        CHECK_HIP_ALLOC(hipFree(D_d));
        CHECK_HIP_ALLOC(hipFree(E_d));
        CHECK_HIP_ALLOC(hipFree(wspace));

        *winnerMetrics = bestMetrics;

        *winner = bestSolution;

        if(bestSolution == nullptr)
        {
            return HIPTENSOR_STATUS_EXECUTION_FAILED;
        }
        else
        {
            return HIPTENSOR_STATUS_SUCCESS;
        }
    }

    hiptensorStatus_t
        actorCriticModel(ContractionSolution**                                   winner,
                         PerfMetrics*                                            winnerMetrics,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<ck::index_t> const&                         a_ms_ks_lengths,
                         std::vector<ck::index_t> const&                         a_ms_ks_strides,
                         hipDataType                                             typeB,
                         std::vector<ck::index_t> const&                         b_ns_ks_lengths,
                         std::vector<ck::index_t> const&                         b_ns_ks_strides,
                         hipDataType                                             typeD,
                         std::vector<ck::index_t> const&                         d_ms_ns_lengths,
                         std::vector<ck::index_t> const&                         d_ms_ns_strides,
                         hipDataType                                             typeE,
                         std::vector<ck::index_t> const&                         e_ms_ns_lengths,
                         std::vector<ck::index_t> const&                         e_ms_ns_strides,
                         const uint64_t                                          workspaceSize)
    {

        int d1 = a_ms_ks_lengths[0];
        int d2 = a_ms_ks_lengths[1];
        int d3 = b_ns_ks_lengths[0];
        int d4 = b_ns_ks_lengths[1];
        int d5 = a_ms_ks_lengths[2];
        int d6 = a_ms_ks_lengths[3];

        size_t unique_id = 0;

        if(d5 <= 35)
        {
            if(d5 <= 23)
            {
                if(d1 <= 140)
                {
                    if(d1 <= 137)
                    {
                        if(d2 <= 240)
                        {
                            unique_id = 3324992315903551472;
                        }
                        else
                        {
                            if(d6 <= 44)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 550562982100054122;
                            }
                        }
                    }
                    else
                    {
                        if(d3 <= 5519)
                        {
                            unique_id = 550562982100054122;
                        }
                        else
                        {
                            unique_id = 3324992315903551472;
                        }
                    }
                }
                else
                {
                    if(d4 <= 11)
                    {
                        if(d5 <= 14)
                        {
                            unique_id = 550562982100054122;
                        }
                        else
                        {
                            unique_id = 4157899012150127975;
                        }
                    }
                    else
                    {
                        if(d1 <= 854)
                        {
                            if(d2 <= 808)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 550562982100054122;
                            }
                        }
                        else
                        {
                            if(d4 <= 26)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 550562982100054122;
                            }
                        }
                    }
                }
            }
            else
            {
                if(d6 <= 61)
                {
                    if(d1 <= 140)
                    {
                        if(d1 <= 53)
                        {
                            unique_id = 550562982100054122;
                        }
                        else
                        {
                            if(d1 <= 75)
                            {
                                unique_id = 3324992315903551472;
                            }
                            else
                            {
                                unique_id = 550562982100054122;
                            }
                        }
                    }
                    else
                    {
                        if(d4 <= 867)
                        {
                            if(d4 <= 866)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 4157899012150127975;
                            }
                        }
                        else
                        {
                            if(d1 <= 718)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 550562982100054122;
                            }
                        }
                    }
                }
                else
                {
                    if(d5 <= 26)
                    {
                        if(d5 <= 25)
                        {
                            if(d5 <= 24)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 550562982100054122;
                            }
                        }
                        else
                        {
                            if(d6 <= 70)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 4157899012150127975;
                            }
                        }
                    }
                    else
                    {
                        if(d2 <= 887)
                        {
                            if(d5 <= 34)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 4157899012150127975;
                            }
                        }
                        else
                        {
                            if(d6 <= 77)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 550562982100054122;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            if(d6 <= 30)
            {
                if(d6 <= 27)
                {
                    if(d6 <= 23)
                    {
                        if(d1 <= 152)
                        {
                            if(d5 <= 67)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 550562982100054122;
                            }
                        }
                        else
                        {
                            if(d4 <= 14)
                            {
                                unique_id = 3324992315903551472;
                            }
                            else
                            {
                                unique_id = 550562982100054122;
                            }
                        }
                    }
                    else
                    {
                        if(d6 <= 24)
                        {
                            if(d5 <= 71)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 4157899012150127975;
                            }
                        }
                        else
                        {
                            if(d2 <= 955)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 550562982100054122;
                            }
                        }
                    }
                }
                else
                {
                    if(d5 <= 71)
                    {
                        if(d1 <= 274)
                        {
                            if(d2 <= 4572)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 3324992315903551472;
                            }
                        }
                        else
                        {
                            if(d1 <= 421)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 550562982100054122;
                            }
                        }
                    }
                    else
                    {
                        if(d2 <= 868)
                        {
                            if(d5 <= 73)
                            {
                                unique_id = 4157899012150127975;
                            }
                            else
                            {
                                unique_id = 550562982100054122;
                            }
                        }
                        else
                        {
                            if(d1 <= 887)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 4157899012150127975;
                            }
                        }
                    }
                }
            }
            else
            {
                if(d5 <= 55)
                {
                    if(d6 <= 55)
                    {
                        if(d5 <= 48)
                        {
                            if(d6 <= 52)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 550562982100054122;
                            }
                        }
                        else
                        {
                            if(d6 <= 43)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 4157899012150127975;
                            }
                        }
                    }
                    else
                    {
                        if(d6 <= 59)
                        {
                            if(d5 <= 43)
                            {
                                unique_id = 4157899012150127975;
                            }
                            else
                            {
                                unique_id = 4157899012150127975;
                            }
                        }
                        else
                        {
                            if(d1 <= 825)
                            {
                                unique_id = 4157899012150127975;
                            }
                            else
                            {
                                unique_id = 4157899012150127975;
                            }
                        }
                    }
                }
                else
                {
                    if(d6 <= 32)
                    {
                        if(d5 <= 62)
                        {
                            if(d5 <= 56)
                            {
                                unique_id = 4157899012150127975;
                            }
                            else
                            {
                                unique_id = 550562982100054122;
                            }
                        }
                        else
                        {
                            if(d3 <= 326)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 4157899012150127975;
                            }
                        }
                    }
                    else
                    {
                        if(d3 <= 348)
                        {
                            if(d6 <= 42)
                            {
                                unique_id = 550562982100054122;
                            }
                            else
                            {
                                unique_id = 4157899012150127975;
                            }
                        }
                        else
                        {
                            if(d1 <= 906)
                            {
                                unique_id = 4157899012150127975;
                            }
                            else
                            {
                                unique_id = 4157899012150127975;
                            }
                        }
                    }
                }
            }
        }

        if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
        {
            *winner = candidate->second;
            return HIPTENSOR_STATUS_SUCCESS;
        }
        else
        {
            return HIPTENSOR_STATUS_EXECUTION_FAILED;
        }
    }

}
