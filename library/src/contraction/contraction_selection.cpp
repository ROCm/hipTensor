/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
                                      std::vector<std::size_t> const&          a_ms_ks_lengths,
                                      std::vector<std::size_t> const&          a_ms_ks_strides,
                                      hipDataType                              typeB,
                                      std::vector<std::size_t> const&          b_ns_ks_lengths,
                                      std::vector<std::size_t> const&          b_ns_ks_strides,
                                      hipDataType                              typeD,
                                      std::vector<std::size_t> const&          d_ms_ns_lengths,
                                      std::vector<std::size_t> const&          d_ms_ns_strides,
                                      hipDataType                              typeE,
                                      std::vector<std::size_t> const&          e_ms_ns_lengths,
                                      std::vector<std::size_t> const&          e_ms_ns_strides,
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
        PerfMetrics          bestMetrics  = {
            0,
            "",
            0,
            0,
            0,
        };

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
                    solution->uid(), // id
                    solution->kernelName(), // name
                    time, // avg time
                    static_cast<float>(flops) / static_cast<float>(1.E9) / time, // tflops
                    static_cast<float>(bytes) / static_cast<float>(1.E6) / time // BW
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

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         PerfMetrics*                                            winnerMetrics,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         const uint64_t                                          workspaceSize)
        {
            return HIPTENSOR_STATUS_SUCCESS;
        }
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::BILINEAR>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         PerfMetrics*                                            winnerMetrics,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         const uint64_t                                          workspaceSize)
        {

            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            if(d5 <= 29)
            {
                if(d6 <= 26)
                {
                    if(d6 <= 12)
                    {
                        if(d5 <= 24)
                        {
                            if(d5 <= 2)
                            {
                                if(d6 <= 4)
                                {
                                    unique_id = 6253145787968798806;
                                }
                                else
                                {
                                    unique_id = 4781938049531404654;
                                }
                            }
                            else
                            {
                                unique_id = 4781938049531404654;
                            }
                        }
                        else
                        {
                            if(d6 <= 7)
                            {
                                unique_id = 4781938049531404654;
                            }
                            else
                            {
                                if(d3 <= 202)
                                {
                                    unique_id = 4781938049531404654;
                                }
                                else
                                {
                                    if(d4 <= 177)
                                    {
                                        unique_id = 4781938049531404654;
                                    }
                                    else
                                    {
                                        if(d1 <= 329)
                                        {
                                            if(d2 <= 617)
                                            {
                                                unique_id = 4781938049531404654;
                                            }
                                            else
                                            {
                                                unique_id = 16334338346691940719;
                                            }
                                        }
                                        else
                                        {
                                            if(d2 <= 219)
                                            {
                                                unique_id = 4781938049531404654;
                                            }
                                            else
                                            {
                                                unique_id = 16334338346691940719;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        if(d5 <= 18)
                        {
                            if(d5 <= 8)
                            {
                                unique_id = 4781938049531404654;
                            }
                            else
                            {
                                if(d1 <= 361)
                                {
                                    unique_id = 4781938049531404654;
                                }
                                else
                                {
                                    if(d6 <= 15)
                                    {
                                        if(d3 <= 400)
                                        {
                                            unique_id = 4781938049531404654;
                                        }
                                        else
                                        {
                                            unique_id = 10972102817010133142;
                                        }
                                    }
                                    else
                                    {
                                        if(d5 <= 12)
                                        {
                                            if(d3 <= 395)
                                            {
                                                unique_id = 4781938049531404654;
                                            }
                                            else
                                            {
                                                unique_id = 4781938049531404654;
                                            }
                                        }
                                        else
                                        {
                                            if(d6 <= 19)
                                            {
                                                unique_id = 4781938049531404654;
                                            }
                                            else
                                            {
                                                if(d4 <= 214)
                                                {
                                                    unique_id = 4781938049531404654;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            if(d1 <= 329)
                            {
                                if(d5 <= 24)
                                {
                                    unique_id = 4781938049531404654;
                                }
                                else
                                {
                                    if(d6 <= 16)
                                    {
                                        unique_id = 4781938049531404654;
                                    }
                                    else
                                    {
                                        if(d2 <= 491)
                                        {
                                            unique_id = 4781938049531404654;
                                        }
                                        else
                                        {
                                            unique_id = 16334338346691940719;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d4 <= 183)
                                {
                                    if(d2 <= 633)
                                    {
                                        if(d6 <= 24)
                                        {
                                            unique_id = 4781938049531404654;
                                        }
                                        else
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                    }
                                    else
                                    {
                                        unique_id = 4781938049531404654;
                                    }
                                }
                                else
                                {
                                    if(d2 <= 183)
                                    {
                                        if(d5 <= 24)
                                        {
                                            unique_id = 4781938049531404654;
                                        }
                                        else
                                        {
                                            unique_id = 16334338346691940719;
                                        }
                                    }
                                    else
                                    {
                                        if(d3 <= 124)
                                        {
                                            if(d6 <= 16)
                                            {
                                                unique_id = 4781938049531404654;
                                            }
                                            else
                                            {
                                                unique_id = 16334338346691940719;
                                            }
                                        }
                                        else
                                        {
                                            if(d5 <= 24)
                                            {
                                                if(d6 <= 14)
                                                {
                                                    unique_id = 4781938049531404654;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 16334338346691940719;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    if(d5 <= 8)
                    {
                        if(d5 <= 4)
                        {
                            unique_id = 4781938049531404654;
                        }
                        else
                        {
                            if(d6 <= 50)
                            {
                                if(d6 <= 48)
                                {
                                    if(d6 <= 39)
                                    {
                                        if(d6 <= 32)
                                        {
                                            unique_id = 4781938049531404654;
                                        }
                                        else
                                        {
                                            unique_id = 4781938049531404654;
                                        }
                                    }
                                    else
                                    {
                                        unique_id = 4781938049531404654;
                                    }
                                }
                                else
                                {
                                    unique_id = 4781938049531404654;
                                }
                            }
                            else
                            {
                                if(d3 <= 292)
                                {
                                    if(d6 <= 73)
                                    {
                                        unique_id = 4781938049531404654;
                                    }
                                    else
                                    {
                                        unique_id = 5897150915348629714;
                                    }
                                }
                                else
                                {
                                    if(d2 <= 325)
                                    {
                                        if(d6 <= 72)
                                        {
                                            unique_id = 4781938049531404654;
                                        }
                                        else
                                        {
                                            unique_id = 10972102817010133142;
                                        }
                                    }
                                    else
                                    {
                                        if(d5 <= 6)
                                        {
                                            if(d4 <= 110)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                if(d1 <= 246)
                                                {
                                                    unique_id = 4781938049531404654;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d6 <= 74)
                                            {
                                                if(d2 <= 701)
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        if(d6 <= 58)
                        {
                            if(d2 <= 529)
                            {
                                if(d4 <= 352)
                                {
                                    if(d5 <= 24)
                                    {
                                        if(d4 <= 144)
                                        {
                                            if(d6 <= 32)
                                            {
                                                if(d5 <= 16)
                                                {
                                                    unique_id = 4781938049531404654;
                                                }
                                                else
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                            }
                                            else
                                            {
                                                if(d5 <= 12)
                                                {
                                                    unique_id = 4781938049531404654;
                                                }
                                                else
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d1 <= 345)
                                            {
                                                if(d5 <= 12)
                                                {
                                                    unique_id = 4781938049531404654;
                                                }
                                                else
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                            }
                                            else
                                            {
                                                if(d3 <= 188)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        unique_id = 5897150915348629714;
                                    }
                                }
                                else
                                {
                                    if(d3 <= 128)
                                    {
                                        if(d5 <= 16)
                                        {
                                            if(d6 <= 42)
                                            {
                                                unique_id = 4781938049531404654;
                                            }
                                            else
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                        }
                                        else
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                    }
                                    else
                                    {
                                        if(d1 <= 345)
                                        {
                                            if(d5 <= 16)
                                            {
                                                if(d6 <= 42)
                                                {
                                                    unique_id = 4781938049531404654;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                            else
                                            {
                                                if(d2 <= 168)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d2 <= 42)
                                            {
                                                if(d5 <= 16)
                                                {
                                                    unique_id = 4781938049531404654;
                                                }
                                                else
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                            }
                                            else
                                            {
                                                if(d6 <= 49)
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d4 <= 108)
                                {
                                    if(d3 <= 337)
                                    {
                                        if(d5 <= 16)
                                        {
                                            unique_id = 4781938049531404654;
                                        }
                                        else
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                    }
                                    else
                                    {
                                        if(d2 <= 749)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            if(d4 <= 24)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                unique_id = 16334338346691940719;
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d3 <= 108)
                                    {
                                        if(d5 <= 16)
                                        {
                                            if(d6 <= 33)
                                            {
                                                unique_id = 4781938049531404654;
                                            }
                                            else
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                        }
                                        else
                                        {
                                            if(d2 <= 704)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                if(d4 <= 355)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d5 <= 12)
                                        {
                                            if(d6 <= 32)
                                            {
                                                unique_id = 4781938049531404654;
                                            }
                                            else
                                            {
                                                if(d1 <= 329)
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d1 <= 23)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                if(d6 <= 49)
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            if(d2 <= 531)
                            {
                                if(d3 <= 271)
                                {
                                    if(d3 <= 124)
                                    {
                                        unique_id = 5897150915348629714;
                                    }
                                    else
                                    {
                                        if(d4 <= 372)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            if(d1 <= 262)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                if(d5 <= 24)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d1 <= 263)
                                    {
                                        if(d3 <= 877)
                                        {
                                            if(d4 <= 447)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                        }
                                        else
                                        {
                                            if(d2 <= 197)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d4 <= 107)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            if(d5 <= 18)
                                            {
                                                if(d6 <= 68)
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                            else
                                            {
                                                if(d2 <= 44)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d5 <= 24)
                                {
                                    if(d3 <= 191)
                                    {
                                        if(d3 <= 121)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            if(d4 <= 324)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                unique_id = 16334338346691940719;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d6 <= 74)
                                        {
                                            if(d4 <= 97)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                if(d5 <= 14)
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d4 <= 108)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                if(d1 <= 248)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d3 <= 111)
                                    {
                                        unique_id = 5897150915348629714;
                                    }
                                    else
                                    {
                                        if(d4 <= 75)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            if(d6 <= 61)
                                            {
                                                unique_id = 10972102817010133142;
                                            }
                                            else
                                            {
                                                if(d1 <= 229)
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                if(d6 <= 38)
                {
                    if(d6 <= 8)
                    {
                        if(d6 <= 4)
                        {
                            if(d6 <= 3)
                            {
                                unique_id = 4781938049531404654;
                            }
                            else
                            {
                                if(d5 <= 72)
                                {
                                    unique_id = 4781938049531404654;
                                }
                                else
                                {
                                    unique_id = 4781938049531404654;
                                }
                            }
                        }
                        else
                        {
                            if(d1 <= 236)
                            {
                                if(d5 <= 65)
                                {
                                    unique_id = 4781938049531404654;
                                }
                                else
                                {
                                    if(d1 <= 118)
                                    {
                                        unique_id = 4781938049531404654;
                                    }
                                    else
                                    {
                                        unique_id = 4781938049531404654;
                                    }
                                }
                            }
                            else
                            {
                                if(d6 <= 6)
                                {
                                    if(d5 <= 33)
                                    {
                                        unique_id = 4781938049531404654;
                                    }
                                    else
                                    {
                                        if(d2 <= 421)
                                        {
                                            if(d4 <= 398)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                if(d3 <= 199)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d3 <= 135)
                                            {
                                                unique_id = 4781938049531404654;
                                            }
                                            else
                                            {
                                                if(d4 <= 134)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d5 <= 64)
                                    {
                                        if(d6 <= 7)
                                        {
                                            if(d3 <= 152)
                                            {
                                                unique_id = 4781938049531404654;
                                            }
                                            else
                                            {
                                                if(d2 <= 351)
                                                {
                                                    unique_id = 4781938049531404654;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d2 <= 355)
                                            {
                                                unique_id = 4781938049531404654;
                                            }
                                            else
                                            {
                                                if(d3 <= 187)
                                                {
                                                    unique_id = 4781938049531404654;
                                                }
                                                else
                                                {
                                                    unique_id = 4781938049531404654;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d4 <= 154)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            if(d3 <= 102)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                if(d2 <= 323)
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        if(d2 <= 355)
                        {
                            if(d4 <= 356)
                            {
                                if(d6 <= 16)
                                {
                                    if(d5 <= 48)
                                    {
                                        if(d6 <= 12)
                                        {
                                            unique_id = 4781938049531404654;
                                        }
                                        else
                                        {
                                            if(d5 <= 33)
                                            {
                                                unique_id = 4781938049531404654;
                                            }
                                            else
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d1 <= 590)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            if(d4 <= 151)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                unique_id = 16334338346691940719;
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d4 <= 152)
                                    {
                                        unique_id = 5897150915348629714;
                                    }
                                    else
                                    {
                                        if(d1 <= 491)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            if(d5 <= 47)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                if(d3 <= 194)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d1 <= 345)
                                {
                                    if(d6 <= 16)
                                    {
                                        if(d5 <= 48)
                                        {
                                            if(d6 <= 12)
                                            {
                                                unique_id = 4781938049531404654;
                                            }
                                            else
                                            {
                                                unique_id = 4781938049531404654;
                                            }
                                        }
                                        else
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                    }
                                    else
                                    {
                                        if(d5 <= 50)
                                        {
                                            if(d6 <= 24)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                        }
                                        else
                                        {
                                            if(d2 <= 187)
                                            {
                                                if(d2 <= 69)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                            }
                                            else
                                            {
                                                if(d1 <= 80)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d3 <= 120)
                                    {
                                        unique_id = 5897150915348629714;
                                    }
                                    else
                                    {
                                        if(d6 <= 23)
                                        {
                                            if(d2 <= 40)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                if(d5 <= 69)
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d5 <= 50)
                                            {
                                                if(d2 <= 53)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                            }
                                            else
                                            {
                                                if(d2 <= 101)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            if(d5 <= 54)
                            {
                                if(d4 <= 152)
                                {
                                    if(d6 <= 16)
                                    {
                                        if(d5 <= 33)
                                        {
                                            unique_id = 4781938049531404654;
                                        }
                                        else
                                        {
                                            if(d3 <= 300)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                unique_id = 16334338346691940719;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d4 <= 82)
                                        {
                                            if(d6 <= 24)
                                            {
                                                if(d5 <= 32)
                                                {
                                                    unique_id = 4781938049531404654;
                                                }
                                                else
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                        }
                                        else
                                        {
                                            if(d3 <= 270)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                if(d2 <= 538)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d3 <= 108)
                                    {
                                        if(d6 <= 16)
                                        {
                                            if(d5 <= 40)
                                            {
                                                unique_id = 4781938049531404654;
                                            }
                                            else
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                        }
                                        else
                                        {
                                            if(d2 <= 701)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                if(d3 <= 48)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d6 <= 23)
                                        {
                                            if(d1 <= 62)
                                            {
                                                if(d6 <= 13)
                                                {
                                                    unique_id = 4781938049531404654;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                            else
                                            {
                                                if(d5 <= 33)
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d5 <= 45)
                                            {
                                                if(d2 <= 529)
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                            else
                                            {
                                                if(d2 <= 858)
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d6 <= 19)
                                {
                                    if(d3 <= 105)
                                    {
                                        unique_id = 5897150915348629714;
                                    }
                                    else
                                    {
                                        if(d4 <= 152)
                                        {
                                            if(d1 <= 582)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                if(d5 <= 69)
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                                else
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d5 <= 69)
                                            {
                                                if(d1 <= 59)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                            else
                                            {
                                                if(d1 <= 345)
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d4 <= 105)
                                    {
                                        if(d4 <= 60)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            if(d3 <= 264)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d3 <= 76)
                                        {
                                            if(d3 <= 48)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                        }
                                        else
                                        {
                                            if(d1 <= 55)
                                            {
                                                if(d2 <= 691)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                            else
                                            {
                                                if(d2 <= 860)
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    if(d5 <= 44)
                    {
                        if(d6 <= 49)
                        {
                            if(d2 <= 530)
                            {
                                if(d4 <= 356)
                                {
                                    unique_id = 5897150915348629714;
                                }
                                else
                                {
                                    if(d3 <= 264)
                                    {
                                        unique_id = 5897150915348629714;
                                    }
                                    else
                                    {
                                        if(d2 <= 335)
                                        {
                                            if(d1 <= 262)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                        }
                                        else
                                        {
                                            if(d1 <= 261)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142;
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d4 <= 145)
                                {
                                    unique_id = 5897150915348629714;
                                }
                                else
                                {
                                    if(d3 <= 117)
                                    {
                                        unique_id = 5897150915348629714;
                                    }
                                    else
                                    {
                                        if(d5 <= 34)
                                        {
                                            if(d6 <= 45)
                                            {
                                                unique_id = 16334338346691940719;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142;
                                            }
                                        }
                                        else
                                        {
                                            if(d6 <= 45)
                                            {
                                                if(d2 <= 860)
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            if(d3 <= 120)
                            {
                                if(d2 <= 538)
                                {
                                    if(d4 <= 409)
                                    {
                                        unique_id = 5897150915348629714;
                                    }
                                    else
                                    {
                                        if(d1 <= 485)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                    }
                                }
                                else
                                {
                                    if(d4 <= 290)
                                    {
                                        unique_id = 5897150915348629714;
                                    }
                                    else
                                    {
                                        if(d6 <= 58)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            if(d3 <= 22)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142;
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d4 <= 82)
                                {
                                    if(d2 <= 527)
                                    {
                                        unique_id = 5897150915348629714;
                                    }
                                    else
                                    {
                                        if(d6 <= 61)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            unique_id = 10972102817010133142;
                                        }
                                    }
                                }
                                else
                                {
                                    if(d2 <= 169)
                                    {
                                        if(d1 <= 265)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            if(d2 <= 41)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                if(d6 <= 59)
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d1 <= 157)
                                        {
                                            if(d2 <= 527)
                                            {
                                                if(d1 <= 51)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                            else
                                            {
                                                if(d1 <= 28)
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d3 <= 266)
                                            {
                                                if(d6 <= 59)
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                            else
                                            {
                                                if(d6 <= 59)
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        if(d4 <= 76)
                        {
                            if(d6 <= 58)
                            {
                                if(d3 <= 301)
                                {
                                    unique_id = 5897150915348629714;
                                }
                                else
                                {
                                    if(d2 <= 420)
                                    {
                                        unique_id = 5897150915348629714;
                                    }
                                    else
                                    {
                                        if(d4 <= 34)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            if(d6 <= 49)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142;
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d1 <= 217)
                                {
                                    if(d2 <= 309)
                                    {
                                        unique_id = 5897150915348629714;
                                    }
                                    else
                                    {
                                        unique_id = 10972102817010133142;
                                    }
                                }
                                else
                                {
                                    if(d3 <= 179)
                                    {
                                        unique_id = 10972102817010133142;
                                    }
                                    else
                                    {
                                        if(d2 <= 179)
                                        {
                                            unique_id = 10972102817010133142;
                                        }
                                        else
                                        {
                                            if(d4 <= 14)
                                            {
                                                unique_id = 10972102817010133142;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            if(d3 <= 100)
                            {
                                if(d6 <= 55)
                                {
                                    if(d2 <= 326)
                                    {
                                        unique_id = 5897150915348629714;
                                    }
                                    else
                                    {
                                        if(d3 <= 26)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            if(d4 <= 354)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                if(d6 <= 45)
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d2 <= 192)
                                    {
                                        if(d1 <= 407)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            unique_id = 10972102817010133142;
                                        }
                                    }
                                    else
                                    {
                                        if(d3 <= 16)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            if(d1 <= 79)
                                            {
                                                unique_id = 10972102817010133142;
                                            }
                                            else
                                            {
                                                if(d4 <= 354)
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d2 <= 40)
                                {
                                    if(d6 <= 49)
                                    {
                                        unique_id = 5897150915348629714;
                                    }
                                    else
                                    {
                                        if(d1 <= 416)
                                        {
                                            unique_id = 5897150915348629714;
                                        }
                                        else
                                        {
                                            if(d2 <= 17)
                                            {
                                                unique_id = 10972102817010133142;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142;
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d1 <= 50)
                                    {
                                        if(d2 <= 501)
                                        {
                                            if(d1 <= 21)
                                            {
                                                unique_id = 5897150915348629714;
                                            }
                                            else
                                            {
                                                if(d6 <= 49)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d1 <= 19)
                                            {
                                                if(d6 <= 61)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                            else
                                            {
                                                if(d6 <= 47)
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d6 <= 45)
                                        {
                                            if(d2 <= 327)
                                            {
                                                if(d5 <= 50)
                                                {
                                                    unique_id = 5897150915348629714;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                            else
                                            {
                                                if(d5 <= 50)
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d2 <= 171)
                                            {
                                                if(d1 <= 159)
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
            {
                *winner = candidate->second;

                int32_t m, n, k;
                std::tie(m, n, k) = (*winner)->problemDims();
                auto time         = 0.0f;
                auto flops        = std::size_t(2) * m * n * k;
                auto bytes        = (*winner)->problemBytes();

                PerfMetrics metrics = {
                    (*winner)->uid(), // id
                    (*winner)->kernelName(), // name
                    time, // avg time
                    static_cast<float>(flops) / static_cast<float>(1.E9) / time, // tflops
                    static_cast<float>(bytes) / static_cast<float>(1.E6) / time // BW
                };

                *winnerMetrics = metrics;
                return HIPTENSOR_STATUS_SUCCESS;
            }
            else
            {
                return HIPTENSOR_STATUS_EXECUTION_FAILED;
            }
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::SCALE>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         PerfMetrics*                                            winnerMetrics,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         const uint64_t                                          workspaceSize)
        {
            return HIPTENSOR_STATUS_SUCCESS;
        }
    };

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::BILINEAR>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
                         PerfMetrics*                                            winnerMetrics,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_lengths[0];
            int d2 = a_ms_ks_lengths[1];
            int d3 = b_ns_ks_lengths[0];
            int d4 = b_ns_ks_lengths[1];
            int d5 = a_ms_ks_lengths[2];
            int d6 = a_ms_ks_lengths[3];

            size_t unique_id = 0;

            if(d6 <= 28)
            {
                if(d6 <= 8)
                {
                    unique_id = 4781938049531404654;
                }
                else
                {
                    if(d5 <= 32)
                    {
                        if(d5 <= 24)
                        {
                            unique_id = 4781938049531404654;
                        }
                        else
                        {
                            if(d6 <= 17)
                            {
                                unique_id = 4781938049531404654;
                            }
                            else
                            {
                                if(d2 <= 369)
                                {
                                    unique_id = 5897150915348629714;
                                }
                                else
                                {
                                    unique_id = 16334338346691940719;
                                }
                            }
                        }
                    }
                    else
                    {
                        if(d4 <= 406)
                        {
                            if(d5 <= 52)
                            {
                                if(d6 <= 16)
                                {
                                    unique_id = 4781938049531404654;
                                }
                                else
                                {
                                    unique_id = 5897150915348629714;
                                }
                            }
                            else
                            {
                                unique_id = 5897150915348629714;
                            }
                        }
                        else
                        {
                            if(d5 <= 56)
                            {
                                if(d6 <= 16)
                                {
                                    unique_id = 4781938049531404654;
                                }
                                else
                                {
                                    unique_id = 16334338346691940719;
                                }
                            }
                            else
                            {
                                if(d1 <= 457)
                                {
                                    unique_id = 5897150915348629714;
                                }
                                else
                                {
                                    unique_id = 10972102817010133142;
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                if(d5 <= 12)
                {
                    if(d5 <= 8)
                    {
                        if(d3 <= 491)
                        {
                            unique_id = 4781938049531404654;
                        }
                        else
                        {
                            unique_id = 4781938049531404654;
                        }
                    }
                    else
                    {
                        if(d6 <= 56)
                        {
                            if(d3 <= 493)
                            {
                                unique_id = 4781938049531404654;
                            }
                            else
                            {
                                if(d1 <= 282)
                                {
                                    unique_id = 4781938049531404654;
                                }
                                else
                                {
                                    unique_id = 16334338346691940719;
                                }
                            }
                        }
                        else
                        {
                            if(d2 <= 274)
                            {
                                unique_id = 5897150915348629714;
                            }
                            else
                            {
                                if(d4 <= 299)
                                {
                                    unique_id = 5897150915348629714;
                                }
                                else
                                {
                                    unique_id = 16334338346691940719;
                                }
                            }
                        }
                    }
                }
                else
                {
                    if(d6 <= 56)
                    {
                        if(d5 <= 50)
                        {
                            if(d2 <= 362)
                            {
                                unique_id = 5897150915348629714;
                            }
                            else
                            {
                                if(d4 <= 320)
                                {
                                    unique_id = 5897150915348629714;
                                }
                                else
                                {
                                    unique_id = 16334338346691940719;
                                }
                            }
                        }
                        else
                        {
                            if(d4 <= 401)
                            {
                                if(d6 <= 46)
                                {
                                    unique_id = 5897150915348629714;
                                }
                                else
                                {
                                    unique_id = 10972102817010133142;
                                }
                            }
                            else
                            {
                                if(d2 <= 77)
                                {
                                    unique_id = 5897150915348629714;
                                }
                                else
                                {
                                    unique_id = 10972102817010133142;
                                }
                            }
                        }
                    }
                    else
                    {
                        if(d2 <= 302)
                        {
                            if(d5 <= 46)
                            {
                                if(d1 <= 457)
                                {
                                    unique_id = 5897150915348629714;
                                }
                                else
                                {
                                    unique_id = 10972102817010133142;
                                }
                            }
                            else
                            {
                                if(d1 <= 115)
                                {
                                    unique_id = 5897150915348629714;
                                }
                                else
                                {
                                    unique_id = 10972102817010133142;
                                }
                            }
                        }
                        else
                        {
                            if(d1 <= 43)
                            {
                                if(d1 <= 19)
                                {
                                    unique_id = 5897150915348629714;
                                }
                                else
                                {
                                    unique_id = 10972102817010133142;
                                }
                            }
                            else
                            {
                                unique_id = 10972102817010133142;
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
    };

    hiptensorStatus_t
        actorCriticModel(ContractionSolution**                                   winner,
                         PerfMetrics*                                            winnerMetrics,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         const uint64_t                                          workspaceSize)
    {
        if(typeA == HIP_R_32F && typeB == HIP_R_32F && typeD == NONE_TYPE && typeE == HIP_R_32F)
        {
            return ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE>::
                selectWinner(winner,
                             winnerMetrics,
                             candidates,
                             typeA,
                             a_ms_ks_lengths,
                             a_ms_ks_strides,
                             typeB,
                             b_ns_ks_lengths,
                             b_ns_ks_strides,
                             typeD,
                             d_ms_ns_lengths,
                             d_ms_ns_strides,
                             typeE,
                             e_ms_ns_lengths,
                             e_ms_ns_strides,
                             workspaceSize);
        }
        else if(typeA == HIP_R_32F && typeB == HIP_R_32F && typeD == HIP_R_32F
                && typeE == HIP_R_32F)
        {
            return ActorCriticSelection<float, float, float, float, ContractionOpId_t::BILINEAR>::
                selectWinner(winner,
                             winnerMetrics,
                             candidates,
                             typeA,
                             a_ms_ks_lengths,
                             a_ms_ks_strides,
                             typeB,
                             b_ns_ks_lengths,
                             b_ns_ks_strides,
                             typeD,
                             d_ms_ns_lengths,
                             d_ms_ns_strides,
                             typeE,
                             e_ms_ns_lengths,
                             e_ms_ns_strides,
                             workspaceSize);
        }
        else if(typeA == HIP_R_64F && typeB == HIP_R_64F && typeD == NONE_TYPE
                && typeE == HIP_R_64F)
        {
            return ActorCriticSelection<double, double, double, double, ContractionOpId_t::SCALE>::
                selectWinner(winner,
                             winnerMetrics,
                             candidates,
                             typeA,
                             a_ms_ks_lengths,
                             a_ms_ks_strides,
                             typeB,
                             b_ns_ks_lengths,
                             b_ns_ks_strides,
                             typeD,
                             d_ms_ns_lengths,
                             d_ms_ns_strides,
                             typeE,
                             e_ms_ns_lengths,
                             e_ms_ns_strides,
                             workspaceSize);
        }
        else if(typeA == HIP_R_64F && typeB == HIP_R_64F && typeD == HIP_R_64F
                && typeE == HIP_R_64F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::BILINEAR>::selectWinner(winner,
                                                                                   winnerMetrics,
                                                                                   candidates,
                                                                                   typeA,
                                                                                   a_ms_ks_lengths,
                                                                                   a_ms_ks_strides,
                                                                                   typeB,
                                                                                   b_ns_ks_lengths,
                                                                                   b_ns_ks_strides,
                                                                                   typeD,
                                                                                   d_ms_ns_lengths,
                                                                                   d_ms_ns_strides,
                                                                                   typeE,
                                                                                   e_ms_ns_lengths,
                                                                                   e_ms_ns_strides,
                                                                                   workspaceSize);
        }
        return HIPTENSOR_STATUS_EXECUTION_FAILED;
    }
}
