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

            if(d5 <= 36)
            {
                if(d6 <= 53)
                {
                    if(d6 <= 2)
                    {
                        if(d5 <= 14)
                        {
                            unique_id = 11803951795078848381ull;
                        }
                        else
                        {
                            unique_id = 12220255318180845084ull;
                        }
                    }
                    else
                    {
                        if(d4 <= 208)
                        {
                            if(d3 <= 437)
                            {
                                unique_id = 15717866180122059508ull;
                            }
                            else
                            {
                                if(d2 <= 174)
                                {
                                    unique_id = 15717866180122059508ull;
                                }
                                else
                                {
                                    unique_id = 10212603695171387835ull;
                                }
                            }
                        }
                        else
                        {
                            if(d5 <= 2)
                            {
                                if(d6 <= 12)
                                {
                                    unique_id = 11803951795078848381ull;
                                }
                                else
                                {
                                    unique_id = 12220255318180845084ull;
                                }
                            }
                            else
                            {
                                if(d2 <= 50)
                                {
                                    unique_id = 15717866180122059508ull;
                                }
                                else
                                {
                                    if(d6 <= 31)
                                    {
                                        if(d6 <= 6)
                                        {
                                            if(d5 <= 17)
                                            {
                                                unique_id = 14031134003016946658ull;
                                            }
                                            else
                                            {
                                                unique_id = 10212603695171387835ull;
                                            }
                                        }
                                        else
                                        {
                                            unique_id = 10212603695171387835ull;
                                        }
                                    }
                                    else
                                    {
                                        if(d5 <= 31)
                                        {
                                            unique_id = 10212603695171387835ull;
                                        }
                                        else
                                        {
                                            if(d1 <= 444)
                                            {
                                                unique_id = 10212603695171387835ull;
                                            }
                                            else
                                            {
                                                unique_id = 11587153158789362298ull;
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
                    if(d5 <= 15)
                    {
                        if(d6 <= 68)
                        {
                            if(d5 <= 1)
                            {
                                unique_id = 14031134003016946658ull;
                            }
                            else
                            {
                                unique_id = 10212603695171387835ull;
                            }
                        }
                        else
                        {
                            if(d1 <= 444)
                            {
                                unique_id = 10212603695171387835ull;
                            }
                            else
                            {
                                unique_id = 11587153158789362298ull;
                            }
                        }
                    }
                    else
                    {
                        if(d1 <= 223)
                        {
                            if(d2 <= 222)
                            {
                                unique_id = 15717866180122059508ull;
                            }
                            else
                            {
                                unique_id = 11587153158789362298ull;
                            }
                        }
                        else
                        {
                            if(d2 <= 50)
                            {
                                unique_id = 15717866180122059508ull;
                            }
                            else
                            {
                                unique_id = 11587153158789362298ull;
                            }
                        }
                    }
                }
            }
            else
            {
                if(d6 <= 23)
                {
                    if(d6 <= 15)
                    {
                        if(d4 <= 454)
                        {
                            if(d3 <= 650)
                            {
                                if(d6 <= 4)
                                {
                                    if(d6 <= 1)
                                    {
                                        unique_id = 12220255318180845084ull;
                                    }
                                    else
                                    {
                                        unique_id = 10212603695171387835ull;
                                    }
                                }
                                else
                                {
                                    if(d6 <= 8)
                                    {
                                        if(d6 <= 7)
                                        {
                                            unique_id = 15717866180122059508ull;
                                        }
                                        else
                                        {
                                            unique_id = 12220255318180845084ull;
                                        }
                                    }
                                    else
                                    {
                                        unique_id = 15717866180122059508ull;
                                    }
                                }
                            }
                            else
                            {
                                unique_id = 10212603695171387835ull;
                            }
                        }
                        else
                        {
                            if(d6 <= 1)
                            {
                                unique_id = 12220255318180845084ull;
                            }
                            else
                            {
                                if(d6 <= 3)
                                {
                                    unique_id = 10212603695171387835ull;
                                }
                                else
                                {
                                    if(d6 <= 8)
                                    {
                                        if(d6 <= 7)
                                        {
                                            unique_id = 10212603695171387835ull;
                                        }
                                        else
                                        {
                                            unique_id = 12220255318180845084ull;
                                        }
                                    }
                                    else
                                    {
                                        unique_id = 10212603695171387835ull;
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        if(d5 <= 50)
                        {
                            if(d6 <= 16)
                            {
                                unique_id = 12220255318180845084ull;
                            }
                            else
                            {
                                if(d4 <= 307)
                                {
                                    unique_id = 15717866180122059508ull;
                                }
                                else
                                {
                                    unique_id = 10212603695171387835ull;
                                }
                            }
                        }
                        else
                        {
                            unique_id = 11587153158789362298ull;
                        }
                    }
                }
                else
                {
                    if(d2 <= 52)
                    {
                        if(d1 <= 372)
                        {
                            unique_id = 15717866180122059508ull;
                        }
                        else
                        {
                            unique_id = 11587153158789362298ull;
                        }
                    }
                    else
                    {
                        if(d1 <= 78)
                        {
                            if(d2 <= 293)
                            {
                                unique_id = 15717866180122059508ull;
                            }
                            else
                            {
                                unique_id = 11587153158789362298ull;
                            }
                        }
                        else
                        {
                            if(d6 <= 31)
                            {
                                if(d5 <= 45)
                                {
                                    unique_id = 10212603695171387835ull;
                                }
                                else
                                {
                                    unique_id = 11587153158789362298ull;
                                }
                            }
                            else
                            {
                                if(d3 <= 114)
                                {
                                    if(d4 <= 149)
                                    {
                                        unique_id = 15717866180122059508ull;
                                    }
                                    else
                                    {
                                        unique_id = 11587153158789362298ull;
                                    }
                                }
                                else
                                {
                                    unique_id = 11587153158789362298ull;
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
    };

    template <>
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::BILINEAR>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
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

            if(d6 <= 45)
            {
                if(d6 <= 8)
                {
                    if(d6 <= 4)
                    {
                        if(d5 <= 2)
                        {
                            if(d1 <= 433)
                            {
                                unique_id = 4781938049531404654ull;
                            }
                            else
                            {
                                unique_id = 6253145787968798806ull;
                            }
                        }
                        else
                        {
                            unique_id = 4781938049531404654ull;
                        }
                    }
                    else
                    {
                        if(d5 <= 26)
                        {
                            if(d5 <= 1)
                            {
                                if(d1 <= 345)
                                {
                                    unique_id = 4781938049531404654ull;
                                }
                                else
                                {
                                    unique_id = 6253145787968798806ull;
                                }
                            }
                            else
                            {
                                unique_id = 4781938049531404654ull;
                            }
                        }
                        else
                        {
                            if(d1 <= 317)
                            {
                                unique_id = 4781938049531404654ull;
                            }
                            else
                            {
                                if(d6 <= 7)
                                {
                                    if(d4 <= 222)
                                    {
                                        unique_id = 4781938049531404654ull;
                                    }
                                    else
                                    {
                                        if(d3 <= 166)
                                        {
                                            unique_id = 4781938049531404654ull;
                                        }
                                        else
                                        {
                                            if(d5 <= 51)
                                            {
                                                if(d6 <= 6)
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                                else
                                                {
                                                    unique_id = 4781938049531404654ull;
                                                }
                                            }
                                            else
                                            {
                                                if(d5 <= 65)
                                                {
                                                    unique_id = 4781938049531404654ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    unique_id = 4781938049531404654ull;
                                }
                            }
                        }
                    }
                }
                else
                {
                    if(d5 <= 16)
                    {
                        if(d5 <= 8)
                        {
                            if(d5 <= 6)
                            {
                                if(d5 <= 1)
                                {
                                    if(d6 <= 13)
                                    {
                                        if(d1 <= 429)
                                        {
                                            unique_id = 4781938049531404654ull;
                                        }
                                        else
                                        {
                                            unique_id = 6253145787968798806ull;
                                        }
                                    }
                                    else
                                    {
                                        unique_id = 4781938049531404654ull;
                                    }
                                }
                                else
                                {
                                    unique_id = 4781938049531404654ull;
                                }
                            }
                            else
                            {
                                unique_id = 4781938049531404654ull;
                            }
                        }
                        else
                        {
                            if(d6 <= 33)
                            {
                                if(d6 <= 21)
                                {
                                    if(d6 <= 10)
                                    {
                                        unique_id = 4781938049531404654ull;
                                    }
                                    else
                                    {
                                        if(d1 <= 419)
                                        {
                                            unique_id = 4781938049531404654ull;
                                        }
                                        else
                                        {
                                            if(d3 <= 432)
                                            {
                                                unique_id = 4781938049531404654ull;
                                            }
                                            else
                                            {
                                                if(d5 <= 9)
                                                {
                                                    unique_id = 4781938049531404654ull;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142ull;
                                                }
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    unique_id = 4781938049531404654ull;
                                }
                            }
                            else
                            {
                                if(d2 <= 259)
                                {
                                    if(d5 <= 12)
                                    {
                                        unique_id = 4781938049531404654ull;
                                    }
                                    else
                                    {
                                        if(d1 <= 337)
                                        {
                                            unique_id = 4781938049531404654ull;
                                        }
                                        else
                                        {
                                            unique_id = 16334338346691940719ull;
                                        }
                                    }
                                }
                                else
                                {
                                    if(d3 <= 112)
                                    {
                                        unique_id = 5897150915348629714ull;
                                    }
                                    else
                                    {
                                        if(d1 <= 329)
                                        {
                                            if(d3 <= 519)
                                            {
                                                unique_id = 4781938049531404654ull;
                                            }
                                            else
                                            {
                                                if(d4 <= 151)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            unique_id = 16334338346691940719ull;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        if(d5 <= 50)
                        {
                            if(d3 <= 175)
                            {
                                if(d6 <= 24)
                                {
                                    if(d5 <= 29)
                                    {
                                        if(d6 <= 16)
                                        {
                                            unique_id = 4781938049531404654ull;
                                        }
                                        else
                                        {
                                            if(d5 <= 24)
                                            {
                                                unique_id = 4781938049531404654ull;
                                            }
                                            else
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d2 <= 528)
                                        {
                                            if(d6 <= 16)
                                            {
                                                if(d5 <= 32)
                                                {
                                                    unique_id = 4781938049531404654ull;
                                                }
                                                else
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                        }
                                        else
                                        {
                                            if(d3 <= 102)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                if(d4 <= 246)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d2 <= 680)
                                    {
                                        if(d3 <= 119)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            if(d2 <= 509)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                if(d4 <= 158)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d3 <= 108)
                                        {
                                            if(d3 <= 43)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                if(d4 <= 265)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d4 <= 152)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                unique_id = 16334338346691940719ull;
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d4 <= 151)
                                {
                                    if(d6 <= 16)
                                    {
                                        if(d5 <= 32)
                                        {
                                            unique_id = 4781938049531404654ull;
                                        }
                                        else
                                        {
                                            if(d6 <= 12)
                                            {
                                                unique_id = 4781938049531404654ull;
                                            }
                                            else
                                            {
                                                if(d2 <= 543)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d2 <= 746)
                                        {
                                            if(d6 <= 24)
                                            {
                                                if(d5 <= 32)
                                                {
                                                    unique_id = 4781938049531404654ull;
                                                }
                                                else
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                        }
                                        else
                                        {
                                            if(d4 <= 51)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                unique_id = 16334338346691940719ull;
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d6 <= 16)
                                    {
                                        if(d5 <= 29)
                                        {
                                            if(d5 <= 24)
                                            {
                                                unique_id = 4781938049531404654ull;
                                            }
                                            else
                                            {
                                                if(d6 <= 11)
                                                {
                                                    unique_id = 4781938049531404654ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d2 <= 168)
                                            {
                                                if(d2 <= 65)
                                                {
                                                    unique_id = 4781938049531404654ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 16334338346691940719ull;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d2 <= 68)
                                        {
                                            if(d5 <= 32)
                                            {
                                                if(d6 <= 24)
                                                {
                                                    unique_id = 4781938049531404654ull;
                                                }
                                                else
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                        }
                                        else
                                        {
                                            if(d1 <= 156)
                                            {
                                                if(d2 <= 505)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 16334338346691940719ull;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            if(d6 <= 20)
                            {
                                if(d4 <= 151)
                                {
                                    if(d2 <= 529)
                                    {
                                        unique_id = 5897150915348629714ull;
                                    }
                                    else
                                    {
                                        if(d4 <= 62)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            if(d3 <= 227)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                if(d1 <= 335)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d3 <= 119)
                                    {
                                        if(d2 <= 585)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            if(d3 <= 48)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                unique_id = 16334338346691940719ull;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d2 <= 81)
                                        {
                                            if(d1 <= 339)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                if(d2 <= 41)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            unique_id = 16334338346691940719ull;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d2 <= 420)
                                {
                                    if(d4 <= 151)
                                    {
                                        unique_id = 5897150915348629714ull;
                                    }
                                    else
                                    {
                                        if(d3 <= 186)
                                        {
                                            if(d4 <= 772)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                if(d3 <= 99)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142ull;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d2 <= 68)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                if(d5 <= 56)
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142ull;
                                                }
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d4 <= 150)
                                    {
                                        if(d4 <= 57)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            if(d3 <= 178)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                if(d3 <= 775)
                                                {
                                                    unique_id = 10972102817010133142ull;
                                                }
                                                else
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d2 <= 887)
                                        {
                                            if(d3 <= 48)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142ull;
                                            }
                                        }
                                        else
                                        {
                                            if(d3 <= 327)
                                            {
                                                if(d3 <= 49)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142ull;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 16334338346691940719ull;
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
                if(d5 <= 37)
                {
                    if(d5 <= 4)
                    {
                        unique_id = 4781938049531404654ull;
                    }
                    else
                    {
                        if(d5 <= 24)
                        {
                            if(d5 <= 8)
                            {
                                if(d6 <= 64)
                                {
                                    if(d5 <= 5)
                                    {
                                        if(d6 <= 48)
                                        {
                                            unique_id = 4781938049531404654ull;
                                        }
                                        else
                                        {
                                            if(d2 <= 510)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142ull;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        unique_id = 4781938049531404654ull;
                                    }
                                }
                                else
                                {
                                    if(d2 <= 686)
                                    {
                                        if(d5 <= 6)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            if(d1 <= 574)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                unique_id = 16334338346691940719ull;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d4 <= 457)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            unique_id = 10972102817010133142ull;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d2 <= 306)
                                {
                                    if(d1 <= 578)
                                    {
                                        if(d5 <= 23)
                                        {
                                            if(d6 <= 52)
                                            {
                                                if(d5 <= 12)
                                                {
                                                    unique_id = 4781938049531404654ull;
                                                }
                                                else
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                        }
                                        else
                                        {
                                            unique_id = 4781938049531404654ull;
                                        }
                                    }
                                    else
                                    {
                                        if(d3 <= 190)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            if(d2 <= 63)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                if(d4 <= 140)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d4 <= 257)
                                    {
                                        if(d5 <= 23)
                                        {
                                            if(d4 <= 115)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                if(d1 <= 578)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            unique_id = 4781938049531404654ull;
                                        }
                                    }
                                    else
                                    {
                                        if(d3 <= 180)
                                        {
                                            if(d1 <= 578)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                if(d3 <= 96)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d5 <= 23)
                                            {
                                                unique_id = 16334338346691940719ull;
                                            }
                                            else
                                            {
                                                if(d6 <= 55)
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                                else
                                                {
                                                    unique_id = 4781938049531404654ull;
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
                                if(d3 <= 212)
                                {
                                    if(d4 <= 744)
                                    {
                                        if(d3 <= 114)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            if(d2 <= 431)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                if(d1 <= 272)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142ull;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d2 <= 406)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            unique_id = 10972102817010133142ull;
                                        }
                                    }
                                }
                                else
                                {
                                    if(d4 <= 130)
                                    {
                                        unique_id = 5897150915348629714ull;
                                    }
                                    else
                                    {
                                        if(d2 <= 169)
                                        {
                                            if(d2 <= 66)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                if(d1 <= 274)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d1 <= 203)
                                            {
                                                if(d2 <= 514)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 16334338346691940719ull;
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d2 <= 421)
                                {
                                    if(d1 <= 207)
                                    {
                                        if(d2 <= 174)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                    }
                                    else
                                    {
                                        if(d2 <= 66)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            if(d3 <= 186)
                                            {
                                                if(d4 <= 760)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142ull;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142ull;
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d4 <= 59)
                                    {
                                        unique_id = 5897150915348629714ull;
                                    }
                                    else
                                    {
                                        if(d1 <= 105)
                                        {
                                            if(d2 <= 684)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142ull;
                                            }
                                        }
                                        else
                                        {
                                            if(d3 <= 102)
                                            {
                                                if(d3 <= 24)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142ull;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142ull;
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
                    if(d2 <= 171)
                    {
                        if(d1 <= 214)
                        {
                            if(d4 <= 752)
                            {
                                if(d2 <= 67)
                                {
                                    unique_id = 5897150915348629714ull;
                                }
                                else
                                {
                                    if(d1 <= 104)
                                    {
                                        unique_id = 5897150915348629714ull;
                                    }
                                    else
                                    {
                                        if(d6 <= 61)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            unique_id = 10972102817010133142ull;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d5 <= 52)
                                {
                                    unique_id = 5897150915348629714ull;
                                }
                                else
                                {
                                    unique_id = 10972102817010133142ull;
                                }
                            }
                        }
                        else
                        {
                            if(d2 <= 37)
                            {
                                if(d6 <= 60)
                                {
                                    unique_id = 5897150915348629714ull;
                                }
                                else
                                {
                                    if(d2 <= 18)
                                    {
                                        unique_id = 5897150915348629714ull;
                                    }
                                    else
                                    {
                                        if(d1 <= 514)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            if(d5 <= 51)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142ull;
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d5 <= 50)
                                {
                                    if(d6 <= 60)
                                    {
                                        if(d4 <= 769)
                                        {
                                            if(d3 <= 288)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                if(d2 <= 68)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            unique_id = 10972102817010133142ull;
                                        }
                                    }
                                    else
                                    {
                                        if(d3 <= 152)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            if(d4 <= 90)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142ull;
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d6 <= 59)
                                    {
                                        if(d2 <= 67)
                                        {
                                            if(d4 <= 755)
                                            {
                                                if(d1 <= 649)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142ull;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142ull;
                                            }
                                        }
                                        else
                                        {
                                            if(d4 <= 225)
                                            {
                                                unique_id = 10972102817010133142ull;
                                            }
                                            else
                                            {
                                                if(d3 <= 50)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142ull;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d3 <= 38)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            unique_id = 10972102817010133142ull;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        if(d5 <= 50)
                        {
                            if(d6 <= 58)
                            {
                                if(d1 <= 204)
                                {
                                    if(d2 <= 505)
                                    {
                                        unique_id = 5897150915348629714ull;
                                    }
                                    else
                                    {
                                        if(d4 <= 148)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            if(d1 <= 27)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142ull;
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d3 <= 328)
                                    {
                                        if(d3 <= 50)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            if(d4 <= 73)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142ull;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d6 <= 50)
                                        {
                                            if(d4 <= 140)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                if(d4 <= 774)
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142ull;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d2 <= 420)
                                            {
                                                if(d4 <= 148)
                                                {
                                                    unique_id = 5897150915348629714ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142ull;
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d1 <= 104)
                                {
                                    if(d2 <= 517)
                                    {
                                        unique_id = 5897150915348629714ull;
                                    }
                                    else
                                    {
                                        if(d4 <= 119)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            if(d1 <= 19)
                                            {
                                                unique_id = 5897150915348629714ull;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142ull;
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d4 <= 60)
                                    {
                                        if(d1 <= 499)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            unique_id = 10972102817010133142ull;
                                        }
                                    }
                                    else
                                    {
                                        if(d3 <= 39)
                                        {
                                            unique_id = 10972102817010133142ull;
                                        }
                                        else
                                        {
                                            if(d2 <= 422)
                                            {
                                                unique_id = 10972102817010133142ull;
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142ull;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            if(d1 <= 46)
                            {
                                if(d4 <= 335)
                                {
                                    unique_id = 5897150915348629714ull;
                                }
                                else
                                {
                                    if(d1 <= 16)
                                    {
                                        unique_id = 5897150915348629714ull;
                                    }
                                    else
                                    {
                                        unique_id = 10972102817010133142ull;
                                    }
                                }
                            }
                            else
                            {
                                if(d4 <= 59)
                                {
                                    if(d1 <= 321)
                                    {
                                        if(d3 <= 167)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            unique_id = 10972102817010133142ull;
                                        }
                                    }
                                    else
                                    {
                                        if(d3 <= 172)
                                        {
                                            unique_id = 10972102817010133142ull;
                                        }
                                        else
                                        {
                                            unique_id = 10972102817010133142ull;
                                        }
                                    }
                                }
                                else
                                {
                                    if(d3 <= 26)
                                    {
                                        if(d6 <= 61)
                                        {
                                            unique_id = 5897150915348629714ull;
                                        }
                                        else
                                        {
                                            unique_id = 10972102817010133142ull;
                                        }
                                    }
                                    else
                                    {
                                        if(d6 <= 50)
                                        {
                                            if(d5 <= 55)
                                            {
                                                if(d3 <= 790)
                                                {
                                                    unique_id = 10972102817010133142ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                            else
                                            {
                                                if(d2 <= 988)
                                                {
                                                    unique_id = 10972102817010133142ull;
                                                }
                                                else
                                                {
                                                    unique_id = 16334338346691940719ull;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            if(d1 <= 205)
                                            {
                                                if(d4 <= 155)
                                                {
                                                    unique_id = 10972102817010133142ull;
                                                }
                                                else
                                                {
                                                    unique_id = 10972102817010133142ull;
                                                }
                                            }
                                            else
                                            {
                                                unique_id = 10972102817010133142ull;
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

            if(d6 <= 40)
            {
                if(d5 <= 4)
                {
                    if(d6 <= 17)
                    {
                        unique_id = 8512122806347362414ull;
                    }
                    else
                    {
                        if(d5 <= 1)
                        {
                            unique_id = 8512122806347362414ull;
                        }
                        else
                        {
                            unique_id = 6766074198369117690ull;
                        }
                    }
                }
                else
                {
                    if(d6 <= 2)
                    {
                        if(d5 <= 37)
                        {
                            unique_id = 8512122806347362414ull;
                        }
                        else
                        {
                            unique_id = 6766074198369117690ull;
                        }
                    }
                    else
                    {
                        if(d6 <= 33)
                        {
                            unique_id = 6766074198369117690ull;
                        }
                        else
                        {
                            if(d5 <= 57)
                            {
                                unique_id = 6766074198369117690ull;
                            }
                            else
                            {
                                unique_id = 5863879104640185307ull;
                            }
                        }
                    }
                }
            }
            else
            {
                if(d5 <= 39)
                {
                    if(d5 <= 33)
                    {
                        if(d5 <= 1)
                        {
                            unique_id = 8512122806347362414ull;
                        }
                        else
                        {
                            if(d6 <= 63)
                            {
                                unique_id = 6766074198369117690ull;
                            }
                            else
                            {
                                if(d5 <= 28)
                                {
                                    unique_id = 6766074198369117690ull;
                                }
                                else
                                {
                                    if(d1 <= 253)
                                    {
                                        unique_id = 6766074198369117690ull;
                                    }
                                    else
                                    {
                                        unique_id = 8205318106496093444ull;
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        if(d6 <= 59)
                        {
                            unique_id = 6766074198369117690ull;
                        }
                        else
                        {
                            if(d1 <= 254)
                            {
                                unique_id = 6766074198369117690ull;
                            }
                            else
                            {
                                unique_id = 5863879104640185307ull;
                            }
                        }
                    }
                }
                else
                {
                    if(d6 <= 63)
                    {
                        if(d5 <= 63)
                        {
                            if(d5 <= 48)
                            {
                                if(d6 <= 50)
                                {
                                    unique_id = 6766074198369117690ull;
                                }
                                else
                                {
                                    if(d5 <= 47)
                                    {
                                        unique_id = 5863879104640185307ull;
                                    }
                                    else
                                    {
                                        unique_id = 8205318106496093444ull;
                                    }
                                }
                            }
                            else
                            {
                                if(d5 <= 55)
                                {
                                    unique_id = 5863879104640185307ull;
                                }
                                else
                                {
                                    if(d6 <= 55)
                                    {
                                        if(d5 <= 56)
                                        {
                                            unique_id = 8205318106496093444ull;
                                        }
                                        else
                                        {
                                            if(d6 <= 47)
                                            {
                                                unique_id = 5863879104640185307ull;
                                            }
                                            else
                                            {
                                                if(d6 <= 48)
                                                {
                                                    unique_id = 8205318106496093444ull;
                                                }
                                                else
                                                {
                                                    unique_id = 5863879104640185307ull;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        unique_id = 5863879104640185307ull;
                                    }
                                }
                            }
                        }
                        else
                        {
                            if(d6 <= 47)
                            {
                                if(d5 <= 64)
                                {
                                    unique_id = 8205318106496093444ull;
                                }
                                else
                                {
                                    unique_id = 5863879104640185307ull;
                                }
                            }
                            else
                            {
                                if(d6 <= 55)
                                {
                                    if(d5 <= 71)
                                    {
                                        if(d5 <= 64)
                                        {
                                            unique_id = 8205318106496093444ull;
                                        }
                                        else
                                        {
                                            if(d6 <= 48)
                                            {
                                                unique_id = 8205318106496093444ull;
                                            }
                                            else
                                            {
                                                unique_id = 5863879104640185307ull;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        unique_id = 8205318106496093444ull;
                                    }
                                }
                                else
                                {
                                    unique_id = 8205318106496093444ull;
                                }
                            }
                        }
                    }
                    else
                    {
                        if(d5 <= 51)
                        {
                            if(d6 <= 64)
                            {
                                unique_id = 8205318106496093444ull;
                            }
                            else
                            {
                                if(d5 <= 47)
                                {
                                    unique_id = 5863879104640185307ull;
                                }
                                else
                                {
                                    if(d5 <= 48)
                                    {
                                        unique_id = 8205318106496093444ull;
                                    }
                                    else
                                    {
                                        unique_id = 5863879104640185307ull;
                                    }
                                }
                            }
                        }
                        else
                        {
                            unique_id = 8205318106496093444ull;
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

    template <>
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::BILINEAR>
    {
        static hiptensorStatus_t
            selectWinner(ContractionSolution**                                   winner,
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

            if(d6 <= 33)
            {
                if(d5 <= 2)
                {
                    if(d6 <= 7)
                    {
                        unique_id = 2415201861364427301ull;
                    }
                    else
                    {
                        unique_id = 6902287373350592119ull;
                    }
                }
                else
                {
                    if(d5 <= 65)
                    {
                        if(d6 <= 2)
                        {
                            if(d5 <= 13)
                            {
                                unique_id = 2415201861364427301ull;
                            }
                            else
                            {
                                unique_id = 550562982100054122ull;
                            }
                        }
                        else
                        {
                            if(d5 <= 7)
                            {
                                if(d6 <= 7)
                                {
                                    unique_id = 6902287373350592119ull;
                                }
                                else
                                {
                                    unique_id = 550562982100054122ull;
                                }
                            }
                            else
                            {
                                unique_id = 550562982100054122ull;
                            }
                        }
                    }
                    else
                    {
                        if(d6 <= 25)
                        {
                            unique_id = 550562982100054122ull;
                        }
                        else
                        {
                            unique_id = 3324992315903551472ull;
                        }
                    }
                }
            }
            else
            {
                if(d5 <= 32)
                {
                    if(d6 <= 64)
                    {
                        unique_id = 550562982100054122ull;
                    }
                    else
                    {
                        if(d5 <= 24)
                        {
                            if(d5 <= 1)
                            {
                                unique_id = 3324992315903551472ull;
                            }
                            else
                            {
                                unique_id = 550562982100054122ull;
                            }
                        }
                        else
                        {
                            if(d2 <= 223)
                            {
                                unique_id = 550562982100054122ull;
                            }
                            else
                            {
                                unique_id = 3324992315903551472ull;
                            }
                        }
                    }
                }
                else
                {
                    if(d6 <= 54)
                    {
                        if(d5 <= 48)
                        {
                            if(d6 <= 48)
                            {
                                unique_id = 550562982100054122ull;
                            }
                            else
                            {
                                if(d5 <= 36)
                                {
                                    unique_id = 550562982100054122ull;
                                }
                                else
                                {
                                    unique_id = 3324992315903551472ull;
                                }
                            }
                        }
                        else
                        {
                            if(d6 <= 47)
                            {
                                if(d5 <= 56)
                                {
                                    if(d6 <= 40)
                                    {
                                        unique_id = 550562982100054122ull;
                                    }
                                    else
                                    {
                                        unique_id = 3324992315903551472ull;
                                    }
                                }
                                else
                                {
                                    unique_id = 3324992315903551472ull;
                                }
                            }
                            else
                            {
                                if(d5 <= 63)
                                {
                                    unique_id = 3324992315903551472ull;
                                }
                                else
                                {
                                    unique_id = 4157899012150127975ull;
                                }
                            }
                        }
                    }
                    else
                    {
                        if(d5 <= 54)
                        {
                            if(d5 <= 47)
                            {
                                unique_id = 3324992315903551472ull;
                            }
                            else
                            {
                                if(d6 <= 67)
                                {
                                    unique_id = 3324992315903551472ull;
                                }
                                else
                                {
                                    if(d1 <= 416)
                                    {
                                        unique_id = 3324992315903551472ull;
                                    }
                                    else
                                    {
                                        unique_id = 4157899012150127975ull;
                                    }
                                }
                            }
                        }
                        else
                        {
                            if(d5 <= 63)
                            {
                                if(d6 <= 63)
                                {
                                    if(d3 <= 252)
                                    {
                                        unique_id = 4157899012150127975ull;
                                    }
                                    else
                                    {
                                        unique_id = 3324992315903551472ull;
                                    }
                                }
                                else
                                {
                                    unique_id = 4157899012150127975ull;
                                }
                            }
                            else
                            {
                                unique_id = 4157899012150127975ull;
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
