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
            return HIPTENSOR_STATUS_SUCCESS;
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
