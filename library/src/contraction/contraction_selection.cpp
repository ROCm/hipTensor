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

            if(d6 <= 43)
            {
                if(d5 <= 61)
                {
                    if(d3 <= 236)
                    {
                        if(d4 <= 519)
                        {
                            if(d1 <= 744)
                            {
                                if(d6 <= 8)
                                {
                                    unique_id = 4671301146928673150ull;
                                }
                                else
                                {
                                    unique_id = 17304057348073251997ull;
                                }
                            }
                            else
                            {
                                unique_id = 4671301146928673150ull;
                            }
                        }
                        else
                        {
                            if(d3 <= 32)
                            {
                                unique_id = 17304057348073251997ull;
                            }
                            else
                            {
                                unique_id = 4671301146928673150ull;
                            }
                        }
                    }
                    else
                    {
                        if(d6 <= 2)
                        {
                            if(d5 <= 15)
                            {
                                unique_id = 17618515137355245877ull;
                            }
                            else
                            {
                                if(d6 <= 1)
                                {
                                    unique_id = 10830479759059230274ull;
                                }
                                else
                                {
                                    if(d5 <= 32)
                                    {
                                        unique_id = 10830479759059230274ull;
                                    }
                                    else
                                    {
                                        unique_id = 4671301146928673150ull;
                                    }
                                }
                            }
                        }
                        else
                        {
                            if(d5 <= 2)
                            {
                                if(d6 <= 8)
                                {
                                    unique_id = 17618515137355245877ull;
                                }
                                else
                                {
                                    unique_id = 10830479759059230274ull;
                                }
                            }
                            else
                            {
                                if(d1 <= 54)
                                {
                                    unique_id = 17304057348073251997ull;
                                }
                                else
                                {
                                    if(d4 <= 218)
                                    {
                                        if(d5 <= 36)
                                        {
                                            unique_id = 4671301146928673150ull;
                                        }
                                        else
                                        {
                                            if(d6 <= 31)
                                            {
                                                unique_id = 4671301146928673150ull;
                                            }
                                            else
                                            {
                                                unique_id = 16481146763982821264ull;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d2 <= 50)
                                        {
                                            unique_id = 4671301146928673150ull;
                                        }
                                        else
                                        {
                                            if(d6 <= 31)
                                            {
                                                unique_id = 4671301146928673150ull;
                                            }
                                            else
                                            {
                                                if(d6 <= 32)
                                                {
                                                    unique_id = 10830479759059230274ull;
                                                }
                                                else
                                                {
                                                    unique_id = 4671301146928673150ull;
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
                    if(d6 <= 18)
                    {
                        unique_id = 4671301146928673150ull;
                    }
                    else
                    {
                        if(d4 <= 557)
                        {
                            if(d2 <= 165)
                            {
                                unique_id = 4671301146928673150ull;
                            }
                            else
                            {
                                unique_id = 16481146763982821264ull;
                            }
                        }
                        else
                        {
                            if(d5 <= 68)
                            {
                                unique_id = 4671301146928673150ull;
                            }
                            else
                            {
                                unique_id = 16481146763982821264ull;
                            }
                        }
                    }
                }
            }
            else
            {
                if(d5 <= 24)
                {
                    if(d3 <= 435)
                    {
                        if(d5 <= 7)
                        {
                            if(d5 <= 1)
                            {
                                unique_id = 3454820663416883703ull;
                            }
                            else
                            {
                                unique_id = 4671301146928673150ull;
                            }
                        }
                        else
                        {
                            if(d1 <= 744)
                            {
                                unique_id = 17304057348073251997ull;
                            }
                            else
                            {
                                if(d6 <= 60)
                                {
                                    unique_id = 4671301146928673150ull;
                                }
                                else
                                {
                                    unique_id = 17304057348073251997ull;
                                }
                            }
                        }
                    }
                    else
                    {
                        if(d5 <= 1)
                        {
                            unique_id = 3454820663416883703ull;
                        }
                        else
                        {
                            if(d5 <= 13)
                            {
                                if(d5 <= 7)
                                {
                                    unique_id = 4671301146928673150ull;
                                }
                                else
                                {
                                    unique_id = 4671301146928673150ull;
                                }
                            }
                            else
                            {
                                if(d6 <= 58)
                                {
                                    unique_id = 4671301146928673150ull;
                                }
                                else
                                {
                                    if(d1 <= 642)
                                    {
                                        unique_id = 17304057348073251997ull;
                                    }
                                    else
                                    {
                                        unique_id = 16481146763982821264ull;
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    if(d6 <= 54)
                    {
                        if(d5 <= 37)
                        {
                            if(d4 <= 556)
                            {
                                unique_id = 16481146763982821264ull;
                            }
                            else
                            {
                                unique_id = 4671301146928673150ull;
                            }
                        }
                        else
                        {
                            if(d1 <= 222)
                            {
                                if(d4 <= 556)
                                {
                                    unique_id = 16481146763982821264ull;
                                }
                                else
                                {
                                    unique_id = 4671301146928673150ull;
                                }
                            }
                            else
                            {
                                unique_id = 16481146763982821264ull;
                            }
                        }
                    }
                    else
                    {
                        if(d4 <= 44)
                        {
                            if(d3 <= 436)
                            {
                                unique_id = 17304057348073251997ull;
                            }
                            else
                            {
                                unique_id = 16481146763982821264ull;
                            }
                        }
                        else
                        {
                            if(d1 <= 220)
                            {
                                if(d2 <= 107)
                                {
                                    unique_id = 17304057348073251997ull;
                                }
                                else
                                {
                                    unique_id = 16481146763982821264ull;
                                }
                            }
                            else
                            {
                                if(d3 <= 72)
                                {
                                    unique_id = 16481146763982821264ull;
                                }
                                else
                                {
                                    if(d2 <= 18)
                                    {
                                        unique_id = 4671301146928673150ull;
                                    }
                                    else
                                    {
                                        unique_id = 16481146763982821264ull;
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

            if(d6 <= 9)
            {
                if(d6 <= 4)
                {
                    unique_id = 9622108777680582053ull;
                }
                else
                {
                    if(d5 <= 16)
                    {
                        unique_id = 9622108777680582053ull;
                    }
                    else
                    {
                        if(d2 <= 196)
                        {
                            unique_id = 9622108777680582053ull;
                        }
                        else
                        {
                            if(d1 <= 113)
                            {
                                unique_id = 9622108777680582053ull;
                            }
                            else
                            {
                                if(d3 <= 219)
                                {
                                    unique_id = 9622108777680582053ull;
                                }
                                else
                                {
                                    unique_id = 13257779901106960809ull;
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
                    if(d6 <= 28)
                    {
                        unique_id = 9622108777680582053ull;
                    }
                    else
                    {
                        if(d5 <= 2)
                        {
                            if(d6 <= 58)
                            {
                                unique_id = 9622108777680582053ull;
                            }
                            else
                            {
                                if(d5 <= 1)
                                {
                                    unique_id = 9622108777680582053ull;
                                }
                                else
                                {
                                    unique_id = 13257779901106960809ull;
                                }
                            }
                        }
                        else
                        {
                            if(d2 <= 163)
                            {
                                unique_id = 9622108777680582053ull;
                            }
                            else
                            {
                                if(d1 <= 465)
                                {
                                    unique_id = 9622108777680582053ull;
                                }
                                else
                                {
                                    unique_id = 13257779901106960809ull;
                                }
                            }
                        }
                    }
                }
                else
                {
                    if(d3 <= 121)
                    {
                        if(d4 <= 483)
                        {
                            if(d6 <= 29)
                            {
                                if(d5 <= 32)
                                {
                                    unique_id = 9622108777680582053ull;
                                }
                                else
                                {
                                    unique_id = 222393107113976106ull;
                                }
                            }
                            else
                            {
                                if(d5 <= 39)
                                {
                                    unique_id = 222393107113976106ull;
                                }
                                else
                                {
                                    if(d2 <= 152)
                                    {
                                        unique_id = 222393107113976106ull;
                                    }
                                    else
                                    {
                                        unique_id = 13257779901106960809ull;
                                    }
                                }
                            }
                        }
                        else
                        {
                            if(d3 <= 37)
                            {
                                unique_id = 222393107113976106ull;
                            }
                            else
                            {
                                if(d6 <= 29)
                                {
                                    if(d5 <= 32)
                                    {
                                        unique_id = 9622108777680582053ull;
                                    }
                                    else
                                    {
                                        unique_id = 15066925687960442338ull;
                                    }
                                }
                                else
                                {
                                    unique_id = 15066925687960442338ull;
                                }
                            }
                        }
                    }
                    else
                    {
                        if(d4 <= 135)
                        {
                            if(d3 <= 413)
                            {
                                if(d6 <= 30)
                                {
                                    if(d5 <= 32)
                                    {
                                        unique_id = 9622108777680582053ull;
                                    }
                                    else
                                    {
                                        unique_id = 222393107113976106ull;
                                    }
                                }
                                else
                                {
                                    if(d5 <= 39)
                                    {
                                        unique_id = 222393107113976106ull;
                                    }
                                    else
                                    {
                                        unique_id = 13257779901106960809ull;
                                    }
                                }
                            }
                            else
                            {
                                if(d4 <= 36)
                                {
                                    unique_id = 222393107113976106ull;
                                }
                                else
                                {
                                    if(d2 <= 120)
                                    {
                                        unique_id = 222393107113976106ull;
                                    }
                                    else
                                    {
                                        if(d6 <= 32)
                                        {
                                            if(d5 <= 32)
                                            {
                                                unique_id = 13257779901106960809ull;
                                            }
                                            else
                                            {
                                                unique_id = 15066925687960442338ull;
                                            }
                                        }
                                        else
                                        {
                                            unique_id = 15066925687960442338ull;
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            if(d2 <= 115)
                            {
                                if(d6 <= 40)
                                {
                                    if(d2 <= 51)
                                    {
                                        unique_id = 222393107113976106ull;
                                    }
                                    else
                                    {
                                        if(d5 <= 32)
                                        {
                                            unique_id = 9622108777680582053ull;
                                        }
                                        else
                                        {
                                            if(d4 <= 486)
                                            {
                                                unique_id = 222393107113976106ull;
                                            }
                                            else
                                            {
                                                unique_id = 15066925687960442338ull;
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    if(d1 <= 235)
                                    {
                                        unique_id = 222393107113976106ull;
                                    }
                                    else
                                    {
                                        if(d2 <= 22)
                                        {
                                            unique_id = 222393107113976106ull;
                                        }
                                        else
                                        {
                                            unique_id = 15066925687960442338ull;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if(d6 <= 32)
                                {
                                    if(d5 <= 26)
                                    {
                                        if(d6 <= 23)
                                        {
                                            if(d1 <= 116)
                                            {
                                                unique_id = 9622108777680582053ull;
                                            }
                                            else
                                            {
                                                unique_id = 13257779901106960809ull;
                                            }
                                        }
                                        else
                                        {
                                            if(d5 <= 18)
                                            {
                                                unique_id = 13257779901106960809ull;
                                            }
                                            else
                                            {
                                                unique_id = 15066925687960442338ull;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if(d5 <= 32)
                                        {
                                            if(d6 <= 16)
                                            {
                                                unique_id = 13257779901106960809ull;
                                            }
                                            else
                                            {
                                                unique_id = 15066925687960442338ull;
                                            }
                                        }
                                        else
                                        {
                                            unique_id = 15066925687960442338ull;
                                        }
                                    }
                                }
                                else
                                {
                                    unique_id = 15066925687960442338ull;
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

            if(d5 <= 36)
            {
                if(d6 <= 35)
                {
                    if(d1 <= 763)
                    {
                        if(d6 <= 3)
                        {
                            if(d5 <= 8)
                            {
                                unique_id = 9769367948782541618ull;
                            }
                            else
                            {
                                unique_id = 3344638327382374968ull;
                            }
                        }
                        else
                        {
                            unique_id = 3344638327382374968ull;
                        }
                    }
                    else
                    {
                        if(d6 <= 24)
                        {
                            unique_id = 3344638327382374968ull;
                        }
                        else
                        {
                            if(d5 <= 17)
                            {
                                unique_id = 3344638327382374968ull;
                            }
                            else
                            {
                                unique_id = 2770278462698889442ull;
                            }
                        }
                    }
                }
                else
                {
                    if(d5 <= 9)
                    {
                        unique_id = 3344638327382374968ull;
                    }
                    else
                    {
                        if(d1 <= 759)
                        {
                            if(d6 <= 67)
                            {
                                if(d3 <= 535)
                                {
                                    unique_id = 3344638327382374968ull;
                                }
                                else
                                {
                                    if(d4 <= 615)
                                    {
                                        unique_id = 3344638327382374968ull;
                                    }
                                    else
                                    {
                                        unique_id = 2770278462698889442ull;
                                    }
                                }
                            }
                            else
                            {
                                if(d5 <= 25)
                                {
                                    if(d4 <= 428)
                                    {
                                        unique_id = 3344638327382374968ull;
                                    }
                                    else
                                    {
                                        unique_id = 2770278462698889442ull;
                                    }
                                }
                                else
                                {
                                    unique_id = 16588612317409292216ull;
                                }
                            }
                        }
                        else
                        {
                            if(d6 <= 64)
                            {
                                if(d3 <= 65)
                                {
                                    unique_id = 3344638327382374968ull;
                                }
                                else
                                {
                                    unique_id = 2770278462698889442ull;
                                }
                            }
                            else
                            {
                                if(d5 <= 25)
                                {
                                    unique_id = 2770278462698889442ull;
                                }
                                else
                                {
                                    unique_id = 16588612317409292216ull;
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                if(d6 <= 33)
                {
                    if(d6 <= 8)
                    {
                        unique_id = 3344638327382374968ull;
                    }
                    else
                    {
                        if(d2 <= 565)
                        {
                            if(d1 <= 646)
                            {
                                unique_id = 3344638327382374968ull;
                            }
                            else
                            {
                                if(d6 <= 27)
                                {
                                    unique_id = 3344638327382374968ull;
                                }
                                else
                                {
                                    if(d5 <= 53)
                                    {
                                        unique_id = 2770278462698889442ull;
                                    }
                                    else
                                    {
                                        unique_id = 16588612317409292216ull;
                                    }
                                }
                            }
                        }
                        else
                        {
                            if(d6 <= 20)
                            {
                                if(d3 <= 168)
                                {
                                    unique_id = 3344638327382374968ull;
                                }
                                else
                                {
                                    unique_id = 2770278462698889442ull;
                                }
                            }
                            else
                            {
                                if(d5 <= 64)
                                {
                                    if(d1 <= 648)
                                    {
                                        unique_id = 3344638327382374968ull;
                                    }
                                    else
                                    {
                                        unique_id = 2770278462698889442ull;
                                    }
                                }
                                else
                                {
                                    if(d6 <= 25)
                                    {
                                        unique_id = 3344638327382374968ull;
                                    }
                                    else
                                    {
                                        unique_id = 16588612317409292216ull;
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    if(d5 <= 45)
                    {
                        if(d6 <= 50)
                        {
                            if(d3 <= 168)
                            {
                                unique_id = 3344638327382374968ull;
                            }
                            else
                            {
                                unique_id = 2770278462698889442ull;
                            }
                        }
                        else
                        {
                            unique_id = 16588612317409292216ull;
                        }
                    }
                    else
                    {
                        if(d6 <= 43)
                        {
                            if(d5 <= 52)
                            {
                                unique_id = 2770278462698889442ull;
                            }
                            else
                            {
                                unique_id = 16588612317409292216ull;
                            }
                        }
                        else
                        {
                            unique_id = 16588612317409292216ull;
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

            if(d5 <= 39)
            {
                if(d3 <= 937)
                {
                    if(d6 <= 1)
                    {
                        unique_id = 1830537384143755749ull;
                    }
                    else
                    {
                        if(d4 <= 754)
                        {
                            if(d5 <= 33)
                            {
                                if(d5 <= 1)
                                {
                                    if(d6 <= 25)
                                    {
                                        unique_id = 3423207643344265161ull;
                                    }
                                    else
                                    {
                                        unique_id = 1830537384143755749ull;
                                    }
                                }
                                else
                                {
                                    if(d6 <= 6)
                                    {
                                        if(d5 <= 8)
                                        {
                                            unique_id = 3423207643344265161ull;
                                        }
                                        else
                                        {
                                            unique_id = 1830537384143755749ull;
                                        }
                                    }
                                    else
                                    {
                                        unique_id = 1830537384143755749ull;
                                    }
                                }
                            }
                            else
                            {
                                unique_id = 1830537384143755749ull;
                            }
                        }
                        else
                        {
                            if(d1 <= 404)
                            {
                                unique_id = 1830537384143755749ull;
                            }
                            else
                            {
                                if(d6 <= 50)
                                {
                                    unique_id = 1830537384143755749ull;
                                }
                                else
                                {
                                    if(d5 <= 33)
                                    {
                                        unique_id = 1830537384143755749ull;
                                    }
                                    else
                                    {
                                        unique_id = 4992687403741300893ull;
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    unique_id = 1830537384143755749ull;
                }
            }
            else
            {
                if(d6 <= 32)
                {
                    if(d2 <= 832)
                    {
                        unique_id = 1830537384143755749ull;
                    }
                    else
                    {
                        if(d6 <= 8)
                        {
                            unique_id = 1830537384143755749ull;
                        }
                        else
                        {
                            if(d6 <= 24)
                            {
                                unique_id = 17689908062647780665ull;
                            }
                            else
                            {
                                if(d5 <= 64)
                                {
                                    unique_id = 1830537384143755749ull;
                                }
                                else
                                {
                                    unique_id = 4992687403741300893ull;
                                }
                            }
                        }
                    }
                }
                else
                {
                    if(d6 <= 46)
                    {
                        if(d5 <= 54)
                        {
                            if(d1 <= 460)
                            {
                                unique_id = 1830537384143755749ull;
                            }
                            else
                            {
                                if(d5 <= 49)
                                {
                                    unique_id = 1830537384143755749ull;
                                }
                                else
                                {
                                    unique_id = 4992687403741300893ull;
                                }
                            }
                        }
                        else
                        {
                            if(d1 <= 182)
                            {
                                if(d5 <= 65)
                                {
                                    unique_id = 1830537384143755749ull;
                                }
                                else
                                {
                                    unique_id = 4992687403741300893ull;
                                }
                            }
                            else
                            {
                                if(d2 <= 33)
                                {
                                    unique_id = 1830537384143755749ull;
                                }
                                else
                                {
                                    unique_id = 4992687403741300893ull;
                                }
                            }
                        }
                    }
                    else
                    {
                        if(d5 <= 49)
                        {
                            if(d6 <= 64)
                            {
                                if(d1 <= 411)
                                {
                                    if(d2 <= 396)
                                    {
                                        unique_id = 1830537384143755749ull;
                                    }
                                    else
                                    {
                                        unique_id = 4992687403741300893ull;
                                    }
                                }
                                else
                                {
                                    unique_id = 4992687403741300893ull;
                                }
                            }
                            else
                            {
                                unique_id = 4992687403741300893ull;
                            }
                        }
                        else
                        {
                            if(d2 <= 53)
                            {
                                if(d1 <= 222)
                                {
                                    unique_id = 1830537384143755749ull;
                                }
                                else
                                {
                                    unique_id = 4992687403741300893ull;
                                }
                            }
                            else
                            {
                                unique_id = 4992687403741300893ull;
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
