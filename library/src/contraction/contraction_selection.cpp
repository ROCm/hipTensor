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
                                      std::vector<int32_t> const&              a_ms_ks_modes,
                                      hipDataType                              typeB,
                                      std::vector<std::size_t> const&          b_ns_ks_lengths,
                                      std::vector<std::size_t> const&          b_ns_ks_strides,
                                      std::vector<int32_t> const&              b_ns_ks_modes,
                                      hipDataType                              typeD,
                                      std::vector<std::size_t> const&          d_ms_ns_lengths,
                                      std::vector<std::size_t> const&          d_ms_ns_strides,
                                      std::vector<int32_t> const&              d_ms_ns_modes,
                                      hipDataType                              typeE,
                                      std::vector<std::size_t> const&          e_ms_ns_lengths,
                                      std::vector<std::size_t> const&          e_ms_ns_strides,
                                      std::vector<int32_t> const&              e_ms_ns_modes,
                                      hiptensorComputeType_t                   computeType,
                                      const uint64_t                           workspaceSize)
    {
        // Make sure that we calculate full element space incase strides are not packed.
        auto sizeA = elementsFromLengths(a_ms_ks_lengths) * hipDataTypeSize(typeA);
        auto sizeB = elementsFromLengths(b_ns_ks_lengths) * hipDataTypeSize(typeB);
        auto sizeD = 0;
        if(typeD != NONE_TYPE)
        {
            sizeD = elementsFromLengths(d_ms_ns_lengths) * hipDataTypeSize(typeD);
        }
        auto sizeE = elementsFromLengths(e_ms_ns_lengths) * hipDataTypeSize(typeE);

        void *A_d, *B_d, *D_d, *E_d, *wspace;

        /*
         * `alpha` and `beta` are void pointer. hiptensor uses readVal to load the value of alpha.
         * ```
         * alphaF = hiptensor::readVal<float>(
         *      alpha, convertToComputeType(HipDataType_v<typename Traits::ComputeDataT>));
         * ```
         * Hence, the `alpha` and `bete` need to point to a ComputeData value
         */
        ScalarData alpha;
        ScalarData beta;
        if(computeType == HIPTENSOR_COMPUTE_C32F || computeType == HIPTENSOR_COMPUTE_C64F)
        {
            writeVal(&alpha, computeType, {computeType, 1.02, 1.03});
            writeVal(&beta, computeType, {computeType, 1.04, 1.05});
        }
        else
        {
            writeVal(&alpha, computeType, ScalarData(computeType, 1.02));
            writeVal(&beta, computeType, ScalarData(computeType, 1.03));
        }

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
            auto [errorCode, time] = (*solution)(&alpha,
                                                 A_d,
                                                 B_d,
                                                 &beta,
                                                 D_d,
                                                 E_d,
                                                 a_ms_ks_lengths,
                                                 a_ms_ks_strides,
                                                 a_ms_ks_modes,
                                                 b_ns_ks_lengths,
                                                 b_ns_ks_strides,
                                                 b_ns_ks_modes,
                                                 d_ms_ns_lengths,
                                                 d_ms_ns_strides,
                                                 d_ms_ns_modes,
                                                 e_ms_ns_lengths,
                                                 e_ms_ns_strides,
                                                 e_ms_ns_modes,
                                                 wspace,
                                                 workspaceSize,
                                                 StreamConfig{nullptr, true});
            if(errorCode == HIPTENSOR_STATUS_SUCCESS && time > 0)
            {
                // Make sure to time the kernels
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
    struct ActorCriticSelection<_Float16,
                                _Float16,
                                _Float16,
                                _Float16,
                                ContractionOpId_t::SCALE,
                                float>
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

            unique_id = 11124293857315312720ull;

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
    struct ActorCriticSelection<_Float16,
                                _Float16,
                                _Float16,
                                _Float16,
                                ContractionOpId_t::BILINEAR,
                                float>
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

            unique_id = 1953020431947874122ull;

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
    struct ActorCriticSelection<hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                ContractionOpId_t::SCALE,
                                float>
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

            unique_id = 14895098881714635802ull;

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
    struct ActorCriticSelection<hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                hip_bfloat16,
                                ContractionOpId_t::BILINEAR,
                                float>
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

            unique_id = 8517235228581081946ull;

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
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, _Float16>
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

            unique_id = 17313709378682913599ull;

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
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::BILINEAR, _Float16>
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

            unique_id = 14397647188602189900ull;

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
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, hip_bfloat16>
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

            unique_id = 8339198051871565944ull;

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
    struct ActorCriticSelection<float,
                                float,
                                float,
                                float,
                                ContractionOpId_t::BILINEAR,
                                hip_bfloat16>
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

            unique_id = 2724417728984064737ull;

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
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, float>
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

            unique_id = 5943247903036531691ull;

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
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::BILINEAR, float>
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

            unique_id = 17972447156160297755ull;

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
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::SCALE, float>
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

            unique_id = 3893144338697524749ull;

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
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::BILINEAR, float>
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
            unique_id        = 15165261158317928321ull;

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
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::SCALE, double>
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

            unique_id = 14511729289005214097ull;

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
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::BILINEAR, double>
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

            unique_id = 3636246152928348445ull;

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
    struct ActorCriticSelection<hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                ContractionOpId_t::SCALE_COMPLEX,
                                hipFloatComplex>
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

            unique_id = 5711776907278244209ull;

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
    struct ActorCriticSelection<hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                hipFloatComplex,
                                ContractionOpId_t::BILINEAR_COMPLEX,
                                hipFloatComplex>
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

            unique_id = 355777364055884033ull;

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
    struct ActorCriticSelection<hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                ContractionOpId_t::SCALE_COMPLEX,
                                hipDoubleComplex>
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

            unique_id = 3085227716611397774ull;

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
    struct ActorCriticSelection<hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                hipDoubleComplex,
                                ContractionOpId_t::BILINEAR_COMPLEX,
                                hipDoubleComplex>
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

            unique_id = 2196983681630807584ull;

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
                         hiptensorComputeType_t                                  computeType,
                         const uint64_t                                          workspaceSize)
    {
        if(typeA == HIP_R_16F && typeB == HIP_R_16F && typeD == NONE_TYPE && typeE == HIP_R_16F
           && computeType == HIPTENSOR_COMPUTE_32F)
        {
            return ActorCriticSelection<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(winner,
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
        else if(typeA == HIP_R_16F && typeB == HIP_R_16F && typeD == HIP_R_16F && typeE == HIP_R_16F
                && computeType == HIPTENSOR_COMPUTE_32F)
        {
            return ActorCriticSelection<_Float16,
                                        _Float16,
                                        _Float16,
                                        _Float16,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(winner,
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
        else if(typeA == HIP_R_16BF && typeB == HIP_R_16BF && typeD == NONE_TYPE
                && typeE == HIP_R_16BF && computeType == HIPTENSOR_COMPUTE_32F)
        {
            return ActorCriticSelection<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(winner,
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
        else if(typeA == HIP_R_16BF && typeB == HIP_R_16BF && typeD == HIP_R_16BF
                && typeE == HIP_R_16BF && computeType == HIPTENSOR_COMPUTE_32F)
        {
            return ActorCriticSelection<hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        hip_bfloat16,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(winner,
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
        else if(typeA == HIP_R_32F && typeB == HIP_R_32F && typeD == NONE_TYPE && typeE == HIP_R_32F
                && computeType == HIPTENSOR_COMPUTE_16F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        _Float16>::selectWinner(winner,
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
        else if(typeA == HIP_R_32F && typeB == HIP_R_32F && typeD == HIP_R_32F && typeE == HIP_R_32F
                && computeType == HIPTENSOR_COMPUTE_16F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        _Float16>::selectWinner(winner,
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
        else if(typeA == HIP_R_32F && typeB == HIP_R_32F && typeD == NONE_TYPE && typeE == HIP_R_32F
                && computeType == HIP_R_16BF)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        hip_bfloat16>::selectWinner(winner,
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
        else if(typeA == HIP_R_32F && typeB == HIP_R_32F && typeD == HIP_R_32F && typeE == HIP_R_32F
                && computeType == HIP_R_16BF)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        hip_bfloat16>::selectWinner(winner,
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
        else if(typeA == HIP_R_32F && typeB == HIP_R_32F && typeD == NONE_TYPE && typeE == HIP_R_32F
                && computeType == HIPTENSOR_COMPUTE_32F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(winner,
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
        else if(typeA == HIP_R_32F && typeB == HIP_R_32F && typeD == HIP_R_32F && typeE == HIP_R_32F
                && computeType == HIPTENSOR_COMPUTE_32F)
        {
            return ActorCriticSelection<float,
                                        float,
                                        float,
                                        float,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(winner,
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
        else if(typeA == HIP_R_64F && typeB == HIP_R_64F && typeD == NONE_TYPE && typeE == HIP_R_64F
                && computeType == HIPTENSOR_COMPUTE_32F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::SCALE,
                                        float>::selectWinner(winner,
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
        else if(typeA == HIP_R_64F && typeB == HIP_R_64F && typeD == HIP_R_64F && typeE == HIP_R_64F
                && computeType == HIPTENSOR_COMPUTE_32F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::BILINEAR,
                                        float>::selectWinner(winner,
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
        else if(typeA == HIP_R_64F && typeB == HIP_R_64F && typeD == NONE_TYPE && typeE == HIP_R_64F
                && computeType == HIPTENSOR_COMPUTE_64F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::SCALE,
                                        double>::selectWinner(winner,
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
        else if(typeA == HIP_R_64F && typeB == HIP_R_64F && typeD == HIP_R_64F && typeE == HIP_R_64F
                && computeType == HIPTENSOR_COMPUTE_64F)
        {
            return ActorCriticSelection<double,
                                        double,
                                        double,
                                        double,
                                        ContractionOpId_t::BILINEAR,
                                        double>::selectWinner(winner,
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
        else if(typeA == HIP_C_32F && typeB == HIP_C_32F && typeD == NONE_TYPE && typeE == HIP_C_32F
                && computeType == HIPTENSOR_COMPUTE_C32F)
        {
            return ActorCriticSelection<hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        ContractionOpId_t::SCALE_COMPLEX,
                                        hipFloatComplex>::selectWinner(winner,
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
        else if(typeA == HIP_C_32F && typeB == HIP_C_32F && typeD == HIP_C_32F && typeE == HIP_C_32F
                && computeType == HIPTENSOR_COMPUTE_C32F)
        {
            return ActorCriticSelection<hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        ContractionOpId_t::BILINEAR_COMPLEX,
                                        hipFloatComplex>::selectWinner(winner,
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
        else if(typeA == HIP_C_64F && typeB == HIP_C_64F && typeD == NONE_TYPE && typeE == HIP_C_64F
                && computeType == HIPTENSOR_COMPUTE_C64F)
        {
            return ActorCriticSelection<hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        ContractionOpId_t::SCALE_COMPLEX,
                                        hipDoubleComplex>::selectWinner(winner,
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
        else if(typeA == HIP_C_64F && typeB == HIP_C_64F && typeD == HIP_C_64F && typeE == HIP_C_64F
                && computeType == HIPTENSOR_COMPUTE_C64F)
        {
            return ActorCriticSelection<hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        ContractionOpId_t::BILINEAR_COMPLEX,
                                        hipDoubleComplex>::selectWinner(winner,
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
