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
#include "logger.hpp"

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

		using hiptensor::Logger;
                auto& logger = Logger::instance();

                // Log Kernel performances access
                char msg[256];
                snprintf(msg,
                        sizeof(msg),
                        "KernelId: %lu, KernelName: %s, AvgTime: %0.3f ms",
                        solution->uid(),
                        solution->kernelName().c_str(),
                        time);

                logger->logPerformanceTrace("BRUTE_FORCE_KERNEL_PERF", msg);

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

    bool is1D(std::vector<std::size_t> const& a_ms_ks_lengths,
                std::vector<std::size_t> const& a_ms_ks_strides,
                std::vector<std::size_t> const& b_ns_ks_lengths)
    {
        bool dim1 = false;
        int dimension = a_ms_ks_lengths.size() / 2;
        
        for (int i = 0; i < dimension; i++)
        {
            if (a_ms_ks_lengths[i] == 1 || 
                b_ns_ks_lengths[i] == 1 || 
                a_ms_ks_lengths[dimension + i] == 1)
            {
                dim1 = true;
            }
        }

        return dim1;
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
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;
            
            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 5113293080724535756ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 17912428805389269017ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 13987975603645579562ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 13987975603645579562ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 13987975603645579562ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 13987975603645579562ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 17912428805389269017ull;
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
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 58303249112943560ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            // if (d1 == 1 || (d2 == 1 && (a_ms_ks_lengths[3] == 1 || b_ns_ks_lengths[3] == 1)))
            {
                unique_id = 58303249112943560ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 2303552229010777601ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 58303249112943560ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 58303249112943560ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 58303249112943560ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 2303552229010777601ull;
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
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 14015334081946837014ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 1015685992483452995ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 1015685992483452995ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 1015685992483452995ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 1015685992483452995ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 1015685992483452995ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 1015685992483452995ull;
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
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 16299024124514902126ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 378062791888302715ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 76527422265261696ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 378062791888302715ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 378062791888302715ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 378062791888302715ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 378062791888302715ull;
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
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, _Float16>
    {
        static hiptensorStatus_t
           selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 4548033111668367400ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 10800739437646028422ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 717325201541913345ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 10800739437646028422ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 10800739437646028422ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 3781627802884041058ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 10800739437646028422ull;
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
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::BILINEAR, _Float16>
    {
        static hiptensorStatus_t
           selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 8251132190088736039ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 2897979232477761524ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 2897979232477761524ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 2897979232477761524ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 2897979232477761524ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 2897979232477761524ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 2897979232477761524ull;
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
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, hip_bfloat16>
    {
        static hiptensorStatus_t
           selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 16420628376898423538ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 17959510091945286251ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 17959510091945286251ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 17959510091945286251ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 17959510091945286251ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 17959510091945286251ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 17959510091945286251ull;
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
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 8067958629699904967ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 8116863550692548667ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 8116863550692548667ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 8116863550692548667ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 8116863550692548667ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 8116863550692548667ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 8116863550692548667ull;
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
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::SCALE, float>
    {
        static hiptensorStatus_t
           selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 8826603567277321994ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 18096417737618973195ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 16893702863645273417ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 16893702863645273417ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 16893702863645273417ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 16893702863645273417ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 16893702863645273417ull;
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
    struct ActorCriticSelection<float, float, float, float, ContractionOpId_t::BILINEAR, float>
    {
        static hiptensorStatus_t
           selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 14915761978535949477ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 14915761978535949477ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 14915761978535949477ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 14915761978535949477ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 14915761978535949477ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 14915761978535949477ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 14915761978535949477ull;
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
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::SCALE, float>
    {
        static hiptensorStatus_t
           selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {

            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 14901158961446820896ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 8188562791036959263ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 8188562791036959263ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 7606237861132087768ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 7606237861132087768ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 8188562791036959263ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 7606237861132087768ull;
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
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::BILINEAR, float>
    {
        static hiptensorStatus_t
           selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 11269655469469274301ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 2143493311543532856ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 2143493311543532856ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 2143493311543532856ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 2143493311543532856ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 2143493311543532856ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 2143493311543532856ull;
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
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::SCALE, double>
    {
        static hiptensorStatus_t
           selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {

            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 3879892272436099392ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 3879892272436099392ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 3879892272436099392ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 3879892272436099392ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 3879892272436099392ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 3879892272436099392ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 3879892272436099392ull;
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
    struct ActorCriticSelection<double, double, double, double, ContractionOpId_t::BILINEAR, double>
    {
        static hiptensorStatus_t
           selectWinner(ContractionSolution**                                   winner,
                         std::unordered_map<size_t, ContractionSolution*> const& candidates,
                         hipDataType                                             typeA,
                         std::vector<std::size_t> const&                         a_ms_ks_lengths,
                         std::vector<std::size_t> const&                         a_ms_ks_strides,
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 2054609181761357786ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 14145390177844245465ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 14145390177844245465ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 14145390177844245465ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 14145390177844245465ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 14145390177844245465ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 14145390177844245465ull;
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
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {

            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 7812623739736355326ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 6132850456576499774ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 3843982104146107466ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 6132850456576499774ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 16674999420449441492ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 3843982104146107466ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 16674999420449441492ull;
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
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 15330878641001915472ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 11537900932066889768ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 8338926107119209426ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 11537900932066889768ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 11537900932066889768ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 11537900932066889768ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 11537900932066889768ull;
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
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {

            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 10254320286859648634ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 12959721676360111684ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 12959721676360111684ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 12959721676360111684ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 12959721676360111684ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 12959721676360111684ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 12959721676360111684ull;
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
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
                         const uint64_t                                          workspaceSize)
        {
            int d1 = a_ms_ks_strides[1];
            int d2 = a_ms_ks_strides[3];
            int d3 = a_ms_ks_strides[5];
            int d4 = a_ms_ks_strides[7];
            int d5 = a_ms_ks_strides[9];
            int d6 = a_ms_ks_strides[11];

            size_t unique_id = 0;

            bool dim1 = is1D(a_ms_ks_lengths, a_ms_ks_strides, b_ns_ks_lengths);

            // rank2 dim1 case
            if (d2 == 1 && dim1)
            {
                unique_id = 14051358583041094215ull;
            }
            // m1n1k1
            else if (d1 == 1) 
            {
                unique_id = 8503926755447648324ull;
            }
            // m2n2k2
            else if (d2 == 1)
            {
                unique_id = 8503926755447648324ull;
            }
            // m3n3k3
            else if (d3 == 1)
            {
                unique_id = 8503926755447648324ull;
            }
            // m4n4k4
            else if (d4 == 1)
            {
                unique_id = 8503926755447648324ull;
            }
            // m5n5k5
            else if (d5 == 1)
            {
                unique_id = 8503926755447648324ull;
            }
            // m6n6k6
            else if (d6 == 1)
            {
                unique_id = 8503926755447648324ull;
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
                         std::vector<int32_t> const&                             a_ms_ks_modes,
                         hipDataType                                             typeB,
                         std::vector<std::size_t> const&                         b_ns_ks_lengths,
                         std::vector<std::size_t> const&                         b_ns_ks_strides,
                         std::vector<int32_t> const&                             b_ns_ks_modes,
                         hipDataType                                             typeD,
                         std::vector<std::size_t> const&                         d_ms_ns_lengths,
                         std::vector<std::size_t> const&                         d_ms_ns_strides,
                         std::vector<int32_t> const&                             d_ms_ns_modes,
                         hipDataType                                             typeE,
                         std::vector<std::size_t> const&                         e_ms_ns_lengths,
                         std::vector<std::size_t> const&                         e_ms_ns_strides,
                         std::vector<int32_t> const&                             e_ms_ns_modes,
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
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
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
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
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
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
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
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
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
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
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
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
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
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
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
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
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
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
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
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
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
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
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
                                                             a_ms_ks_modes,
                                                             typeB,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             b_ns_ks_modes,
                                                             typeD,
                                                             d_ms_ns_lengths,
                                                             d_ms_ns_strides,
                                                             d_ms_ns_modes,
                                                             typeE,
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides,
                                                             e_ms_ns_modes,
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
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
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
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
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
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
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
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
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
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
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
                                                                a_ms_ks_modes,
                                                                typeB,
                                                                b_ns_ks_lengths,
                                                                b_ns_ks_strides,
                                                                b_ns_ks_modes,
                                                                typeD,
                                                                d_ms_ns_lengths,
                                                                d_ms_ns_strides,
                                                                d_ms_ns_modes,
                                                                typeE,
                                                                e_ms_ns_lengths,
                                                                e_ms_ns_strides,
                                                                e_ms_ns_modes,
                                                                workspaceSize);
        }
        return HIPTENSOR_STATUS_EXECUTION_FAILED;
    }
}
