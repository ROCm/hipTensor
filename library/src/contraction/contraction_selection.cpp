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

namespace hiptensor
{
    hiptensorStatus_t bruteForceModel(ContractionSolution**                    winner,
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
        auto sizeD = elementSpaceFromLengthsAndStrides(d_ms_ns_lengths, d_ms_ns_strides)
                     * hipDataTypeSize(typeD);
        auto sizeE = elementSpaceFromLengthsAndStrides(e_ms_ns_lengths, e_ms_ns_strides)
                     * hipDataTypeSize(typeE);

        void *A_d, *B_d, *D_d, *E_d;
        float alpha = 1.02f;
        float beta  = 1.03f;

        CHECK_HIP_ALLOC(hipMalloc(&A_d, sizeA));
        CHECK_HIP_ALLOC(hipMalloc(&B_d, sizeB));
        CHECK_HIP_ALLOC(hipMalloc(&D_d, sizeD));
        CHECK_HIP_ALLOC(hipMalloc(&E_d, sizeE));

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
                                  std::vector<std::vector<ck::index_t>>{d_ms_ns_lengths},
                                  std::vector<std::vector<ck::index_t>>{d_ms_ns_strides},
                                  e_ms_ns_lengths,
                                  e_ms_ns_strides))
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

#if !NDEBUG
        std::cout << bestMetrics << std::endl;
#endif // !NDEBUG

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

    hiptensorStatus_t actorCriticModel(ContractionSolution**                    winner,
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
    }
}
