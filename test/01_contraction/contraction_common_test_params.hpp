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

#ifndef HIPTENSOR_CONTRACTION_COMMON_TEST_PARAMS_HPP
#define HIPTENSOR_CONTRACTION_COMMON_TEST_PARAMS_HPP

#include <tuple>
#include <vector>

#include <hiptensor/hiptensor_types.hpp>

namespace hiptensor
{
    ///
    /// Generalized params for contraction tests
    ///
    struct ContractionCommonTestParams
    {
        
        using TestDataTypeT = std::vector<hipDataType>;
        using TestComputeTypeT = hiptensorComputeType_t;
 
        using AlgorithmT = hiptensorAlgo_t;
        using OperatorT = hiptensorOperator_t;
        using WorkSizePrefT = hiptensorWorksizePreference_t;
        using LogLevelT = hiptensorLogLevel_t;

        using LengthsT     = std::vector<std::size_t>;
        using StridesT     = std::vector<std::size_t>;
        using AlphaT       = double;
        using BetaT        = double;

        //Data types of input and output tensors
        static inline std::vector<TestDataTypeT> dataTypes()
        {
            return
            {
                // clang-format off
                {HIP_R_32F, HIP_R_32F, HIP_R_32F}, // scale F32
                {HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F}, // bilinear F32
                {HIP_R_64F, HIP_R_64F, HIP_R_64F}, // scale F64
                {HIP_R_64F, HIP_R_64F, HIP_R_64F, HIP_R_64F}, // bilinear F64
                // clang-format on
            };
        }

        // compute type of contraction operation
        static inline std::vector<TestComputeTypeT> computeTypes()
        {
            return
            {
                // clang-format off
                HIPTENSOR_COMPUTE_32F,
                HIPTENSOR_COMPUTE_64F,
                // clang-format on
            };
        }

        static inline std::vector<AlgorithmT> algorithms()
        {
            return
            {
                // clang-format off
                HIPTENSOR_ALGO_ACTOR_CRITIC,
                HIPTENSOR_ALGO_DEFAULT,
                HIPTENSOR_ALGO_DEFAULT_PATIENT,
                // clang-format on
            };
        }

        static inline std::vector<OperatorT> operators()
        {
            return
            {
                // clang-format off
                HIPTENSOR_OP_IDENTITY,
                // clang-format on
            };
        }

        static inline std::vector<WorkSizePrefT> workSizePrefrences()
        {
            return
            {
                // clang-format off
                HIPTENSOR_WORKSPACE_MIN,
                HIPTENSOR_WORKSPACE_RECOMMENDED,
                HIPTENSOR_WORKSPACE_MAX,
                // clang-format on
            };
        }

        static inline std::vector<LogLevelT> logLevels()
        {
            return
            {
                // clang-format off
                HIPTENSOR_LOG_LEVEL_OFF,
                HIPTENSOR_LOG_LEVEL_ERROR,
                HIPTENSOR_LOG_LEVEL_PERF_TRACE,
                HIPTENSOR_LOG_LEVEL_PERF_HINT,
                HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE,
                HIPTENSOR_LOG_LEVEL_API_TRACE,
                // clang-format on
            };
        }

        static inline std::vector<LengthsT> problemLengths()
        {
            return
            {
                // clang-format off
                {150, 131, 14, 540, 17, 10},
                {382, 33, 141, 58, 47, 29},
                {19, 526, 40, 81, 7, 33},
                {216, 61, 119, 165, 31, 24},
                {72, 145, 59, 227, 6, 53},
                {45, 52, 182, 13, 20, 5},
                {31, 164, 22, 893, 78, 25},
                {673, 18, 35, 128, 66, 7},
                {23, 312, 312, 48, 39, 60},
                {44, 341, 31, 312, 8, 12},
                {308, 30, 18, 352, 17, 43},
                {532, 31, 353, 34, 61, 70},
                {128, 149, 400, 23, 33, 24},
                {155, 45, 102, 147, 41, 37},
                // clang-format on
            };
        }

        static inline std::vector<StridesT> problemStrides()
        {
            return
            {
                // clang-format off
                // clang-format on
            };
        }

        static inline std::vector<AlphaT> alphas()
        {
            return {static_cast<AlphaT>(2)};
        }

        static inline std::vector<BetaT> betas()
        {
            return {static_cast<BetaT>(2)};
        }
    };

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_COMMON_TEST_PARAMS_HPP
