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

#include <hiptensor/hiptensor.hpp>
#include <hiptensor/hiptensor_types.hpp>
// #include "contraction_kernel_base.hpp"

namespace hiptensor
{
    ///
    /// Generalized params for contraction tests
    ///
    struct ContractionCommonTestParams
    {
        using TestDataTypeT    = std::vector<hipDataType>;
        using TestComputeTypeT = hiptensorComputeType_t;

        using AlgorithmT    = hiptensorAlgo_t;
        using OperatorT     = hiptensorOperator_t;
        using WorkSizePrefT = hiptensorWorksizePreference_t;
        using LogLevelT     = hiptensorLogLevel_t;

        using LengthsT = std::vector<std::size_t>;
        using StridesT = std::vector<std::size_t>;
        using AlphaT   = double;
        using BetaT    = double;

        //TODO: Include after kernel generator/ kernel base class is created
        // using KernelT      = std::shared_ptr<KernelI>; // Kernel test interface

        //Data types of input and output tensors
        static inline std::vector<TestDataTypeT> dataTypes()
        {
            return
            {
                // clang-format off
                {HIP_R_32F, HIP_R_32F, NONE_TYPE, HIP_R_32F}, // scale F32
                {HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F}, // bilinear F32
                {HIP_R_64F, HIP_R_64F, NONE_TYPE, HIP_R_64F}, // scale F64
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
                //'m','n','u','v','h','k'
                {150, 131, 14, 540, 17, 10},
                {382, 33, 141, 58, 47, 29},
                {19, 526, 40, 81, 7, 33},
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
