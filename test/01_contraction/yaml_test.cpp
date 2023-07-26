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

#include <algorithm>
#include <fstream>
#include <iterator>
#include <numeric>
#include <unordered_map>

// hiptensor includes
#include <hiptensor/hiptensor.hpp>
#include <hiptensor/hiptensor_types.hpp>
#include <hiptensor/internal/hiptensor_utility.hpp>

#include "common.hpp"
#include "../yaml/yaml_parser.hpp"

namespace hiptensor
{
    struct NoneType;
    static constexpr hipDataType NONE_TYPE = (hipDataType)31;
    struct ContractionTestParams
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
        std::vector<TestDataTypeT> mDataTypes;
        std::vector<TestComputeTypeT> mComputeTypes;
        std::vector<AlgorithmT> mAlgorithms;
        std::vector<OperatorT> mOperators;
        std::vector<WorkSizePrefT> mWorkSizePrefs;
        LogLevelT mLogLevelMask;
        std::vector<LengthsT> mProblemLengths;
        std::vector<StridesT> mProblemStrides;
        std::vector<AlphaT> mAlphas;
        std::vector<BetaT> mBetas;
    };
}


#define MAX_ELEMENTS_PRINT_COUNT 512

int main(int argc, char* argv[])
{
    auto yee = hiptensor::ContractionTestParams{};
    yee.mLogLevelMask = (hiptensorLogLevel_t)(HIPTENSOR_LOG_LEVEL_OFF);
    yee.mDataTypes = {
                // clang-format off
                {HIP_R_32F, HIP_R_32F, hiptensor::NONE_TYPE, HIP_R_32F}, // scale F32
                {HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F}, // bilinear F32
                {HIP_R_64F, HIP_R_64F, hiptensor::NONE_TYPE, HIP_R_64F}, // scale F64
                {HIP_R_64F, HIP_R_64F, HIP_R_64F, HIP_R_64F}, // bilinear F64
                // clang-format on
            };

    yee.mComputeTypes = { HIPTENSOR_COMPUTE_32F, HIPTENSOR_COMPUTE_64F };
    yee.mAlgorithms = { HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_ALGO_DEFAULT_PATIENT, HIPTENSOR_ALGO_ACTOR_CRITIC};
    yee.mOperators = { HIPTENSOR_OP_IDENTITY };
    yee.mWorkSizePrefs = { HIPTENSOR_WORKSPACE_RECOMMENDED, HIPTENSOR_WORKSPACE_MIN, HIPTENSOR_WORKSPACE_MAX};
    yee.mLogLevelMask = { hiptensorLogLevel_t(HIPTENSOR_LOG_LEVEL_ERROR | HIPTENSOR_LOG_LEVEL_PERF_TRACE)};
    yee.mProblemLengths = {
        {5, 6, 7, 8, 4, 2, 3, 4},
        {1, 2, 3, 4},
        {99, 12, 44, 31, 59, 23, 54, 22}
    };
    yee.mAlphas = {0, 1, 1};
    yee.mBetas = {2, 2, 2};

    hiptensor::YamlConfigLoader<hiptensor::ContractionTestParams>::storeToFile("/home/cmillett/hipTensor-cgmillette/test-out.txt", yee);
    auto yee1 = hiptensor::YamlConfigLoader<hiptensor::ContractionTestParams>::loadFromFile("/home/cmillett/hipTensor-cgmillette/test-out.txt");
 
   
    return 0;
}
