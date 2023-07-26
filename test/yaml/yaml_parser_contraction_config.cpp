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

#include <vector>

#include <llvm/ObjectYAML/YAML.h>

#include <hiptensor/hiptensor.hpp>

#include "yaml_parser_impl.hpp"

// Fwd declare NoneType
namespace hiptensor
{
    struct NoneType;
    static constexpr hipDataType NONE_TYPE = (hipDataType)31;
}

namespace hiptensor
{
    ///
    /// Generalized params for contraction tests
    ///
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

// Treatment of types as vector elements
// Flow sequence vector is inline comma-separated values [val0, val1, ...]
// Sequence vector is line-break separated values
// - val0
// - val1
// ...
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(hipDataType)
LLVM_YAML_IS_SEQUENCE_VECTOR(hiptensorComputeType_t)
LLVM_YAML_IS_SEQUENCE_VECTOR(hiptensorAlgo_t)
LLVM_YAML_IS_SEQUENCE_VECTOR(hiptensorOperator_t)
LLVM_YAML_IS_SEQUENCE_VECTOR(hiptensorWorksizePreference_t)
LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<hipDataType>)
LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<std::size_t>)

namespace llvm
{
    
namespace yaml
{
    ///
    // Enumeration definitions
    ///

    template<>
    struct ScalarEnumerationTraits<hipDataType>
    {
        static void enumeration(IO &io, hipDataType &value) 
        {
            io.enumCase(value, "HIP_R_32F", HIP_R_32F);
            io.enumCase(value, "HIP_R_64F", HIP_R_64F);
            io.enumCase(value, "NONE_TYPE", hiptensor::NONE_TYPE);
        }
    };

    template<>
    struct ScalarEnumerationTraits<hiptensorComputeType_t>
    {
        static void enumeration(IO &io, hiptensorComputeType_t &value) 
        {
            io.enumCase(value, "HIPTENSOR_COMPUTE_32F", HIPTENSOR_COMPUTE_32F);
            io.enumCase(value, "HIPTENSOR_COMPUTE_64F", HIPTENSOR_COMPUTE_64F);
        }
    };
    
    template<>
    struct ScalarEnumerationTraits<hiptensorAlgo_t>
    {
        static void enumeration(IO &io, hiptensorAlgo_t &value) 
        {
            io.enumCase(value, "HIPTENSOR_ALGO_ACTOR_CRITIC", HIPTENSOR_ALGO_ACTOR_CRITIC);
            io.enumCase(value, "HIPTENSOR_ALGO_DEFAULT", HIPTENSOR_ALGO_DEFAULT);
            io.enumCase(value, "HIPTENSOR_ALGO_DEFAULT_PATIENT", HIPTENSOR_ALGO_DEFAULT_PATIENT);
        }
    };

    template<>
    struct ScalarEnumerationTraits<hiptensorOperator_t>
    {
        static void enumeration(IO &io, hiptensorOperator_t &value) 
        {
            io.enumCase(value, "HIPTENSOR_OP_IDENTITY", HIPTENSOR_OP_IDENTITY);
            io.enumCase(value, "HIPTENSOR_OP_UNKNOWN", HIPTENSOR_OP_UNKNOWN);
        }
    };

    template<>
    struct ScalarEnumerationTraits<hiptensorWorksizePreference_t>
    {
        static void enumeration(IO &io, hiptensorWorksizePreference_t &value) 
        {
            io.enumCase(value, "HIPTENSOR_WORKSPACE_MIN", HIPTENSOR_WORKSPACE_MIN);
            io.enumCase(value, "HIPTENSOR_WORKSPACE_RECOMMENDED", HIPTENSOR_WORKSPACE_RECOMMENDED);
            io.enumCase(value, "HIPTENSOR_WORKSPACE_MAX", HIPTENSOR_WORKSPACE_MAX);
        }
    };

    // Treating the hiptensorLogLevel as a bitfield, which is human readable.
    // Comma-separated values are inlined, but combined (|) into a final bit pattern.
    template<>
    struct ScalarBitSetTraits<hiptensorLogLevel_t>
    {
        static void bitset(IO &io, hiptensorLogLevel_t &value) 
        {
            io.bitSetCase(value, "HIPTENSOR_LOG_LEVEL_ERROR", HIPTENSOR_LOG_LEVEL_ERROR);
            io.bitSetCase(value, "HIPTENSOR_LOG_LEVEL_PERF_TRACE", HIPTENSOR_LOG_LEVEL_PERF_TRACE);
            io.bitSetCase(value, "HIPTENSOR_LOG_LEVEL_PERF_HINT", HIPTENSOR_LOG_LEVEL_PERF_HINT);
            io.bitSetCase(value, "HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE", HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE);
            io.bitSetCase(value, "HIPTENSOR_LOG_LEVEL_API_TRACE", HIPTENSOR_LOG_LEVEL_API_TRACE);
        }
    };

    // Finally, mapping of the test param elements for reading / writing.
    template <>
    struct MappingTraits<hiptensor::ContractionTestParams> 
    {
        static void mapping(IO &io, hiptensor::ContractionTestParams &doc) 
        {
            io.mapRequired("Log Level", doc.mLogLevelMask);
            io.mapRequired("Tensor Data Types", doc.mDataTypes);
            io.mapRequired("Compute Types", doc.mComputeTypes);
            io.mapRequired("Algorithm Types", doc.mAlgorithms);
            io.mapRequired("Operators", doc.mOperators);
            io.mapRequired("Worksize Prefs", doc.mWorkSizePrefs);
            io.mapRequired("Alphas", doc.mAlphas);
            io.mapRequired("Lengths", doc.mProblemLengths);

            // Default values for optional values
            auto defaultStrides = std::vector<std::vector<std::size_t>>(doc.mProblemLengths);
            for(auto i = 0; i < defaultStrides.size(); i++)
            {
                defaultStrides[i] = std::vector<std::size_t>(doc.mProblemLengths[i].size(), 0);
            }

            // Optional values
            io.mapOptional("Strides", doc.mProblemStrides, defaultStrides);
            io.mapOptional("Betas", doc.mBetas, std::vector<double>(doc.mAlphas.size(), 0));
                
        }

        // Additional validation for input / output of the config
        static std::string validate(IO &io, hiptensor::ContractionTestParams &doc) 
        {
            if(doc.mProblemLengths.size() == 0)
            {
                return "Error: Empty Lengths";
            }

            if(doc.mAlphas.size() == 0)
            {
                return "Error: Empty Alphas";
            }

            return std::string{};
        }
    };
    
} // namespace yaml

} // namespace llvm

// Instantiate the yaml loader for the ContractionTestParams
namespace hiptensor
{
    template struct YamlConfigLoader<ContractionTestParams>;
}