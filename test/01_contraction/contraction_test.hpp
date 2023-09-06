/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021-2023 Advanced Micro Devices, Inc.
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

#ifndef HIPTENSOR_CONTRACTION_TEST_HPP
#define HIPTENSOR_CONTRACTION_TEST_HPP

#include "../library/src/include/types.hpp"
#include <hiptensor/hiptensor.hpp>

#include "common.hpp"
#include "contraction_cpu_reference.hpp"
#include "contraction_resource.hpp"
#include "contraction_test_params.hpp"

#include "llvm/hiptensor_options.hpp"
#include "llvm/yaml_parser.hpp"

#ifdef HIPTENSOR_TEST_YAML_INCLUDE
#include HIPTENSOR_TEST_YAML_INCLUDE
#define HIPTENSOR_TEST_YAML_BUNDLE 1
#else
#define HIPTENSOR_TEST_YAML_BUNDLE 0
#endif // HIPTENSOR_TEST_YAML_INCLUDE

#include <gtest/gtest.h>

namespace hiptensor
{
    static void logMessage(int32_t logLevel, const char* funcName = "", const char* msg = "");

    class ContractionTest : public ::testing::TestWithParam<
                                std::tuple<typename ContractionTestParams::TestDataTypeT,
                                           typename ContractionTestParams::TestComputeTypeT,
                                           typename ContractionTestParams::AlgorithmT,
                                           typename ContractionTestParams::OperatorT,
                                           typename ContractionTestParams::WorkSizePrefT,
                                           typename ContractionTestParams::LogLevelT,
                                           typename ContractionTestParams::LengthsT,
                                           typename ContractionTestParams::StridesT,
                                           typename ContractionTestParams::AlphaT,
                                           typename ContractionTestParams::BetaT>>
    {
    protected: // Types
        using Base
            = ::testing::TestWithParam<std::tuple<typename ContractionTestParams::TestDataTypeT,
                                                  typename ContractionTestParams::TestComputeTypeT,
                                                  typename ContractionTestParams::AlgorithmT,
                                                  typename ContractionTestParams::OperatorT,
                                                  typename ContractionTestParams::WorkSizePrefT,
                                                  typename ContractionTestParams::LogLevelT,
                                                  typename ContractionTestParams::LengthsT,
                                                  typename ContractionTestParams::StridesT,
                                                  typename ContractionTestParams::AlphaT,
                                                  typename ContractionTestParams::BetaT>>;

        // Shared access to Contraction storage
        using DataStorage = ContractionResource;

        friend void logMessage(int32_t, const char*, const char*);

    public:
        ContractionTest();
        virtual ~ContractionTest() = default;

    protected: // Functions
        ContractionTest(ContractionTest&&)            = delete;
        ContractionTest(ContractionTest const&)       = delete;
        ContractionTest& operator=(ContractionTest&)  = delete;
        ContractionTest& operator=(ContractionTest&&) = delete;

        bool checkDevice(hipDataType datatype) const;
        bool checkSizes() const;
        void reset();

        ContractionResource* getResource() const;

        void SetUp() final;
        void TearDown() final;

        void Warmup() {}
        void RunKernel();

        void reportResults(std::ostream& stream,
                           hipDataType   DDataType,
                           bool          omitSkipped,
                           bool          omitFailed,
                           bool          omitPassed) const;

    protected:
        // Workspace items
        hiptensorHandle_t*               handle = nullptr;
        hiptensorContractionPlan_t       plan;
        hiptensorContractionDescriptor_t desc;
        hiptensorContractionFind_t       find;
        uint64_t                         worksize;
        void*                            workspace = nullptr;

        hiptensorTensorDescriptor_t a_ms_ks, b_ns_ks, c_ms_ns, d_ms_ns;

        // Execution flow control
        uint32_t mRepeats;
        bool     mRunFlag          = true;
        bool     mValidationResult = false;
        bool     mPrintElements    = false;
        double   mMaxRelativeError;

        // Output buffer
        static std::stringstream sOutputBuff;
    };

} // namespace hiptensor

auto inline load_config_helper()
{
    hiptensor::ContractionTestParams testParams;
    using Options     = hiptensor::HiptensorOptions;
    auto& testOptions = Options::instance();

    if(testOptions->usingDefaultConfig() && HIPTENSOR_TEST_YAML_BUNDLE)
    {
        testParams = hiptensor::YamlConfigLoader<hiptensor::ContractionTestParams>::loadFromString(
            HIPTENSOR_TEST_GET_YAML);
    }
    else
    {
        testParams = hiptensor::YamlConfigLoader<hiptensor::ContractionTestParams>::loadFromFile(
            testOptions->inputFilename());
    }

    // testParams.printParams();

    return ::testing::Combine(::testing::ValuesIn(testParams.dataTypes()),
                              ::testing::ValuesIn(testParams.computeTypes()),
                              ::testing::ValuesIn(testParams.algorithms()),
                              ::testing::ValuesIn(testParams.operators()),
                              ::testing::ValuesIn(testParams.workSizePrefrences()),
                              ::testing::Values(testParams.logLevelMask()),
                              ::testing::ValuesIn(testParams.problemLengths()),
                              ::testing::ValuesIn(testParams.problemStrides()),
                              ::testing::ValuesIn(testParams.alphas()),
                              ::testing::ValuesIn(testParams.betas()));
}

#endif // HIPTENSOR_CONTRACTION_TEST_HPP
