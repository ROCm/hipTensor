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

#include "common.hpp"
#include "contraction_test_params.hpp"
#include "llvm/hiptensor_options.hpp"
#include "llvm/yaml_parser.hpp"

#include <gtest/gtest.h>

namespace hiptensor
{
    struct ContractionTest : public ::testing::TestWithParam<
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

        void SetUp() override
        {
            auto param        = Base::GetParam();
            auto testType     = std::get<0>(param);
            auto computeType  = std::get<1>(param);
            auto algorithm    = std::get<2>(param);
            auto operatorType = std::get<3>(param);
            auto workSizePref = std::get<4>(param);
            auto logLevel     = std::get<5>(param);
            auto lengths      = std::get<6>(param);
            auto strides      = std::get<7>(param);
            auto alpha        = std::get<8>(param);
            auto beta         = std::get<9>(param);

            std::cout << testType << ", " << computeType << ", " << algorithm << ", "
                      << operatorType << ", " << workSizePref << ", " << logLevel << ", " << lengths
                      << ", " << strides << ", " << alpha << ", " << beta << "\n";
        }

        virtual void RunKernel()
        {
            auto param        = Base::GetParam();
            auto testType     = std::get<0>(param);
            auto computeType  = std::get<1>(param);
            auto algorithm    = std::get<2>(param);
            auto operatorType = std::get<3>(param);
            auto workSizePref = std::get<4>(param);
            auto logLevel     = std::get<5>(param);
            auto lengths      = std::get<6>(param);
            auto strides      = std::get<7>(param);
            auto alpha        = std::get<8>(param);
            auto beta         = std::get<9>(param);

            EXPECT_TRUE(beta > 0.0);
        }

        virtual void Warmup() {}

        void TearDown() override {}
    };

} // namespace rocwmma

#endif // HIPTENSOR_CONTRACTION_TEST_HPP
