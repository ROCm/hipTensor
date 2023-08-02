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

#include "contraction_common_test_params.hpp"

#include <gtest/gtest.h>

namespace hiptensor
{
    struct ContractionTest : public ::testing::TestWithParam<
                                 std::tuple<typename ContractionCommonTestParams::TestDataTypeT,
                                            typename ContractionCommonTestParams::TestComputeTypeT,
                                            typename ContractionCommonTestParams::AlgorithmT,
                                            typename ContractionCommonTestParams::OperatorT,
                                            typename ContractionCommonTestParams::WorkSizePrefT,
                                            typename ContractionCommonTestParams::LogLevelT,
                                            typename ContractionCommonTestParams::LengthsT,
                                            typename ContractionCommonTestParams::StridesT,
                                            typename ContractionCommonTestParams::AlphaT,
                                            typename ContractionCommonTestParams::BetaT>>
    {
        using Base = ::testing::TestWithParam<
            std::tuple<typename ContractionCommonTestParams::TestDataTypeT,
                       typename ContractionCommonTestParams::TestComputeTypeT,
                       typename ContractionCommonTestParams::AlgorithmT,
                       typename ContractionCommonTestParams::OperatorT,
                       typename ContractionCommonTestParams::WorkSizePrefT,
                       typename ContractionCommonTestParams::LogLevelT,
                       typename ContractionCommonTestParams::LengthsT,
                       typename ContractionCommonTestParams::StridesT,
                       typename ContractionCommonTestParams::AlphaT,
                       typename ContractionCommonTestParams::BetaT>>;

        void SetUp() override {}

        virtual void RunKernel() {}

        virtual void Warmup() {}

        void TearDown() override {}
    };
    // pass enum template values through Base::<name>

} // namespace rocwmma

#endif // HIPTENSOR_CONTRACTION_TEST_HPP
