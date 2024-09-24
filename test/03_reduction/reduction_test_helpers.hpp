/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_REDUCTION_TEST_HELPERS_HPP
#define HIPTENSOR_REDUCTION_TEST_HELPERS_HPP

#include <gtest/gtest.h>

#include "llvm/hiptensor_options.hpp"
#include "llvm/yaml_parser.hpp"

#ifdef HIPTENSOR_TEST_YAML_INCLUDE
#include HIPTENSOR_TEST_YAML_INCLUDE
#define HIPTENSOR_TEST_YAML_BUNDLE 1
#else
#define HIPTENSOR_TEST_YAML_BUNDLE 0
#endif // HIPTENSOR_TEST_YAML_INCLUDE

auto inline load_config_helper()
{
    hiptensor::ReductionTestParams testParams;
    using Options     = hiptensor::HiptensorOptions;
    auto& testOptions = Options::instance();

    if(testOptions->usingDefaultConfig() && HIPTENSOR_TEST_YAML_BUNDLE)
    {
        auto params = hiptensor::YamlConfigLoader<hiptensor::ReductionTestParams>::loadFromString(
            HIPTENSOR_TEST_GET_YAML);
        if(params)
        {
            testParams = params.value();
        }
    }
    else
    {
        auto params = hiptensor::YamlConfigLoader<hiptensor::ReductionTestParams>::loadFromFile(
            testOptions->inputFilename());
        if(params)
        {
            testParams = params.value();
        }
    }

    // testParams.printParams();

    return ::testing::Combine(::testing::ValuesIn(testParams.dataTypes()),
                              ::testing::Values(testParams.logLevelMask()),
                              ::testing::ValuesIn(testParams.problemLengths()),
                              ::testing::ValuesIn(testParams.outputDims()),
                              ::testing::ValuesIn(testParams.alphas()),
                              ::testing::ValuesIn(testParams.betas()),
                              ::testing::ValuesIn(testParams.operators()),
                              ::testing::ValuesIn(testParams.testTypes()));
}

#endif // HIPTENSOR_REDUCTION_TEST_HELPERS_HPP
