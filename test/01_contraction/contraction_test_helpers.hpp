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

#ifndef HIPTENSOR_CONTRACTION_TEST_HELPERS_HPP
#define HIPTENSOR_CONTRACTION_TEST_HELPERS_HPP

#include <gtest/gtest.h>

#include "llvm/hiptensor_options.hpp"
#include "llvm/yaml_parser.hpp"

#ifdef HIPTENSOR_TEST_YAML_INCLUDE
#include HIPTENSOR_TEST_YAML_INCLUDE
#define HIPTENSOR_TEST_YAML_BUNDLE 1
#else
#define HIPTENSOR_TEST_YAML_BUNDLE 0
#endif // HIPTENSOR_TEST_YAML_INCLUDE

auto inline load_config_params()
{
    hiptensor::ContractionTestParams testParams;
    using Options     = hiptensor::HiptensorOptions;
    auto& testOptions = Options::instance();

    if(testOptions->usingDefaultConfig() && HIPTENSOR_TEST_YAML_BUNDLE)
    {
        auto params = hiptensor::YamlConfigLoader<hiptensor::ContractionTestParams>::loadFromString(
            HIPTENSOR_TEST_GET_YAML);
        if(params)
        {
            testParams = params.value();
        }
    }
    else
    {
        auto params = hiptensor::YamlConfigLoader<hiptensor::ContractionTestParams>::loadFromFile(
            testOptions->inputFilename());
        if(params)
        {
            testParams = params.value();
        }
    }

    // testParams.printParams();
    return testParams;
}

auto inline load_combined_config_params()
{
    auto testParams = load_config_params();
    return ::testing::Combine(::testing::ValuesIn(testParams.dataTypes()),
                              ::testing::ValuesIn(testParams.algorithms()),
                              ::testing::ValuesIn(testParams.operators()),
                              ::testing::ValuesIn(testParams.workSizePrefrences()),
                              ::testing::Values(testParams.logLevelMask()),
                              ::testing::ValuesIn(testParams.problemLengths()),
                              ::testing::ValuesIn(testParams.problemStrides()),
                              ::testing::ValuesIn(testParams.problemModes()),
                              ::testing::ValuesIn(testParams.alphas()),
                              ::testing::ValuesIn(testParams.betas()),
                              ::testing::ValuesIn(testParams.testTypes()));
}

auto inline load_sequence_config_params()
{
    auto testParams = load_config_params();

    auto dataTypes          = testParams.dataTypes();
    auto algorithms         = testParams.algorithms();
    auto operators          = testParams.operators();
    auto workSizePrefrences = testParams.workSizePrefrences();
    auto logLevelMask       = std::vector<std::decay_t<decltype(testParams.logLevelMask())>>(
        1, testParams.logLevelMask());
    auto problemLengths = testParams.problemLengths();
    auto problemStrides = testParams.problemStrides();
    auto problemModes   = testParams.problemModes();
    auto alphas         = testParams.alphas();
    auto betas          = testParams.betas();
    auto testTypes      = testParams.testTypes();

    std::vector<size_t> lengths   = {dataTypes.size(),
                                     algorithms.size(),
                                     operators.size(),
                                     workSizePrefrences.size(),
                                     logLevelMask.size(),
                                     problemLengths.size(),
                                     problemStrides.size(),
                                     problemModes.size(),
                                     alphas.size(),
                                     betas.size(),
                                     testTypes.size()};
    auto                maxLength = *std::max_element(lengths.begin(), lengths.end());

    dataTypes.resize(maxLength, dataTypes.back());
    algorithms.resize(maxLength, algorithms.back());
    operators.resize(maxLength, operators.back());
    workSizePrefrences.resize(maxLength, workSizePrefrences.back());
    logLevelMask.resize(maxLength, logLevelMask.back());
    problemLengths.resize(maxLength, problemLengths.back());
    problemStrides.resize(maxLength, problemStrides.back());
    problemModes.resize(maxLength, problemModes.back());
    alphas.resize(maxLength, alphas.back());
    betas.resize(maxLength, betas.back());
    testTypes.resize(maxLength, testTypes.back());

    using ParamsTuple = decltype(std::make_tuple(dataTypes.front(),
                                                 algorithms.front(),
                                                 operators.front(),
                                                 workSizePrefrences.front(),
                                                 logLevelMask.front(),
                                                 problemLengths.front(),
                                                 problemStrides.front(),
                                                 problemModes.front(),
                                                 alphas.front(),
                                                 betas.front(),
                                                 testTypes.front()));

    std::vector<ParamsTuple> paramsSequence;
    for(int i = 0; i < maxLength; i++)
    {
        paramsSequence.push_back(std::make_tuple(dataTypes[i],
                                                 algorithms[i],
                                                 operators[i],
                                                 workSizePrefrences[i],
                                                 logLevelMask[i],
                                                 problemLengths[i],
                                                 problemStrides[i],
                                                 problemModes[i],
                                                 alphas[i],
                                                 betas[i],
                                                 testTypes[i]));
    }

    return ::testing::ValuesIn(paramsSequence);
}

#endif // HIPTENSOR_CONTRACTION_TEST_HELPERS_HPP
