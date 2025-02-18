/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_PERMUTATION_TEST_HELPERS_HPP
#define HIPTENSOR_PERMUTATION_TEST_HELPERS_HPP

#include <gtest/gtest.h>

#include "hiptensor_options.hpp"
#include "llvm/yaml_parser.hpp"

#ifdef HIPTENSOR_TEST_YAML_INCLUDE
#include HIPTENSOR_TEST_YAML_INCLUDE
#define HIPTENSOR_TEST_YAML_BUNDLE 1
#else
#define HIPTENSOR_TEST_YAML_BUNDLE 0
#endif // HIPTENSOR_TEST_YAML_INCLUDE

auto inline load_config_params()
{
    hiptensor::PermutationTestParams testParams;
    using Options     = hiptensor::HiptensorOptions;
    auto& testOptions = Options::instance();

    if(testOptions->usingDefaultConfig() && HIPTENSOR_TEST_YAML_BUNDLE)
    {
        auto params = hiptensor::YamlConfigLoader<hiptensor::PermutationTestParams>::loadFromString(
            HIPTENSOR_TEST_GET_YAML);
        if(params)
        {
            testParams = params.value();
        }
    }
    else
    {
        auto params = hiptensor::YamlConfigLoader<hiptensor::PermutationTestParams>::loadFromFile(
            testOptions->inputFilename());
        if(params)
        {
            testParams = params.value();
        }
    }

    // testParams.printParams();
    return testParams;
}

bool checkMemoryLimit(uint64_t minMemory,
                      uint64_t maxMemory,
                      uint64_t rank,
                      uint64_t i = 1,
                      uint64_t j = 1,
                      uint64_t k = 1,
                      uint64_t x = 1,
                      uint64_t y = 1,
                      uint64_t z = 1)
{
    uint64_t size = 2 * pow(i * j * k * x * y * z, 2);

    return (size >= minMemory) && (size <= maxMemory);
}

void generate_lengths(std::vector<std::vector<size_t>>  ranges,
                      std::vector<std::vector<size_t>>& lengths,
                      uint32_t                          rank)
{
    // Minimum and maximum memory footprint of each tensor
    uint64_t minMemory = 64;
    uint64_t maxMemory = 268435456;

    for(auto r = 0; r < ranges.size(); r++)
    {
        uint64_t i, j, k, x, y, z = 1;
        uint64_t minVal = ranges[r][0];
        uint64_t maxVal = ranges[r][1];
        uint64_t step   = ranges[r][2];

        // set loop limits basaed on rank
        uint64_t min_i = rank > 0 ? minVal : 1;
        uint64_t min_j = rank > 1 ? minVal : 1;
        uint64_t min_k = rank > 2 ? minVal : 1;
        uint64_t min_x = rank > 3 ? minVal : 1;
        uint64_t min_y = rank > 4 ? minVal : 1;
        uint64_t min_z = rank > 5 ? minVal : 1;

        uint64_t max_i = rank > 0 ? maxVal : 1;
        uint64_t max_j = rank > 1 ? maxVal : 1;
        uint64_t max_k = rank > 2 ? maxVal : 1;
        uint64_t max_x = rank > 3 ? maxVal : 1;
        uint64_t max_y = rank > 4 ? maxVal : 1;
        uint64_t max_z = rank > 5 ? maxVal : 1;

        for(z = min_z; z <= max_z; z *= step)
        {
            for(y = min_y; y <= max_y; y *= step)
            {
                for(x = min_x; x <= max_x; x *= step)
                {
                    for(k = min_k; k <= max_k; k *= step)
                    {
                        for(j = min_j; j <= max_j; j *= step)
                        {
                            for(i = min_i; i <= max_i; i *= step)
                            {
                                if(checkMemoryLimit(minMemory, maxMemory, rank, i, j, k, x, y, z))
                                {
                                    if(rank == 1)
                                    {
                                        lengths.push_back({i});
                                    }
                                    else if(rank == 2)
                                    {
                                        lengths.push_back({j, i});
                                    }
                                    else if(rank == 3)
                                    {
                                        lengths.push_back({k, j, i});
                                    }
                                    else if(rank == 4)
                                    {
                                        lengths.push_back({x, k, j, i});
                                    }
                                    else if(rank == 5)
                                    {
                                        lengths.push_back({y, x, k, j, i});
                                    }
                                    else if(rank == 6)
                                    {
                                        lengths.push_back({z, y, x, k, j, i});
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << lengths << std::endl;
}

auto inline load_config_helper()
{
    auto testParams = load_config_params();

    // Append sizes to problemLengths if problemRanges are given
    if(!testParams.problemRanges().empty())
    {
        uint32_t rank = testParams.permutedDims()[0].size();
        generate_lengths(testParams.problemRanges(), testParams.problemLengths(), rank);
    }

    // testParams.printParams();

    return ::testing::Combine(::testing::ValuesIn(testParams.dataTypes()),
                              ::testing::Values(testParams.logLevelMask()),
                              ::testing::ValuesIn(testParams.problemLengths()),
                              ::testing::ValuesIn(testParams.permutedDims()),
                              ::testing::ValuesIn(testParams.alphas()),
                              ::testing::ValuesIn(testParams.operators()));
}

#endif // HIPTENSOR_PERMUTATION_TEST_HELPERS_HPP
