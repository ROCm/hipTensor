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

#include <hiptensor/hiptensor.hpp>
#include <hiptensor/hiptensor_types.hpp>

#include "contraction_test.hpp"

class BilinearF32ContractionTest : public hiptensor::ContractionTest
{
};

TEST_P(BilinearF32ContractionTest, RunKernel)
{
    static bool ranWarmup = false;
    if(!ranWarmup)
    {
        this->Warmup();
        ranWarmup = true;
    }
    this->RunKernel();
}

auto loadConfig()
{
    hiptensor::ContractionTestParams testParams;
    std::string                      path = "";

    using Options     = hiptensor::HiptensorOptions;
    auto& testOptions = Options::instance();

    if(testOptions->usingDefaultConfig())
    {
        path = "/home/mkarunan/WORKSPACE/hipTensor-test/test/01_contraction/configs/bilinear_f32_test_params.yaml";
    }
    else
    {
        path = testOptions->inputFilename();
    }

    testParams = hiptensor::YamlConfigLoader<hiptensor::ContractionTestParams>::loadFromFile(path);
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

INSTANTIATE_TEST_SUITE_P(ContractionTests, BilinearF32ContractionTest, loadConfig());
