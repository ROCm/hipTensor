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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "01_contraction/common.hpp"
#include "01_contraction/contraction_default_test_params.hpp"
#include "llvm/hiptensor_options.hpp"

#include <gtest/gtest.h>

// // Get input/output file names
// llvm::cl::OptionCategory   HiptensorCategory("hipTensor Options",
//                                            "Options for hipTensor testing framework");
// llvm::cl::opt<std::string> inputFilename("y",
//                                          llvm::cl::desc("Specify input YAML filename"),
//                                          llvm::cl::value_desc("filename"),
//                                          llvm::cl::cat(HiptensorCategory));
// llvm::cl::opt<std::string> outputFilename("o",
//                                           llvm::cl::desc("Specify output filename"),
//                                           llvm::cl::value_desc("filename"),
//                                           llvm::cl::cat(HiptensorCategory));

int main(int argc, char** argv)
{
    // Parse hiptensor test options
    using Options     = hiptensor::HiptensorOptions;
    auto& testOptions = Options::instance();
    testOptions->parseOptions(argc, argv);

    // Initialize Google Tests
    testing::InitGoogleTest(&argc, argv);

    // Run the tests
    int status = RUN_ALL_TESTS();

    return status;
}
