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

#include <llvm/Support/CommandLine.h>

#include "hiptensor_options.hpp"

// extern llvm::cl::opt<std::string> inputFilename;
// extern llvm::cl::opt<std::string> outputFilename;

// Get input/output file names
llvm::cl::OptionCategory   HiptensorCategory("hipTensor Options",
                                           "Options for hipTensor testing framework");
llvm::cl::opt<std::string> hiptensorInputFilename("y",
                                                  llvm::cl::desc("Specify input YAML filename"),
                                                  llvm::cl::value_desc("filename"),
                                                  llvm::cl::cat(HiptensorCategory));
llvm::cl::opt<std::string> hiptensorOutputFilename("o",
                                                   llvm::cl::desc("Specify output filename"),
                                                   llvm::cl::value_desc("filename"),
                                                   llvm::cl::cat(HiptensorCategory));

namespace hiptensor
{
    HiptensorOptions::HiptensorOptions()
        : mOstream()
        , mOmitSkipped(false)
        , mOmitFailed(false)
        , mOmitPassed(false)
        , mOmitCout(false)
        , mUsingYAML(false)
        , mInputFilename("")
        , mOutputFilename("")
    {
    }

    void HiptensorOptions::parseOptions(int argc, char** argv)
    {
        // Setup LLVM command line parser
        llvm::StringMap<llvm::cl::Option*>& optionMap = llvm::cl::getRegisteredOptions();

        assert(optionMap.count("version") > 0);
        optionMap["version"]->setDescription("Display the version of LLVM used");
        optionMap["version"]->setArgStr("llvm-version");

        llvm::cl::HideUnrelatedOptions(HiptensorCategory);
        llvm::cl::ParseCommandLineOptions(argc, argv);

        // set I/O files if present
        mInputFilename  = hiptensorInputFilename;
        mOutputFilename = hiptensorOutputFilename;
        std::cout << "\n\nCommand Line Parameters \n\n";
        std::cout << mInputFilename << ", " << mOutputFilename << '\n';

        // if input file is valid hook into YAML parsing for setting parameter values

        // otherwise use default parameter values
    }

    void HiptensorOptions::setOmits(int mask)
    {
        if(mask & 1)
            mOmitSkipped = true;
        if(mask & 2)
            mOmitFailed = true;
        if(mask & 4)
            mOmitPassed = true;
        if(mask & 8)
            mOmitCout = true;
    }

    HiptensorOStream& HiptensorOptions::ostream()
    {
        return mOstream;
    }

    bool HiptensorOptions::omitSkipped()
    {
        return mOmitSkipped;
    }

    bool HiptensorOptions::omitFailed()
    {
        return mOmitFailed;
    }

    bool HiptensorOptions::omitPassed()
    {
        return mOmitPassed;
    }

    bool HiptensorOptions::omitCout()
    {
        return mOmitCout;
    }

    auto HiptensorOptions::testParams() -> ContractionTestParams&
    {
        return mTestParams;
    }

} // namespace hiptensor
