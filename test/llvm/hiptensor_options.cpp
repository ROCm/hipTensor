/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hiptensor/hiptensor-version.hpp>

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

llvm::cl::opt<int32_t>
    hiptensorOmitMask("omit",
                      llvm::cl::desc("Output verbosity omission\n 0x1 - Skipped Result\n 0x2 - "
                                     "Failed Result\n 0x4 - Passed Result\n 0x8 - Cout Messages"),
                      llvm::cl::value_desc("Bitmask [3:0]"),
                      llvm::cl::cat(HiptensorCategory));

namespace hiptensor
{
    HiptensorOptions::HiptensorOptions()
        : mOstream()
        , mOmitSkipped(false)
        , mOmitFailed(false)
        , mOmitPassed(false)
        , mOmitCout(false)
        , mUsingDefaultParams(true)
        , mInputFilename("")
        , mOutputFilename("")
    {
    }

    void HiptensorOptions::parseOptions(int argc, char** argv)
    {
        // Setup LLVM command line parser
        llvm::cl::SetVersionPrinter([](llvm::raw_ostream& os) {
            os << "hipTensor version: " << std::to_string(hiptensorGetVersion()) << "\n";
        });

        llvm::cl::HideUnrelatedOptions(HiptensorCategory);
        llvm::cl::ParseCommandLineOptions(argc, argv);

        // set I/O files if present
        mInputFilename  = hiptensorInputFilename;
        mOutputFilename = hiptensorOutputFilename;

        setOmits(hiptensorOmitMask);

        // Load testing params from YAML file if present
        if(!mInputFilename.empty())
        {
            mUsingDefaultParams = false;
        }

        // Initialize output stream
        if (!mOutputFilename.empty())
        {
            mOstream.initializeStream(mOutputFilename);
        }
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

    bool HiptensorOptions::usingDefaultConfig()
    {
        return mUsingDefaultParams;
    }

    std::string HiptensorOptions::inputFilename()
    {
        return mInputFilename;
    }

    std::string HiptensorOptions::outputFilename()
    {
        return mOutputFilename;
    }

} // namespace hiptensor
