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

#ifndef HIPTENSOR_OPTIONS
#define HIPTENSOR_OPTIONS

#include <llvm/Support/CommandLine.h>
#include <stdlib.h>

#include "hiptensor_ostream.hpp"
#include "singleton.hpp"

namespace hiptensor
{
    struct HiptensorOptions : public LazySingleton<HiptensorOptions>
    {
        // For static initialization
        friend std::unique_ptr<HiptensorOptions> std::make_unique<HiptensorOptions>();

    private: // No public instantiation except make_unique.
             // No copy
        HiptensorOptions(HiptensorOptions const&)            = delete;
        HiptensorOptions& operator=(HiptensorOptions const&) = delete;

    public:
        HiptensorOptions(HiptensorOptions&&) = default;
        ~HiptensorOptions()                  = default;

        HiptensorOptions()
            : mOstream()
            , mOmitSkipped(false)
            , mOmitFailed(false)
            , mOmitPassed(false)
            , mOmitCout(false)
        {
        }

        void parseOptions()
        {
            // Get input/output file names
            llvm::cl::opt<std::string> inputFilename(
                "i", llvm::cl::desc("Specify input filename"), llvm::cl::value_desc("filename"));
            llvm::cl::opt<std::string> outputFilename(
                "o", llvm::cl::desc("Specify output filename"), llvm::cl::value_desc("filename"));

            mInputFilename  = inputFilename;
            mOutputFilename = outputFilename;

            std::cout << mInputFilename << ", " << mOutputFilename << '\n';
        }

        void setOmits(int mask)
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

        HiptensorOStream& ostream()
        {
            return mOstream;
        }

        bool omitSkipped()
        {
            return mOmitSkipped;
        }

        bool omitFailed()
        {
            return mOmitFailed;
        }

        bool omitPassed()
        {
            return mOmitPassed;
        }

        bool omitCout()
        {
            return mOmitCout;
        }

    protected:
        HiptensorOStream mOstream;

        bool        mOmitSkipped, mOmitFailed, mOmitPassed, mOmitCout;
        std::string mInputFilename, mOutputFilename;
    };

} // namespace hiptensor

#endif // HIPTENSOR_OPTIONS
