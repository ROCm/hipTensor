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
#include <hiptensor/hiptensor.hpp>

#include "data_types.hpp"
#include "logger.hpp"
#include "reduction/reduction_cpu_reference.hpp"
#include "reduction_test.hpp"
#include "utils.hpp"
#include "llvm/hiptensor_options.hpp"

namespace hiptensor
{
    /*static*/ std::stringstream ReductionTest::sAPILogBuff = std::stringstream();

    static void logMessage(int32_t logLevel, const char* funcName /*=""*/, const char* msg /*=""*/)
    {
        ReductionTest::sAPILogBuff << msg;
    }

    ReductionTest::ReductionTest()
        : Base()
    {
        reset();
        // Handle our own outputs
        hiptensorLoggerOpenFile("/dev/null");
        hiptensorLoggerSetCallback(logMessage);
    }

    // Kernel run checks. Virtual as different Reduction kernels have different requirements
    // True = run test
    // False = skip test
    bool ReductionTest::checkDevice(hipDataType            datatype,
                                    hiptensorComputeType_t computeDataType) const
    {
        return !(((datatype == HIP_R_32F || computeDataType == HIP_R_32F) && !isF32Supported())
                 || ((datatype == HIP_R_64F || computeDataType == HIP_R_64F) && !isF64Supported()));
    }

    bool ReductionTest::checkSizes() const
    {
        return true;
    }

    void ReductionTest::reset()
    {
        handle = nullptr;

        mRepeats          = 1u;
        mRunFlag          = true;
        mValidationResult = false;
        mMaxRelativeError = 0.0;
    }

    ReductionResource* ReductionTest::getResource() const
    {
        return DataStorage::instance().get();
    }

    void ReductionTest::SetUp()
    {
        // reset API log buffer
        sAPILogBuff.str(std::string());

        auto param      = Base::GetParam();
        auto testType   = std::get<0>(param);
        auto logLevel   = std::get<1>(param);
        auto lengths    = std::get<2>(param);
        auto outputDims = std::get<3>(param);
        auto alpha      = std::get<4>(param);
        auto beta       = std::get<5>(param);
        auto op         = std::get<6>(param);

        EXPECT_TRUE((lengths.size() > 1) && (lengths.size() <= 6));
        EXPECT_TRUE((outputDims.size() >= 1) && (outputDims.size() <= 6));

        EXPECT_TRUE((op == HIPTENSOR_OP_ADD) || (op == HIPTENSOR_OP_MUL) || (op == HIPTENSOR_OP_MAX)
                    || (op == HIPTENSOR_OP_MIN));

        EXPECT_EQ(testType.size(), 2); // HIP_R_16F or HIP_R_32F
        auto acDataType      = testType[0];
        auto computeDataType = convertToComputeType(testType[1]);
        EXPECT_TRUE((acDataType == HIP_R_16F && computeDataType == HIPTENSOR_COMPUTE_16F)
                    || (acDataType == HIP_R_16F && computeDataType == HIPTENSOR_COMPUTE_32F)
                    || (acDataType == HIP_R_16BF && computeDataType == HIPTENSOR_COMPUTE_16BF)
                    || (acDataType == HIP_R_16BF && computeDataType == HIPTENSOR_COMPUTE_32F)
                    || (acDataType == HIP_R_32F && computeDataType == HIPTENSOR_COMPUTE_32F)
                    || (acDataType == HIP_R_64F && computeDataType == HIPTENSOR_COMPUTE_64F));

        mRunFlag &= checkDevice(acDataType, computeDataType);
        mRunFlag &= lengths.size() > outputDims.size();

        if(!mRunFlag)
        {
            GTEST_SKIP();
        }
        else
        {
            std::vector<size_t> outputLengths;
            for(auto dim : outputDims)
            {
                outputLengths.push_back(lengths[dim]);
            }
            getResource()->setupStorage(lengths, outputLengths, acDataType);

            // set mPrintElements to true to print element
            mPrintElements = false;
        }
    }

    void ReductionTest::reportResults(std::ostream& stream,
                                      hipDataType   dataType,
                                      bool          omitSkipped,
                                      bool          omitFailed,
                                      bool          omitPassed) const
    {
        // Conditionally print outputs
        if((mRunFlag || !omitSkipped) && (mValidationResult || !omitFailed)
           && (!mValidationResult || !omitPassed))
        {
            stream << ReductionTest::sAPILogBuff.str();

            if(mPrintElements)
            {
                auto resource = getResource();

                auto param      = Base::GetParam();
                auto testType   = std::get<0>(param);
                auto logLevel   = std::get<1>(param);
                auto lengths    = std::get<2>(param);
                auto outputDims = std::get<3>(param);
                auto alpha      = std::get<4>(param);
                auto beta       = std::get<5>(param);
                auto op         = std::get<6>(param);
                stream << "Input [type: " << testType << ", lengths: " << lengths
                       << ", outputDims: " << outputDims << ", alpha: " << alpha
                       << ", beta: " << beta << ", opReduce: " << op << "]\n";

                size_t elementsA = resource->getCurrentMatrixAElement();
                size_t elementsC = resource->getCurrentMatrixCElement();

                if(dataType == HIP_R_16BF)
                {
                    stream << "Tensor A elements (" << elementsA << "):\n";
                    hiptensorPrintArrayElements<bfloat16_t>(
                        stream, (bfloat16_t*)resource->hostA().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor C elements (" << elementsC << "):\n";
                    hiptensorPrintArrayElements<bfloat16_t>(
                        stream, (bfloat16_t*)resource->hostC().get(), elementsC);
                    stream << std::endl;

                    stream << "Refenrence elements (" << elementsC << "):\n";
                    hiptensorPrintArrayElements<bfloat16_t>(
                        stream, (bfloat16_t*)resource->hostReference().get(), elementsC);
                    stream << std::endl;
                }
                else if(dataType == HIP_R_16F)
                {
                    stream << "Tensor A elements (" << elementsA << "):\n";
                    hiptensorPrintArrayElements<float16_t>(
                        stream, (float16_t*)resource->hostA().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor C elements (" << elementsC << "):\n";
                    hiptensorPrintArrayElements<float16_t>(
                        stream, (float16_t*)resource->hostC().get(), elementsC);
                    stream << std::endl;

                    stream << "Refenrence elements (" << elementsC << "):\n";
                    hiptensorPrintArrayElements<float16_t>(
                        stream, (float16_t*)resource->hostReference().get(), elementsC);
                    stream << std::endl;
                }
                else if(dataType == HIP_R_32F)
                {
                    stream << "Tensor A elements (" << elementsA << "):\n";
                    hiptensorPrintArrayElements<float32_t>(
                        stream, (float32_t*)resource->hostA().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor C elements (" << elementsC << "):\n";
                    hiptensorPrintArrayElements<float32_t>(
                        stream, (float32_t*)resource->hostC().get(), elementsC);
                    stream << std::endl;

                    stream << "Refenrence elements (" << elementsC << "):\n";
                    hiptensorPrintArrayElements<float32_t>(
                        stream, (float32_t*)resource->hostReference().get(), elementsC);
                    stream << std::endl;
                }
                else if(dataType == HIP_R_64F)
                {
                    stream << "Tensor A elements (" << elementsA << "):\n";
                    hiptensorPrintArrayElements<float64_t>(
                        stream, (float64_t*)resource->hostA().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor C elements (" << elementsC << "):\n";
                    hiptensorPrintArrayElements<float64_t>(
                        stream, (float64_t*)resource->hostC().get(), elementsC);
                    stream << std::endl;

                    stream << "Refenrence elements (" << elementsC << "):\n";
                    hiptensorPrintArrayElements<float64_t>(
                        stream, (float64_t*)resource->hostReference().get(), elementsC);
                    stream << std::endl;
                }
            }
        }
    }

    void ReductionTest::RunKernel()
    {
        auto param      = Base::GetParam();
        auto testType   = std::get<0>(param);
        auto logLevel   = std::get<1>(param);
        auto lengths    = std::get<2>(param);
        auto outputDims = std::get<3>(param);
        auto alpha      = std::get<4>(param);
        auto beta       = std::get<5>(param);
        auto opReduce   = std::get<6>(param);

        auto acDataType      = testType[0];
        auto computeDataType = convertToComputeType(testType[1]);

        if(!mRunFlag)
        {
            GTEST_SKIP();
        }
        auto resource = getResource();

        if(mRunFlag)
        {
            std::vector<int> modeA(lengths.size());
            std::iota(modeA.begin(), modeA.end(), 'a');
            std::vector<int> modeC;
            for(auto dim : outputDims)
            {
                modeC.push_back(modeA[dim]);
            }

            int                              nmodeA = modeA.size();
            int                              nmodeC = modeC.size();
            std::unordered_map<int, int64_t> extent;
            for(auto [modeIt, i] = std::tuple{modeA.begin(), 0}; modeIt != modeA.end();
                ++modeIt, ++i)
            {
                extent[*modeIt] = lengths[i];
            }

            std::vector<int64_t> extentA;
            for(auto mode : modeA)
                extentA.push_back(extent[mode]);
            std::vector<int64_t> extentC;
            for(auto mode : modeC)
                extentC.push_back(extent[mode]);

            hiptensorStatus_t  err;
            hiptensorHandle_t* handle;
            CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

            hiptensorTensorDescriptor_t descA;
            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                                &descA,
                                                                nmodeA,
                                                                extentA.data(),
                                                                NULL /* stride */,
                                                                acDataType,
                                                                HIPTENSOR_OP_IDENTITY));

            hiptensorTensorDescriptor_t descC;
            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                                &descC,
                                                                nmodeC,
                                                                extentC.data(),
                                                                NULL /* stride */,
                                                                acDataType,
                                                                HIPTENSOR_OP_IDENTITY));

            uint64_t worksize = 0;
            CHECK_HIPTENSOR_ERROR(hiptensorReductionGetWorkspaceSize(handle,
                                                                     resource->deviceA().get(),
                                                                     &descA,
                                                                     modeA.data(),
                                                                     resource->deviceC().get(),
                                                                     &descC,
                                                                     modeC.data(),
                                                                     resource->deviceC().get(),
                                                                     &descC,
                                                                     modeC.data(),
                                                                     opReduce,
                                                                     computeDataType,
                                                                     &worksize));
            void* work = nullptr;
            if(worksize > 0)
            {
                if(hipSuccess != hipMalloc(&work, worksize))
                {
                    work     = nullptr;
                    worksize = 0;
                }
            }

            double alphaValue{};
            double betaValue{};
            writeVal(&alphaValue, computeDataType, {computeDataType, alpha});
            writeVal(&betaValue, computeDataType, {computeDataType, beta});
            CHECK_HIPTENSOR_ERROR(hiptensorReduction(handle,
                                                     (const void*)&alphaValue,
                                                     resource->deviceA().get(),
                                                     &descA,
                                                     modeA.data(),
                                                     (const void*)&betaValue,
                                                     resource->deviceC().get(),
                                                     &descC,
                                                     modeC.data(),
                                                     resource->deviceC().get(),
                                                     &descC,
                                                     modeC.data(),
                                                     opReduce,
                                                     computeDataType,
                                                     work,
                                                     worksize,
                                                     0 /* stream */));

            resource->copyCToHost();

            CHECK_HIPTENSOR_ERROR(hiptensorReductionReference(&alphaValue,
                                                              resource->hostA().get(),
                                                              &descA,
                                                              modeA.data(),
                                                              &betaValue,
                                                              resource->hostReference().get(),
                                                              &descC,
                                                              modeC.data(),
                                                              resource->hostReference().get(),
                                                              &descC,
                                                              modeC.data(),
                                                              opReduce,
                                                              computeDataType,
                                                              0 /* stream */));
            resource->copyReferenceToDevice();

            if(acDataType == HIP_R_16F)
            {
                std::tie(mValidationResult, mMaxRelativeError)
                    = compareEqualLaunchKernel<float16_t>(
                        (float16_t*)resource->deviceC().get(),
                        (float16_t*)resource->deviceReference().get(),
                        resource->getCurrentMatrixCElement(),
                        computeDataType);
            }
            else if(acDataType == HIP_R_16BF)
            {
                std::tie(mValidationResult, mMaxRelativeError)
                    = compareEqualLaunchKernel<bfloat16_t>(
                        (bfloat16_t*)resource->deviceC().get(),
                        (bfloat16_t*)resource->deviceReference().get(),
                        resource->getCurrentMatrixCElement(),
                        computeDataType);
            }
            else if(acDataType == HIP_R_32F)
            {
                auto reducedSize
                    = resource->getCurrentMatrixAElement() / resource->getCurrentMatrixCElement();
                double tolerance = reducedSize * getEpsilon(computeDataType);
                std::tie(mValidationResult, mMaxRelativeError)
                    = compareEqualLaunchKernel<float32_t>(
                        (float32_t*)resource->deviceC().get(),
                        (float32_t*)resource->deviceReference().get(),
                        resource->getCurrentMatrixCElement(),
                        computeDataType,
                        tolerance);
            }
            else if(acDataType == HIP_R_64F)
            {
                auto reducedSize
                    = resource->getCurrentMatrixAElement() / resource->getCurrentMatrixCElement();
                double tolerance = reducedSize * getEpsilon(computeDataType);
                std::tie(mValidationResult, mMaxRelativeError)
                    = compareEqualLaunchKernel<float64_t>(
                        (float64_t*)resource->deviceC().get(),
                        (float64_t*)resource->deviceReference().get(),
                        resource->getCurrentMatrixCElement(),
                        computeDataType,
                        tolerance);
            }
        }

        EXPECT_TRUE(mValidationResult) << "Max relative error: " << mMaxRelativeError;

        using Options        = hiptensor::HiptensorOptions;
        auto& loggingOptions = Options::instance();

        if(!loggingOptions->omitCout())
        {
            reportResults(std::cout,
                          acDataType,
                          loggingOptions->omitSkipped(),
                          loggingOptions->omitFailed(),
                          loggingOptions->omitPassed());
        }

        if(loggingOptions->ostream().isOpen())
        {
            reportResults(loggingOptions->ostream().fstream(),
                          acDataType,
                          loggingOptions->omitSkipped(),
                          loggingOptions->omitFailed(),
                          loggingOptions->omitPassed());
        }
    }

    void ReductionTest::TearDown() {}

} // namespace hiptensor
