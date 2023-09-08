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

#include "../library/src/include/types.hpp"
#include "llvm/hiptensor_options.hpp"

#include "common.hpp"
#include "contraction_cpu_reference.hpp"
#include "contraction_test.hpp"

namespace hiptensor
{
    /*static*/ std::stringstream ContractionTest::sAPILogBuff = std::stringstream();

    static void logMessage(int32_t logLevel, const char* funcName /*=""*/, const char* msg /*=""*/)
    {
        ContractionTest::sAPILogBuff << msg;
    }

    ContractionTest::ContractionTest()
        : Base()
    {
        reset();

        // Handle our own outputs
        hiptensorLoggerOpenFile("/dev/null");
        hiptensorLoggerSetCallback(logMessage);
    }

    // Kernel run checks. Virtual as different Contraction kernels have different requirements
    // True = run test
    // False = skip test
    bool ContractionTest::checkDevice(hipDataType datatype) const
    {
        return (isF32Supported() && datatype == HIP_R_32F)
               || (isF64Supported() && datatype == HIP_R_64F);
    }

    bool ContractionTest::checkSizes() const
    {
        return true;
    }

    void ContractionTest::reset()
    {
        handle    = nullptr;
        workspace = nullptr;

        worksize = 0u;

        mRepeats          = 1u;
        mRunFlag          = true;
        mValidationResult = false;
        mMaxRelativeError = 0.0;
    }

    ContractionResource* ContractionTest::getResource() const
    {
        return DataStorage::instance().get();
    }

    void ContractionTest::SetUp()
    {
        // reset API log buffer
        sAPILogBuff.str(std::string());

        auto param        = Base::GetParam();
        auto testType     = std::get<0>(param);
        auto computeType  = std::get<1>(param);
        auto algorithm    = std::get<2>(param);
        auto operatorType = std::get<3>(param);
        auto workSizePref = std::get<4>(param);
        auto logLevel     = std::get<5>(param);
        auto lengths      = std::get<6>(param);
        auto strides      = std::get<7>(param);
        auto alpha        = std::get<8>(param);
        auto beta         = std::get<9>(param);

        // 4D tensors only at the moment.
        EXPECT_EQ(testType.size(), 4);
        EXPECT_EQ(lengths.size(), 6); // Format {'m', 'n', 'u', 'v', 'h', 'k'}
        if(!strides.empty())
        {
            EXPECT_EQ(strides.size(), 6);
        }

        auto ADataType = testType[0];
        auto BDataType = testType[1];
        auto CDataType = testType[2];
        auto DDataType = testType[3];

        EXPECT_TRUE((ADataType == HIP_R_32F) || (ADataType == HIP_R_64F));
        EXPECT_TRUE((BDataType == HIP_R_32F) || (BDataType == HIP_R_64F));
        EXPECT_TRUE((CDataType == HIP_R_32F) || (CDataType == HIP_R_64F)
                    || (CDataType == NONE_TYPE));
        EXPECT_TRUE((DDataType == HIP_R_32F) || (DDataType == HIP_R_64F));

        mRunFlag &= checkDevice(DDataType);

        if(!mRunFlag)
        {
            GTEST_SKIP();
        }
        else
        {
            std::vector<int> modeA{0, 1, 4, 5};
            std::vector<int> modeB{2, 3, 4, 5};
            std::vector<int> modeCD{0, 1, 2, 3};

            std::vector<int64_t> a_ms_ks_lengths, a_ms_ks_strides;
            for(auto mode : modeA)
            {
                a_ms_ks_lengths.push_back(lengths[mode]);
                if(!strides.empty())
                {
                    a_ms_ks_strides.push_back(strides[mode]);
                }
            }

            std::vector<int64_t> b_ns_ks_lengths, b_ns_ks_strides;
            for(auto mode : modeB)
            {
                b_ns_ks_lengths.push_back(lengths[mode]);
                if(!strides.empty())
                {
                    b_ns_ks_strides.push_back(strides[mode]);
                }
            }

            std::vector<int64_t> cd_ms_ns_lengths, cd_ms_ns_strides;
            for(auto mode : modeCD)
            {
                cd_ms_ns_lengths.push_back(lengths[mode]);
                if(!strides.empty())
                {
                    cd_ms_ns_strides.push_back(strides[mode]);
                }
            }

            size_t elementsA  = std::accumulate(a_ms_ks_lengths.begin(),
                                               a_ms_ks_lengths.end(),
                                               size_t{1},
                                               std::multiplies<size_t>());
            size_t elementsB  = std::accumulate(b_ns_ks_lengths.begin(),
                                               b_ns_ks_lengths.end(),
                                               size_t{1},
                                               std::multiplies<size_t>());
            size_t elementsCD = std::accumulate(cd_ms_ns_lengths.begin(),
                                                cd_ms_ns_lengths.end(),
                                                size_t{1},
                                                std::multiplies<size_t>());

            CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

            CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(logLevel));

            // lengths - m, n, u, v, h, k
            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
                handle,
                &a_ms_ks,
                modeA.size(),
                a_ms_ks_lengths.data(),
                strides.empty() ? NULL : a_ms_ks_strides.data(), /*stride*/
                ADataType,
                operatorType));

            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
                handle,
                &b_ns_ks,
                modeB.size(),
                b_ns_ks_lengths.data(),
                strides.empty() ? NULL : b_ns_ks_strides.data(), /*stride*/
                BDataType,
                operatorType));

            if(CDataType != NONE_TYPE)
            {
                CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
                    handle,
                    &c_ms_ns,
                    modeCD.size(),
                    cd_ms_ns_lengths.data(),
                    strides.empty() ? NULL : cd_ms_ns_strides.data(), /*stride*/
                    CDataType,
                    operatorType));
            }

            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
                handle,
                &d_ms_ns,
                modeCD.size(),
                cd_ms_ns_lengths.data(),
                strides.empty() ? NULL : cd_ms_ns_strides.data(), /*stride*/
                DDataType,
                operatorType));

            std::tuple<int32_t, int32_t, int32_t, int32_t> elementBytes(hipDataTypeSize(ADataType),
                                                                        hipDataTypeSize(BDataType),
                                                                        hipDataTypeSize(CDataType),
                                                                        hipDataTypeSize(DDataType));

            auto resource = getResource();
            resource->resizeStorage(lengths, elementBytes);

            if(ADataType == HIP_R_32F && BDataType == HIP_R_32F && DDataType == HIP_R_32F)
            {
                // Initialize matrix data on device
                fillLaunchKernel<float>((float*)resource->deviceA().get(), elementsA);
                fillLaunchKernel<float>((float*)resource->deviceB().get(), elementsB);
                if(CDataType == HIP_R_32F)
                {
                    fillLaunchKernel<float>((float*)resource->deviceC().get(), elementsCD);
                }
                fillValLaunchKernel<float>((float*)resource->deviceD().get(),
                                           elementsCD,
                                           std::numeric_limits<float>::signaling_NaN());
            }
            else if(ADataType == HIP_R_64F && BDataType == HIP_R_64F && DDataType == HIP_R_64F)
            {
                // Initialize matrix data on device
                fillLaunchKernel<double>((double*)resource->deviceA().get(), elementsA);
                fillLaunchKernel<double>((double*)resource->deviceB().get(), elementsB);
                if(CDataType == HIP_R_64F)
                {
                    fillLaunchKernel<double>((double*)resource->deviceC().get(), elementsCD);
                }
                fillValLaunchKernel<double>((double*)resource->deviceD().get(),
                                            elementsCD,
                                            std::numeric_limits<double>::signaling_NaN());
            }

            resource->copyDeviceToHostAll(elementBytes);

            uint32_t alignmentRequirementA;
            CHECK_HIPTENSOR_ERROR(hiptensorGetAlignmentRequirement(
                handle, resource->deviceA().get(), &a_ms_ks, &alignmentRequirementA));

            uint32_t alignmentRequirementB;
            CHECK_HIPTENSOR_ERROR(hiptensorGetAlignmentRequirement(
                handle, resource->deviceB().get(), &b_ns_ks, &alignmentRequirementB));

            uint32_t alignmentRequirementC = 0;
            if(CDataType != NONE_TYPE)
            {
                CHECK_HIPTENSOR_ERROR(hiptensorGetAlignmentRequirement(
                    handle, resource->deviceC().get(), &c_ms_ns, &alignmentRequirementC));
            }

            uint32_t alignmentRequirementD;
            CHECK_HIPTENSOR_ERROR(hiptensorGetAlignmentRequirement(
                handle, resource->deviceD().get(), &d_ms_ns, &alignmentRequirementD));

            CHECK_HIPTENSOR_ERROR(hiptensorInitContractionDescriptor(
                handle,
                &desc,
                &a_ms_ks,
                modeA.data(),
                alignmentRequirementA,
                &b_ns_ks,
                modeB.data(),
                alignmentRequirementB,
                (CDataType != NONE_TYPE) ? &c_ms_ns : nullptr,
                (CDataType != NONE_TYPE) ? modeCD.data() : nullptr,
                alignmentRequirementC,
                &d_ms_ns,
                modeCD.data(),
                alignmentRequirementD,
                computeType));
            /**************************
            * Set the algorithm to use
            ***************************/

            CHECK_HIPTENSOR_ERROR(hiptensorInitContractionFind(handle, &find, algorithm));

            /**********************
            * Query workspace
            **********************/

            CHECK_HIPTENSOR_ERROR(hiptensorContractionGetWorkspaceSize(
                handle, &desc, &find, workSizePref, &worksize));

            if(worksize > 0)
            {
                CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&workspace), worksize));
            }
        }
    }

    void ContractionTest::reportResults(std::ostream& stream,
                                        hipDataType   DDataType,
                                        bool          omitSkipped,
                                        bool          omitFailed,
                                        bool          omitPassed) const
    {
        // Conditionally print outputs
        if((mRunFlag || !omitSkipped) && (mValidationResult || !omitFailed)
           && (!mValidationResult || !omitPassed))
        {
            if(mPrintElements)
            {
                auto resource = getResource();

                int size = ((DDataType == HIP_R_32F) ? sizeof(float) : sizeof(double));

                size_t elementsA  = std::accumulate(a_ms_ks.mLengths.begin(),
                                                   a_ms_ks.mLengths.end(),
                                                   size_t{1},
                                                   std::multiplies<size_t>());
                size_t elementsB  = std::accumulate(b_ns_ks.mLengths.begin(),
                                                   b_ns_ks.mLengths.end(),
                                                   size_t{1},
                                                   std::multiplies<size_t>());
                size_t elementsCD = std::accumulate(d_ms_ns.mLengths.begin(),
                                                    d_ms_ns.mLengths.end(),
                                                    size_t{1},
                                                    std::multiplies<size_t>());

                auto D = resource->allocHost(elementsCD * size);
                resource->copyData(D, resource->deviceD(), elementsCD * size);

                if(DDataType == HIP_R_32F)
                {
                    stream << "Tensor A elements:\n";
                    hiptensorPrintArrayElements<float>(
                        stream, (float*)resource->hostA().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor B elements:\n";
                    hiptensorPrintArrayElements<float>(
                        stream, (float*)resource->hostB().get(), elementsB);
                    stream << std::endl;

                    stream << "Tensor C elements:\n";
                    hiptensorPrintArrayElements<float>(
                        stream, (float*)resource->hostC().get(), elementsCD);
                    stream << std::endl;

                    stream << "Tensor D elements:\n";
                    hiptensorPrintArrayElements<float>(stream, (float*)D.get(), elementsCD);
                    stream << std::endl;
                }
                else
                {
                    stream << "Tensor A elements:\n";
                    hiptensorPrintArrayElements<double>(
                        stream, (double*)resource->hostA().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor B elements:\n";
                    hiptensorPrintArrayElements<double>(
                        stream, (double*)resource->hostB().get(), elementsB);
                    stream << std::endl;

                    stream << "Tensor C elements:\n";
                    hiptensorPrintArrayElements<double>(
                        stream, (double*)resource->hostC().get(), elementsCD);
                    stream << std::endl;

                    stream << "Tensor D elements:\n";
                    hiptensorPrintArrayElements<double>(stream, (double*)D.get(), elementsCD);
                    stream << std::endl;
                }
            }
        }
    }

    void ContractionTest::RunKernel()
    {
        auto param        = Base::GetParam();
        auto testType     = std::get<0>(param);
        auto computeType  = std::get<1>(param);
        auto algorithm    = std::get<2>(param);
        auto operatorType = std::get<3>(param);
        auto workSizePref = std::get<4>(param);
        auto logLevel     = std::get<5>(param);
        auto lengths      = std::get<6>(param);
        auto strides      = std::get<7>(param);
        auto alpha        = std::get<8>(param);
        auto beta         = std::get<9>(param);

        if(mRunFlag)
        {
            auto ADataType = testType[0];
            auto BDataType = testType[1];
            auto CDataType = testType[2];
            auto DDataType = testType[3];

            CHECK_HIPTENSOR_ERROR(
                hiptensorInitContractionPlan(handle, &plan, &desc, &find, worksize));

            auto resource = getResource();

            CHECK_HIPTENSOR_ERROR(hiptensorContraction(handle,
                                                       &plan,
                                                       (void*)&alpha,
                                                       resource->deviceA().get(),
                                                       resource->deviceB().get(),
                                                       (void*)&beta,
                                                       resource->deviceC().get(),
                                                       resource->deviceD().get(),
                                                       workspace,
                                                       worksize,
                                                       0 /* stream */));

            CHECK_HIPTENSOR_ERROR(hiptensorContractionReference((void*)&alpha,
                                                                resource->hostA().get(),
                                                                resource->hostB().get(),
                                                                (void*)&beta,
                                                                resource->hostC().get(),
                                                                resource->hostD().get(),
                                                                a_ms_ks.mLengths,
                                                                a_ms_ks.mStrides,
                                                                b_ns_ks.mLengths,
                                                                b_ns_ks.mStrides,
                                                                d_ms_ns.mLengths,
                                                                d_ms_ns.mStrides,
                                                                d_ms_ns.mLengths,
                                                                d_ms_ns.mStrides,
                                                                ADataType,
                                                                BDataType,
                                                                CDataType,
                                                                DDataType,
                                                                workspace));

            size_t elementsCD = std::accumulate(c_ms_ns.mLengths.begin(),
                                                c_ms_ns.mLengths.end(),
                                                size_t{1},
                                                std::multiplies<size_t>());

            int  sizeD = elementsCD * ((DDataType == HIP_R_32F) ? sizeof(float) : sizeof(double));
            auto reference = resource->allocDevice(sizeD);
            resource->copyData(reference, resource->hostD(), sizeD);

            if(DDataType == HIP_R_32F)
            {
                std::tie(mValidationResult, mMaxRelativeError) = compareEqualLaunchKernel<float>(
                    (float*)resource->deviceD().get(), (float*)reference.get(), elementsCD);
            }
            else if(DDataType == HIP_R_64F)
            {
                std::tie(mValidationResult, mMaxRelativeError) = compareEqualLaunchKernel<double>(
                    (double*)resource->deviceD().get(), (double*)reference.get(), elementsCD);
            }

            EXPECT_TRUE(mValidationResult) << "Max relative error: " << mMaxRelativeError;

            using Options        = hiptensor::HiptensorOptions;
            auto& loggingOptions = Options::instance();

            if(!loggingOptions->omitCout())
            {
                reportResults(std::cout,
                              DDataType,
                              loggingOptions->omitSkipped(),
                              loggingOptions->omitFailed(),
                              loggingOptions->omitPassed());
            }

            if(loggingOptions->ostream().isOpen())
            {
                reportResults(loggingOptions->ostream().fstream(),
                              DDataType,
                              loggingOptions->omitSkipped(),
                              loggingOptions->omitFailed(),
                              loggingOptions->omitPassed());
            }
        }
    }

    void ContractionTest::TearDown()
    {
        if(mRunFlag)
        {
            CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
        }
    }

} // namespace hiptensor
