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

#ifndef HIPTENSOR_CONTRACTION_TEST_IMPL_HPP
#define HIPTENSOR_CONTRACTION_TEST_IMPL_HPP

#include "common.hpp"
#include "contraction_test_params.hpp"
#include "llvm/hiptensor_options.hpp"
#include "llvm/yaml_parser.hpp"

#include <gtest/gtest.h>

namespace hiptensor
{
    struct ContractionTest : public ::testing::TestWithParam<
                                 std::tuple<typename ContractionTestParams::TestDataTypeT,
                                            typename ContractionTestParams::TestComputeTypeT,
                                            typename ContractionTestParams::AlgorithmT,
                                            typename ContractionTestParams::OperatorT,
                                            typename ContractionTestParams::WorkSizePrefT,
                                            typename ContractionTestParams::LogLevelT,
                                            typename ContractionTestParams::LengthsT,
                                            typename ContractionTestParams::StridesT,
                                            typename ContractionTestParams::AlphaT,
                                            typename ContractionTestParams::BetaT>>
    {
        using Base
            = ::testing::TestWithParam<std::tuple<typename ContractionTestParams::TestDataTypeT,
                                                  typename ContractionTestParams::TestComputeTypeT,
                                                  typename ContractionTestParams::AlgorithmT,
                                                  typename ContractionTestParams::OperatorT,
                                                  typename ContractionTestParams::WorkSizePrefT,
                                                  typename ContractionTestParams::LogLevelT,
                                                  typename ContractionTestParams::LengthsT,
                                                  typename ContractionTestParams::StridesT,
                                                  typename ContractionTestParams::AlphaT,
                                                  typename ContractionTestParams::BetaT>>;

        virtual void reset()
        {
            handle = plan = desc = find = workspace = nullptr;

            worksize = 0u;
            
            mRepeats = 1u;
            mRunFlag          = true;
            mValidationResult = false;
            mMaxRelativeError = 0.0;
        }

        HipResource* getResource() const
        {
            return DataStorage::instance().get();
        }

        void SetUp() override
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

            std::cout << testType << ", " << computeType << ", " << algorithm << ", "
                      << operatorType << ", " << workSizePref << ", " << logLevel << ", " << lengths
                      << ", " << strides << ", " << alpha << ", " << beta << "\n";

            assert ( testType.size() == 4);
            assert ( lengths.size() == 6); // Format {'m', 'n', 'u', 'v', 'h', 'k'}

            using ADataType = testType[0];
            using BDataType = testType[1];
            using CDataType = testType[2];
            using DDataType = testType[3];
            
            std::vector<int> modeA{0, 1, 4, 5};
            std::vector<int> modeB{2, 3, 4, 5};
            std::vector<int> modeCD{0, 1, 2, 3};

            std::vector<int64_t> a_ms_ks_lengths;
            for(auto mode : modeA)
            {
                a_ms_ks_lengths.push_back(lengths[mode]);
            }

            std::vector<int64_t> b_ns_ks_lengths;
            for(auto mode : modeB)
            {
                b_ns_ks_lengths.push_back(lengths[mode]);
            }

            std::vector<int64_t> cd_ms_ns_lengths;
            for(auto mode : modeC)
            {
                cd_ms_ns_lengths.push_back(lengths[mode]);
            }

            size_t elementsA = std::accumulate(
            a_ms_ks_lengths.begin(), a_ms_ks_lengths.end(), size_t{1}, std::multiplies<size_t>());
            size_t elementsB = std::accumulate(
            b_ns_ks_lengths.begin(), b_ns_ks_lengths.end(), size_t{1}, std::multiplies<size_t>());
            size_t elementsCD = std::accumulate(
            cd_ms_ns_lengths.begin(), cd_ms_ns_lengths.end(), size_t{1}, std::multiplies<size_t>());

            CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

            CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(logLevel));

            // lengths - m, n, u, v, h, k
            hiptensorTensorDescriptor_t a_ms_ks;
            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                                &a_ms_ks,
                                                                modeA.size(),
                                                                a_ms_ks_lengths,
                                                                strides ? strides[0].data() : NULL, /*stride*/
                                                                ADataType,
                                                                operatorType));

            hiptensorTensorDescriptor_t b_ns_ks;
            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                                &b_ns_ks,
                                                                modeB.size(),
                                                                b_ns_ks_lengths,
                                                                strides ? strides[1].data() : NULL, /*stride*/
                                                                BDataType,
                                                                operatorType));

            if(testType[2]  != NONE_TYPE)
            {
                hiptensorTensorDescriptor_t c_ms_ns;
                CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                                    &c_ms_ns,
                                                                    modeCD.size(),
                                                                    cd_ms_ns_lengths,
                                                                    strides ? strides[2].data() : NULL, /*stride*/
                                                                    CDataType,
                                                                    operatorType));
            }

            hiptensorTensorDescriptor_t d_ms_ns;
            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                                &d_ms_ns,
                                                                modeCD.size(),
                                                                cd_ms_ns_lengths,
                                                                strides ? strides[3].data() : NULL, /*stride*/
                                                                DDataType,
                                                                operatorType));


            std::tuple<int32_t, int32_t, int32_t, int32_t> elementBytes(hipDataTypeSize(ADataType),
                                                                        hipDataTypeSize(BDataType),
                                                                        hipDataTypeSize(CDataType),
                                                                        hipDataTypeSize(DDataType));

            auto& resource = getResource();
            resource->resizeStorage(std::tuple(lengths), elementBytes);

            // Initialize matrix data on device
            fillLaunchKernel<ADataType>(resource->deviceA().get(), elementsA);
            fillLaunchKernel<BDataType>(resource->deviceB().get(), elementsB);
            fillLaunchKernel<CDataType>(resource->deviceC().get(), elementsCD);
            fillValLaunchKernel<DDataType>(resource->deviceD().get(),
                                           elementsD,
                                           std::numeric_limits<DDataType>::signaling_NaN());

            resource->copyDeviceToHostAll();

            uint32_t alignmentRequirementA;
            CHECK_HIPTENSOR_ERROR(
                hiptensorGetAlignmentRequirement(handle, resource->deviceA().get(), &a_ms_ks, &alignmentRequirementA));

            uint32_t alignmentRequirementB;
            CHECK_HIPTENSOR_ERROR(
                hiptensorGetAlignmentRequirement(handle, resource->deviceB().get(), &b_ns_ks, &alignmentRequirementB));

            uint32_t alignmentRequirementC;
            CHECK_HIPTENSOR_ERROR(
                hiptensorGetAlignmentRequirement(handle, resource->deviceC().get(), &c_ms_ns, &alignmentRequirementC));

            uint32_t alignmentRequirementD;
            CHECK_HIPTENSOR_ERROR(
                hiptensorGetAlignmentRequirement(handle, resource->deviceD().get(), &d_ms_ns, &alignmentRequirementD));

            CHECK_HIPTENSOR_ERROR(hiptensorInitContractionDescriptor(handle,
                                                                    &desc,
                                                                    &a_ms_ks,
                                                                    modeA.data(),
                                                                    alignmentRequirementA,
                                                                    &b_ns_ks,
                                                                    modeB.data(),
                                                                    alignmentRequirementB,
                                                                    &c_ms_ns,
                                                                    modeC.data(),
                                                                    alignmentRequirementC,
                                                                    &d_ms_ns,
                                                                    modeD.data(),
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

        virtual void RunKernel()
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

            EXPECT_TRUE(beta > 0.0);
                    
            std::cout << "Initializing contraction plan..." << std::endl;

            CHECK_HIPTENSOR_ERROR(hiptensorInitContractionPlan(handle, &plan, &desc, &find, worksize));

            std::cout << "Launching contraction kernel..." << std::endl;

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

            int sizeD = elementsD * sizeof(DDataType);
            auto reference = resource->template allocDevice<DDataType>(sizeD);
            dataInstance->copyData(reference, dataInstance->hostD(), sizeD);

            std::tie(mValidationResult, mMaxRelativeError) = compareEqualLaunchKernel<DDataType>(
                resource->deviceD().get(), reference.get(), elementsD);

            if(mValidationResult == true)
            {
                std::cout << "Validation Successful" << std::endl;
            }
            else
            {
                std::cout << "Validation Failed" << std::endl;
            }

            std::cout << "Max relative error: " << mMaxRelativeError << std::endl;

            bool printElements = false;
            bool storeElements = false;

            DDataType *D;
            if(printElements || storeElements)
            {
                D = resource->template allocHost<DDataType>(sizeD);
                dataInstance->copyData(D, dataInstance->deviceD(), sizeD);
            }

            if(printElements)
            {
                if(elementsA < MAX_ELEMENTS_PRINT_COUNT)
                {
                    std::cout << "Tensor A elements:\n";
                    hiptensorPrintArrayElements(resource->hostA().get(), elementsA);
                    std::cout << std::endl;
                }

                if(elementsB < MAX_ELEMENTS_PRINT_COUNT)
                {
                    std::cout << "Tensor B elements:\n";
                    hiptensorPrintArrayElements(resource->hostB().get(), elementsB);
                    std::cout << std::endl;
                }

                if(elementsCD < MAX_ELEMENTS_PRINT_COUNT)
                {
                    std::cout << "Tensor C elements:\n";
                    hiptensorPrintArrayElements(resource->hostC().get(), elementsCD);
                    std::cout << std::endl;
                }

                if(elementsCD < MAX_ELEMENTS_PRINT_COUNT)
                {
                    std::cout << "Tensor D elements:\n";
                    hiptensorPrintArrayElements(D, elementsCD);
                    std::cout << std::endl;
                }
            }

            if(storeElements)
            {
                std::ofstream tensorA, tensorB, tensorD;
                tensorA.open("tensor_A.txt");
                hiptensorPrintElementsToFile(tensorA, resource->hostA().get(), elementsA, ", ");
                tensorA.close();

                tensorB.open("tensor_B.txt");
                hiptensorPrintElementsToFile(tensorB, resource->hostB().get(), elementsB, ", ");
                tensorB.close();

                tensorC.open("tensor_C.txt");
                hiptensorPrintElementsToFile(tensorC, resource->hostC().get(), elementsCD, ", ");
                tensorC.close();

                tensorD.open("tensor_D_scale_contraction_results.txt");
                hiptensorPrintElementsToFile(tensorD, D, elementsCD, ", ");
                tensorD.close();
            }
        }

        virtual void Warmup() {}

        void TearDown() override 
        {
            CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
        }
    };

} // namespace rocwmma

#endif // HIPTENSOR_CONTRACTION_TEST_IMPL_HPP
