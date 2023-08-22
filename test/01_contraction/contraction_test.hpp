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

#ifndef HIPTENSOR_CONTRACTION_TEST_HPP
#define HIPTENSOR_CONTRACTION_TEST_HPP

#include <iostream>
#include <sstream>
#include <string>

#include "contraction_resource.hpp"
#include "hip_device.hpp"

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
    protected: // Types
        // Shared access to Contraction storage
        using DataStorage = ContractionResource;

        // Using Hip device backend
        using deviceInfo = HipDevice;

        // Kernel run checks.
        // True = run test
        // False = skip test
        virtual bool checkDevice() const;
        virtual bool checkSizes() const;
        virtual bool checkQuirks() const;

        // Reset all members to default values
        virtual void reset();

    protected:
        hiptensorHandle_t* handle;
        hiptensorContractionPlan_t plan;
        hiptensorContractionDescriptor_t desc;
        hiptensorContractionFind_t find;
        uint64_t worksize;
        void* workspace = nullptr;

        //hiptensorTensorDescriptor_t a_ms_ks, b_ns_ks, c_ms_ns;

        // Execution flow control
        uint32_t mRepeats;
        bool     mRunFlag          = true;
        bool     mValidationResult = false;
        double   mMaxRelativeError;
    };

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_TEST_HPP
