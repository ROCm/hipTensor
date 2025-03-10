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
#include <hiptensor/hiptensor.hpp>

#include "data_types.hpp"
#include "hiptensor_options.hpp"
#include "logger.hpp"
#include "permutation/permutation_cpu_reference.hpp"
#include "permutation_test.hpp"
#include "utils.hpp"

template <typename InputType, typename OutputType, typename ComputeType>
auto elementaryBinaryOpWithCpu(hipDataType inputType, hipDataType outputType, hipDataType typeCompute)
{
    std::vector<int> inMode{'w', 'h', 'c', 'n'};
    std::vector<int> outputMode{'c', 'n', 'h', 'w'};
    int              ninMode = inMode.size();
    int              noutputMode = outputMode.size();

    std::unordered_map<int, int64_t> extent;
    extent['w'] = 3;
    extent['h'] = 2;
    extent['c'] = 4;
    extent['n'] = 5;

    std::vector<int64_t> inExtent;
    for(auto mode : inMode)
    {
        inExtent.push_back(extent[mode]);
    }
    std::vector<int64_t> outputExtent;
    for(auto mode : outputMode)
    {
        outputExtent.push_back(extent[mode]);
    }

    /**********************
     * Allocating data
     **********************/

    size_t inElements = 1;
    for(auto mode : inMode)
    {
        inElements *= extent[mode];
    }
    size_t outputElements = 1;
    for(auto mode : outputMode)
    {
        outputElements *= extent[mode];
    }

    // size_t sizeA = sizeof(InputType) * inElements;
    // size_t sizeB = sizeof(OutputType) * outputElements;

    std::vector<InputType> aArray(inElements);
    std::iota(aArray.begin(), aArray.end(), 0);
    std::vector<InputType> cArray(aArray);
    std::vector<OutputType> dArray(outputElements);
    std::vector<OutputType> referenceArray;

    using hiptensor::HiptensorOptions;
    auto& options = HiptensorOptions::instance();

    if(options->isColMajorStrides())
	{
		referenceArray
			= {  0. ,  19.8,  39.6,  59.4,  79.2,  99. , 118.8, 138.6, 158.4,
				178.2, 198. , 217.8, 237.6, 257.4, 277.2, 297. , 316.8, 336.6,
				356.4, 376.2,   9.9,  29.7,  49.5,  69.3,  89.1, 108.9, 128.7,
				148.5, 168.3, 188.1, 207.9, 227.7, 247.5, 267.3, 287.1, 306.9,
				326.7, 346.5, 366.3, 386.1,   3.3,  23.1,  42.9,  62.7,  82.5,
				102.3, 122.1, 141.9, 161.7, 181.5, 201.3, 221.1, 240.9, 260.7,
				280.5, 300.3, 320.1, 339.9, 359.7, 379.5,  13.2,  33. ,  52.8,
				72.6,  92.4, 112.2, 132. , 151.8, 171.6, 191.4, 211.2, 231. ,
				250.8, 270.6, 290.4, 310.2, 330. , 349.8, 369.6, 389.4,   6.6,
				26.4,  46.2,  66. ,  85.8, 105.6, 125.4, 145.2, 165. , 184.8,
				204.6, 224.4, 244.2, 264. , 283.8, 303.6, 323.4, 343.2, 363. ,
				382.8,  16.5,  36.3,  56.1,  75.9,  95.7, 115.5, 135.3, 155.1,
				174.9, 194.7, 214.5, 234.3, 254.1, 273.9, 293.7, 313.5, 333.3,
				353.1, 372.9, 392.7};
	}
    else
    {
        referenceArray
            = {0.,   84.,   168.,  42.,  126.,  210.,  2.1,  86.1,  170.1, 44.1, 128.1, 212.1,
               4.2,  88.2,  172.2, 46.2, 130.2, 214.2, 6.3,  90.3,  174.3, 48.3, 132.3, 216.3,
               8.4,  92.4,  176.4, 50.4, 134.4, 218.4, 10.5, 94.5,  178.5, 52.5, 136.5, 220.5,
               12.6, 96.6,  180.6, 54.6, 138.6, 222.6, 14.7, 98.7,  182.7, 56.7, 140.7, 224.7,
               16.8, 100.8, 184.8, 58.8, 142.8, 226.8, 18.9, 102.9, 186.9, 60.9, 144.9, 228.9,
               21.,  105.,  189.,  63.,  147.,  231.,  23.1, 107.1, 191.1, 65.1, 149.1, 233.1,
               25.2, 109.2, 193.2, 67.2, 151.2, 235.2, 27.3, 111.3, 195.3, 69.3, 153.3, 237.3,
               29.4, 113.4, 197.4, 71.4, 155.4, 239.4, 31.5, 115.5, 199.5, 73.5, 157.5, 241.5,
               33.6, 117.6, 201.6, 75.6, 159.6, 243.6, 35.7, 119.7, 203.7, 77.7, 161.7, 245.7,
               37.8, 121.8, 205.8, 79.8, 163.8, 247.8, 39.9, 123.9, 207.9, 81.9, 165.9, 249.9};
    }

    const ComputeType alphaValue = 2.1f;
    const ComputeType gammaValue = 1.2f;
    hiptensorHandle_t*     handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
    hiptensorTensorDescriptor_t descA;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
        handle, &descA, ninMode, inExtent.data(), NULL /* stride */, inputType, HIPTENSOR_OP_IDENTITY));
    hiptensorTensorDescriptor_t descC;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
        handle, &descC, ninMode, inExtent.data(), NULL /* stride */, inputType, HIPTENSOR_OP_IDENTITY));

    hiptensorTensorDescriptor_t descD;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
        handle, &descD, noutputMode, outputExtent.data(), NULL /* stride */, outputType, HIPTENSOR_OP_IDENTITY));

	hiptensorElementwiseBianryOpReference(handle,
                                  &alphaValue,
                                  aArray.data(),
                                  &descA,
                                  inMode.data(),
                                  &gammaValue,
                                  cArray.data(),
                                  &descC,
                                  inMode.data(),
                                  dArray.data(),
                                  &descD,
                                  outputMode.data(),
								  HIPTENSOR_OP_ADD,
                                  typeCompute,
                                  0);

    return compareEqual(referenceArray.data(),
                        dArray.data(),
                        dArray.size(),
                        hiptensor::convertToComputeType(typeCompute),
                        0);
}

TEST(ElementaryBinaryOpCpuImplTest, CompareF32ResultWithReference)
{
    typedef float InputType;
    typedef float OutputType;
    typedef float ComputeType;

    hipDataType inputType       = HIP_R_32F;
    hipDataType outputType       = HIP_R_32F;
    hipDataType typeCompute = HIP_R_32F;

    auto [result, maxRelativeError]
        = elementaryBinaryOpWithCpu<InputType, OutputType, ComputeType>(inputType, outputType, typeCompute);
    EXPECT_TRUE(result) << "max_relative_error: " << maxRelativeError;
}

TEST(ElementaryBinaryOpCpuImplTest, CompareF16ResultWithReference)
{
    typedef _Float16 InputType;
    typedef _Float16 OutputType;
    typedef _Float16 ComputeType;

    hipDataType inputType       = HIP_R_16F;
    hipDataType outputType       = HIP_R_16F;
    hipDataType typeCompute = HIP_R_16F;

    auto [result, maxRelativeError]
        = elementaryBinaryOpWithCpu<InputType, OutputType, ComputeType>(inputType, outputType, typeCompute);
    EXPECT_TRUE(result) << "max_relative_error: " << maxRelativeError;
}
