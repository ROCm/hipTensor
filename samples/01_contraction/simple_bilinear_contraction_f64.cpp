/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/
#include "simple_bilinear_contraction.hpp"

int main(int argc, char* argv[])
{
    /***************************************
   * Check device support                 *
   **************************************/
    if(!isF32Supported())
    {
        std::cout << "unsupported host device" << std::endl;
        exit(EXIT_FAILURE);
    }

    typedef double ADataType;
    typedef double BDataType;
    typedef double CDataType;
    typedef double floatTypeCompute;

    constexpr hipDataType            typeA       = HIP_R_64F;
    constexpr hipDataType            typeB       = HIP_R_64F;
    constexpr hipDataType            typeC       = HIP_R_64F;
    constexpr hiptensorComputeType_t typeCompute = HIPTENSOR_COMPUTE_64F;

    return bilinearContractionSample<ADataType,
                                     BDataType,
                                     CDataType,
                                     floatTypeCompute,
                                     typeA,
                                     typeB,
                                     typeC,
                                     typeCompute>();
}
