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
#include <algorithm>
#include <fstream>
#include <hiptensor/hiptensor.hpp>
#include <hiptensor/hiptensor_types.hpp>
#include <hiptensor/internal/hiptensor_utility.hpp>
#include <iterator>
#include <numeric>
#include <unordered_map>

#include "common.hpp"
#include "../../library/src/include/types.hpp"
#include "../../library/src/contraction/contraction_cpu_reference.hpp"

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

    typedef hipFloatComplex ADataType;
    typedef hipFloatComplex BDataType;
    typedef hipFloatComplex DDataType;
    typedef float floatTypeCompute;

    hipDataType            typeA       = HIP_C_32F;
    hipDataType            typeB       = HIP_C_32F;
    hipDataType            typeD       = HIP_C_32F;
    hiptensorComputeType_t typeCompute = HIPTENSOR_COMPUTE_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.0f;

    /**********************
   * Computing: C_{m,n,u,v} = A_{m,n,h,k} B_{h,k,u,v}
   **********************/

    std::vector<int> modeD{'m', 'n', 'u', 'v'};
    std::vector<int> modeA{'m', 'n', 'h', 'k'};
    std::vector<int> modeB{'u', 'v', 'h', 'k'};

    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeD = modeD.size();

    std::unordered_map<int, int64_t> extent;
    std::unordered_map<int, int64_t> stride;
    
    extent['m'] = 5;
    extent['n'] = 6;
    extent['u'] = 3;
    extent['v'] = 4;
    extent['h'] = 3;
    extent['k'] = 4;

    stride['m'] = 72;
    stride['n'] = 12;
    stride['u'] = 4;
    stride['v'] = 1;
    stride['h'] = 4;
    stride['k'] = 1;

    std::vector<size_t> d_ms_ns_lengths;
    std::vector<size_t> d_ms_ns_strides;
    for(auto mode : modeD)
    {
        d_ms_ns_lengths.push_back(extent[mode]);
        d_ms_ns_strides.push_back(stride[mode]);
    }

    std::vector<size_t> a_ms_ks_lengths;
    std::vector<size_t> a_ms_ks_strides;
    for(auto mode : modeA)
    {
        a_ms_ks_lengths.push_back(extent[mode]);
        a_ms_ks_strides.push_back(stride[mode]);
    }

    std::vector<size_t> b_ns_ks_lengths;
    std::vector<size_t> b_ns_ks_strides;
    for(auto mode : modeB)
    {
        b_ns_ks_lengths.push_back(extent[mode]);
        b_ns_ks_strides.push_back(stride[mode]);
    }

    /**********************
   * Allocating data
   **********************/
    std::cout << "Initializing host data..." << std::endl;

    size_t elementsA = std::accumulate(
        a_ms_ks_lengths.begin(), a_ms_ks_lengths.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsB = std::accumulate(
        b_ns_ks_lengths.begin(), b_ns_ks_lengths.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsD = std::accumulate(
        d_ms_ns_lengths.begin(), d_ms_ns_lengths.end(), size_t{1}, std::multiplies<size_t>());

    size_t sizeA = sizeof(ADataType) * elementsA;
    size_t sizeB = sizeof(BDataType) * elementsB;
    size_t sizeD = sizeof(DDataType) * elementsD;

    ADataType* A = (ADataType*)malloc(sizeA);
    BDataType* B = (BDataType*)malloc(sizeB);
    DDataType* D = (DDataType*)malloc(sizeD);

    /*******************
   * Initialize data
   *******************/
    for(int64_t i = 0; i < elementsA; i++)
    {
        A[i] = make_hipFloatComplex(((float(std::rand())) / float(RAND_MAX) - 0.5) * 100,
                                    ((float(std::rand())) / float(RAND_MAX) - 0.5) * 100);
    }

    for(int64_t i = 0; i < elementsB; i++)
    {
        B[i] = make_hipFloatComplex(((float(std::rand())) / float(RAND_MAX) - 0.5) * 100,
                                    ((float(std::rand())) / float(RAND_MAX) - 0.5) * 100);
    }

    for(int64_t i = 0; i < elementsD; i++)
    {
        D[i] = make_hipFloatComplex((float)std::numeric_limits<float>::signaling_NaN(),
                                    (float)std::numeric_limits<float>::signaling_NaN());
    }

    CHECK_HIPTENSOR_ERROR(hiptensorContractionReference((void*)&alpha,
                                                               A,
                                                               B,
                                                               0,
                                                               nullptr,
                                                               D,
                                                               a_ms_ks_lengths,
                                                               a_ms_ks_strides,
                                                               b_ns_ks_lengths,
                                                               b_ns_ks_strides,
                                                               d_ms_ns_lengths,
                                                               d_ms_ns_strides,
                                                               d_ms_ns_lengths,
                                                               d_ms_ns_strides,
                                                               typeA,
                                                               typeB,
                                                               hiptensor::NONE_TYPE,
                                                               typeD,
                                                               0));

#if !NDEBUG
    bool printElements = false;
    bool storeElements = false;

    if(printElements)
    {
        if(elementsA < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor A elements:\n";
            hiptensorPrintArrayElements(std::cout, A, elementsA);
            std::cout << std::endl;
        }

        if(elementsB < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor B elements:\n";
            hiptensorPrintArrayElements(std::cout, B, elementsB);
            std::cout << std::endl;
        }

        if(elementsD < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor D elements:\n";
            hiptensorPrintArrayElements(std::cout, D, elementsD);
            std::cout << std::endl;
        }
    }

    if(storeElements)
    {
        std::ofstream tensorA, tensorB, tensorD;
        tensorA.open("tensor_A.txt");
        hiptensorPrintElementsToFile(tensorA, A, elementsA, ", ");
        tensorA.close();

        tensorB.open("tensor_B.txt");
        hiptensorPrintElementsToFile(tensorB, B, elementsB, ", ");
        tensorB.close();

        tensorD.open("tensor_D_scale_contraction_results.txt");
        hiptensorPrintElementsToFile(tensorD, D, elementsD, ", ");
        tensorD.close();
    }

#endif

    HIPTENSOR_FREE_HOST(A);
    HIPTENSOR_FREE_HOST(B);
    HIPTENSOR_FREE_HOST(D);

    std::cout << "Finished!" << std::endl;

    return 0;
}
