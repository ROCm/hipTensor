/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2023 Advanced Micro Devices, Inc.
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

#include <algorithm>
#include <fstream>
#include <iterator>
#include <numeric>
#include <unordered_map>

// hiptensor includes
#include <hiptensor/hiptensor.hpp>
#include <hiptensor/hiptensor_types.hpp>
#include <hiptensor/internal/hiptensor_utility.hpp>

#if !NDEBUG
#include "common.hpp"
#endif

#define MAX_ELEMENTS_PRINT_COUNT 512

int main(int argc, char* argv[])
{
    typedef float ADataType;
    typedef float BDataType;
    typedef float DDataType;
    typedef float floatTypeCompute;

    hipDataType            typeA       = HIP_R_32F;
    hipDataType            typeB       = HIP_R_32F;
    hipDataType            typeC       = hipDataType(-1);
    hipDataType            typeD       = HIP_R_32F;
    hiptensorComputeType_t typeCompute = HIPTENSOR_COMPUTE_32F;

    floatTypeCompute alpha = (floatTypeCompute)2.0f;

#if !NDEBUG
    std::cout << "RAND_MAX value is " << RAND_MAX << std::endl;
#endif
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

    extent['m'] = 5;
    extent['n'] = 6;
    extent['u'] = 3;
    extent['v'] = 4;
    extent['h'] = 3;
    extent['k'] = 4;

    std::vector<int64_t> d_ms_ns_lengths;
    for(auto mode : modeD)
    {
        d_ms_ns_lengths.push_back(extent[mode]);
    }
    std::vector<int64_t> a_ms_ks_lengths;
    for(auto mode : modeA)
    {
        a_ms_ks_lengths.push_back(extent[mode]);
    }
    std::vector<int64_t> b_ks_ns_lengths;
    for(auto mode : modeB)
    {
        b_ks_ns_lengths.push_back(extent[mode]);
    }

    hiptensorHandle_t* handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

    /********************************************
   * Intialise Tensors with the input lengths *
   ********************************************/
    hiptensorTensorDescriptor_t a_ms_ks;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                        &a_ms_ks,
                                                        nmodeA,
                                                        a_ms_ks_lengths.data(),
                                                        NULL, /*stride*/
                                                        typeA,
                                                        HIPTENSOR_OP_IDENTITY));

    hiptensorTensorDescriptor_t b_ks_ns;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                        &b_ks_ns,
                                                        nmodeB,
                                                        b_ks_ns_lengths.data(),
                                                        NULL, /*stride*/
                                                        typeB,
                                                        HIPTENSOR_OP_IDENTITY));

    hiptensorTensorDescriptor_t d_ms_ns;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                        &d_ms_ns,
                                                        nmodeD,
                                                        d_ms_ns_lengths.data(),
                                                        NULL, /*stride*/
                                                        typeD,
                                                        HIPTENSOR_OP_IDENTITY));

    /**********************
   * Allocating data
   **********************/
    std::cout << "Initializing host data..." << std::endl;

    size_t elementsA = std::accumulate(
        a_ms_ks_lengths.begin(), a_ms_ks_lengths.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsB = std::accumulate(
        b_ks_ns_lengths.begin(), b_ks_ns_lengths.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsD = std::accumulate(
        d_ms_ns_lengths.begin(), d_ms_ns_lengths.end(), size_t{1}, std::multiplies<size_t>());

    size_t sizeA = sizeof(ADataType) * elementsA;
    size_t sizeB = sizeof(BDataType) * elementsB;
    size_t sizeD = sizeof(DDataType) * elementsD;

    ADataType* A = (ADataType*)malloc(sizeA);
    BDataType* B = (BDataType*)malloc(sizeB);
    DDataType* D = (DDataType*)malloc(sizeD);
#if !NDEBUG
    DDataType* D_host = (DDataType*)malloc(sizeD);
#endif

    void *A_d, *B_d, *D_d;

    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&A_d), sizeA));
    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&B_d), sizeB));
    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&D_d), sizeD));

    /*******************
   * Initialize data
   *******************/
    for(int64_t i = 0; i < elementsA; i++)
    {
        A[i] = ((float(std::rand())) / float(RAND_MAX) - 0.5) * 100;
    }
    for(int64_t i = 0; i < elementsB; i++)
    {
        B[i] = ((float(std::rand())) / float(RAND_MAX) - 0.5) * 100;
    }
    for(int64_t i = 0; i < elementsD; i++)
    {
        D[i] = std::numeric_limits<DDataType>::signaling_NaN();
#if !NDEBUG
        D_host[i] = std::numeric_limits<DDataType>::signaling_NaN();
#endif
    }

    /********************************************
   * Transfer the Host Tensor to Device Memory *
   ********************************************/
    std::cout << "Initializing device data..." << std::endl;

    CHECK_HIP_ERROR(hipMemcpy(A_d, static_cast<const void*>(A), sizeA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(B_d, static_cast<const void*>(B), sizeB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(D_d, 0, sizeD));

    /************************************************
   * Retrieve the memory alignment for each tensor
   ************************************************/
    uint32_t alignmentRequirementA;
    CHECK_HIPTENSOR_ERROR(
        hiptensorGetAlignmentRequirement(handle, A_d, &a_ms_ks, &alignmentRequirementA));

    uint32_t alignmentRequirementB;
    CHECK_HIPTENSOR_ERROR(
        hiptensorGetAlignmentRequirement(handle, B_d, &b_ks_ns, &alignmentRequirementB));

    uint32_t alignmentRequirementD;
    CHECK_HIPTENSOR_ERROR(
        hiptensorGetAlignmentRequirement(handle, D_d, &d_ms_ns, &alignmentRequirementD));

    /*******************************
   * Create Contraction Descriptor
   *******************************/

    std::cout << "a_ms_ks: " << a_ms_ks << std::endl;
    std::cout << "b_ks_ns: " << b_ks_ns << std::endl;
    std::cout << "d_ms_ns: " << d_ms_ns << std::endl;

    hiptensorContractionDescriptor_t desc;
    CHECK_HIPTENSOR_ERROR(hiptensorInitContractionDescriptor(handle,
                                                             &desc,
                                                             &a_ms_ks,
                                                             modeA.data(),
                                                             alignmentRequirementA,
                                                             &b_ks_ns,
                                                             modeB.data(),
                                                             alignmentRequirementB,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             &d_ms_ns,
                                                             modeD.data(),
                                                             alignmentRequirementD,
                                                             typeCompute));
    /**************************
   * Set the algorithm to use
   ***************************/

    hiptensorContractionFind_t find;
    CHECK_HIPTENSOR_ERROR(hiptensorInitContractionFind(handle, &find, HIPTENSOR_ALGO_DEFAULT));

    /**********************
   * Query workspace
   **********************/

    uint64_t worksize = 0;
    CHECK_HIPTENSOR_ERROR(hiptensorContractionGetWorkspaceSize(
        handle, &desc, &find, HIPTENSOR_WORKSPACE_RECOMMENDED, &worksize));

    void* workspace = nullptr;

    if(worksize > 0)
    {
        CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&workspace), worksize));
    }

    /**************************
   * Create Contraction Plan
   **************************/
    std::cout << "Finding contraction solution..." << std::endl;

    hiptensorContractionPlan_t plan;
    CHECK_HIPTENSOR_ERROR(hiptensorInitContractionPlan(handle, &plan, &desc, &find, worksize));

    std::cout << "Launching contraction kernel..." << std::endl;

    CHECK_HIPTENSOR_ERROR(hiptensorContraction(handle,
                                               &plan,
                                               (void*)&alpha,
                                               A_d,
                                               B_d,
                                               nullptr,
                                               nullptr,
                                               D_d,
                                               workspace,
                                               worksize,
                                               0 /* stream */));

#if !NDEBUG
    CHECK_HIP_ERROR(hipMemcpy(D, D_d, sizeD, hipMemcpyDeviceToHost));

    std::ofstream tensorA, tensorB, tensorD;
    bool          printElements = false;

    if(printElements)
    {
        if(elementsA < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor A elements:\n";
            hiptensorPrintArrayElements(A, elementsA);
            std::cout << std::endl;
        }
        tensorA.open("tensor_A.txt");
        hiptensorPrintElementsToFile(tensorA, A, elementsA, ", ");
        std::cout << std::endl;
        tensorA.close();
        if(elementsB < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor B elements:\n";
            hiptensorPrintArrayElements(B, elementsB);
            std::cout << std::endl;
        }
        tensorB.open("tensor_B.txt");
        hiptensorPrintElementsToFile(tensorB, B, elementsB, ", ");
        std::cout << std::endl;
        tensorB.close();
        if(elementsD < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor D elements:\n";
            hiptensorPrintArrayElements(D, elementsD);
            std::cout << std::endl;
        }
        tensorD.open("tensor_D_scale_contraction_results.txt");
        hiptensorPrintElementsToFile(tensorD, D, elementsD, ", ");
        std::cout << std::endl;
        tensorD.close();
    }

    CHECK_HIPTENSOR_ERROR(hiptensorContractionReference((void*)&alpha,
                                                        A,
                                                        B,
                                                        nullptr,
                                                        nullptr,
                                                        D_host,
                                                        a_ms_ks.mLengths,
                                                        a_ms_ks.mStrides,
                                                        b_ks_ns.mLengths,
                                                        b_ks_ns.mStrides,
                                                        d_ms_ns.mLengths,
                                                        d_ms_ns.mStrides,
                                                        d_ms_ns.mLengths,
                                                        d_ms_ns.mStrides,
                                                        typeA,
                                                        typeB,
                                                        typeC,
                                                        typeD,
                                                        workspace));

    bool   mValidationResult = false;
    double mMaxRelativeError;

    std::tie(mValidationResult, mMaxRelativeError) = compareEqual<DDataType>(D, D_host, elementsD);

    if(mValidationResult == true)
    {
        std::cout << "Validation Successful" << std::endl;
    }
    else
    {
        std::cout << "Validation Failed" << std::endl;
    }

    std::cout << "Max relative error: " << mMaxRelativeError << std::endl;

    if(D_host)
    {
        free(D_host);
    }
#endif

    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));

    if(A)
    {
        free(A);
    }

    if(B)
    {
        free(B);
    }

    if(D)
    {
        free(D);
    }

    if(A_d)
    {
        CHECK_HIP_ERROR(hipFree(A_d));
    }

    if(B_d)
    {
        CHECK_HIP_ERROR(hipFree(B_d));
    }

    if(D_d)
    {
        CHECK_HIP_ERROR(hipFree(D_d));
    }

    if(workspace)
    {
        CHECK_HIP_ERROR(hipFree(workspace));
    }

    return 0;
}
