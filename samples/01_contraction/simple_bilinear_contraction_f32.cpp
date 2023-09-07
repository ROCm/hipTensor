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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/
#include <algorithm>
#include <fstream>
#include <hiptensor/hiptensor.hpp>
#include <hiptensor/hiptensor_types.hpp>
#include <hiptensor/internal/hiptensor_utility.hpp>
#include <iostream>
#include <iterator>
#include <numeric>
#include <unordered_map>

#define MAX_ELEMENTS_PRINT_COUNT 512

#define HIPTENSOR_FREE_DEVICE(ptr)     \
    if(ptr != nullptr)                 \
    {                                  \
        CHECK_HIP_ERROR(hipFree(ptr)); \
    }

#define HIPTENSOR_FREE_HOST(ptr) \
    if(ptr != nullptr)           \
    {                            \
        free(ptr);               \
    }

inline bool isF32Supported()
{
    hipDevice_t     mHandle;
    hipDeviceProp_t mProps;

    CHECK_HIP_ERROR(hipGetDevice(&mHandle));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&mProps, mHandle));

    std::string deviceName(mProps.gcnArchName);

    return (deviceName.find("gfx908") != std::string::npos)
            || (deviceName.find("gfx90a") != std::string::npos)
            || (deviceName.find("gfx940") != std::string::npos);
}

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

    typedef float ADataType;
    typedef float BDataType;
    typedef float CDataType;
    typedef float floatTypeCompute;

    hipDataType            typeA       = HIP_R_32F;
    hipDataType            typeB       = HIP_R_32F;
    hipDataType            typeC       = HIP_R_32F;
    hiptensorComputeType_t typeCompute = HIPTENSOR_COMPUTE_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)1.0f;

    /**********************
   * Computing: C_{m,n,u,v} = alpha * A_{m,n,h,k} B_{u,v,h,k} + beta *
   *C_{m,n,u,v}
   **********************/

    std::vector<int> modeC{'m', 'n', 'u', 'v'};
    std::vector<int> modeA{'m', 'n', 'h', 'k'};
    std::vector<int> modeB{'u', 'v', 'h', 'k'};

    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;

    extent['m'] = 5;
    extent['n'] = 6;
    extent['u'] = 3;
    extent['v'] = 4;
    extent['h'] = 3;
    extent['k'] = 4;

    std::vector<int64_t> c_ms_ns_lengths;
    for(auto mode : modeC)
    {
        c_ms_ns_lengths.push_back(extent[mode]);
    }

    std::vector<int64_t> a_ms_ks_lengths;
    for(auto mode : modeA)
    {
        a_ms_ks_lengths.push_back(extent[mode]);
    }

    std::vector<int64_t> b_ns_ks_lengths;
    for(auto mode : modeB)
    {
        b_ns_ks_lengths.push_back(extent[mode]);
    }

    hiptensorHandle_t* handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

    CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_PERF_TRACE));

    /********************************************
   * Initialize tensors with the input lengths *
   ********************************************/
    hiptensorTensorDescriptor_t a_ms_ks;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                        &a_ms_ks,
                                                        nmodeA,
                                                        a_ms_ks_lengths.data(),
                                                        NULL, /*stride*/
                                                        typeA,
                                                        HIPTENSOR_OP_IDENTITY));

    hiptensorTensorDescriptor_t b_ns_ks;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                        &b_ns_ks,
                                                        nmodeB,
                                                        b_ns_ks_lengths.data(),
                                                        NULL, /*stride*/
                                                        typeB,
                                                        HIPTENSOR_OP_IDENTITY));

    hiptensorTensorDescriptor_t c_ms_ns;
    CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                        &c_ms_ns,
                                                        nmodeC,
                                                        c_ms_ns_lengths.data(),
                                                        NULL, /*stride*/
                                                        typeC,
                                                        HIPTENSOR_OP_IDENTITY));

    /**********************
   * Allocating data
   **********************/
    std::cout << "Initializing host data..." << std::endl;

    size_t elementsA = std::accumulate(
        a_ms_ks_lengths.begin(), a_ms_ks_lengths.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsB = std::accumulate(
        b_ns_ks_lengths.begin(), b_ns_ks_lengths.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsC = std::accumulate(
        c_ms_ns_lengths.begin(), c_ms_ns_lengths.end(), size_t{1}, std::multiplies<size_t>());

    size_t sizeA = sizeof(ADataType) * elementsA;
    size_t sizeB = sizeof(BDataType) * elementsB;
    size_t sizeC = sizeof(CDataType) * elementsC;

    ADataType* A = (ADataType*)malloc(sizeA);
    BDataType* B = (BDataType*)malloc(sizeB);
    CDataType* C = (CDataType*)malloc(sizeC);

    void *A_d, *B_d, *C_d;

    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&A_d), sizeA));
    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&B_d), sizeB));
    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&C_d), sizeC));

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

    for(int64_t i = 0; i < elementsC; i++)
    {
        C[i] = ((float(std::rand())) / float(RAND_MAX) - 0.5) * 100;
    }

    /********************************************
   * Transfer the Host Tensor to Device Memory *
   ********************************************/
    std::cout << "Initializing device data..." << std::endl;

    CHECK_HIP_ERROR(hipMemcpy(A_d, static_cast<const void*>(A), sizeA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(B_d, static_cast<const void*>(B), sizeB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(C_d, static_cast<const void*>(C), sizeC, hipMemcpyHostToDevice));

    /************************************************
   * Retrieve the memory alignment for each tensor
   ************************************************/

    uint32_t alignmentRequirementA;
    CHECK_HIPTENSOR_ERROR(
        hiptensorGetAlignmentRequirement(handle, A_d, &a_ms_ks, &alignmentRequirementA));

    uint32_t alignmentRequirementB;
    CHECK_HIPTENSOR_ERROR(
        hiptensorGetAlignmentRequirement(handle, B_d, &b_ns_ks, &alignmentRequirementB));

    uint32_t alignmentRequirementC;
    CHECK_HIPTENSOR_ERROR(
        hiptensorGetAlignmentRequirement(handle, C_d, &c_ms_ns, &alignmentRequirementC));

    /*******************************
   * Create Contraction Descriptor
   *******************************/

    std::cout << "a_ms_ks: " << a_ms_ks << std::endl;
    std::cout << "b_ns_ks: " << b_ns_ks << std::endl;
    std::cout << "c_ms_ns: " << c_ms_ns << std::endl;

    hiptensorContractionDescriptor_t desc;
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
                                                             &c_ms_ns,
                                                             modeC.data(),
                                                             alignmentRequirementC,
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
    std::cout << "Initializing contraction plan..." << std::endl;

    hiptensorContractionPlan_t plan;
    CHECK_HIPTENSOR_ERROR(hiptensorInitContractionPlan(handle, &plan, &desc, &find, worksize));

    std::cout << "Launching contraction kernel..." << std::endl;

    CHECK_HIPTENSOR_ERROR(hiptensorContraction(handle,
                                               &plan,
                                               (void*)&alpha,
                                               A_d,
                                               B_d,
                                               (void*)&beta,
                                               C_d,
                                               C_d,
                                               workspace,
                                               worksize,
                                               0 /* stream */));

#if !NDEBUG
    bool printElements = false;
    bool storeElements = false;

    if(printElements || storeElements)
    {
        CHECK_HIP_ERROR(hipMemcpy(C, C_d, sizeC, hipMemcpyDeviceToHost));
    }

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

        if(elementsC < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor C elements:\n";
            hiptensorPrintArrayElements(std::cout, C, elementsC);
            std::cout << std::endl;
        }
    }

    if(storeElements)
    {
        std::ofstream tensorA, tensorB, tensorC;
        tensorA.open("tensor_A.txt");
        hiptensorPrintElementsToFile(tensorA, A, elementsA, ", ");
        tensorA.close();

        tensorB.open("tensor_B.txt");
        hiptensorPrintElementsToFile(tensorB, B, elementsB, ", ");
        tensorB.close();

        tensorC.open("tensor_C_scale_contraction_results.txt");
        hiptensorPrintElementsToFile(tensorC, C, elementsC, ", ");
        tensorC.close();
    }

#endif

    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));

    HIPTENSOR_FREE_HOST(A);
    HIPTENSOR_FREE_HOST(B);
    HIPTENSOR_FREE_HOST(C);

    HIPTENSOR_FREE_DEVICE(A_d);
    HIPTENSOR_FREE_DEVICE(B_d);
    HIPTENSOR_FREE_DEVICE(C_d);
    HIPTENSOR_FREE_DEVICE(workspace);

    std::cout << "Finished!" << std::endl;

    return 0;
}
