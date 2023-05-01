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
#include <iostream>
#include <iterator>
#include <numeric>
#include <unordered_map>

// hiptensor includes
#include <hiptensor/hiptensor.hpp>
#include <hiptensor/hiptensor_types.hpp>
#include <hiptensor/internal/hiptensor_utility.hpp>

#define MAX_ELEMENTS_PRINT_COUNT 512

int main(int argc, char* argv[])
{
    typedef double ADataType;
    typedef double BDataType;
    typedef double CDataType;
    typedef double doubleTypeCompute;

    hiptensorDataType_t    typeA       = HIPTENSOR_R_32F;
    hiptensorDataType_t    typeB       = HIPTENSOR_R_32F;
    hiptensorDataType_t    typeC       = HIPTENSOR_R_32F;
    hiptensorComputeType_t typeCompute = HIPTENSOR_COMPUTE_32F;

    doubleTypeCompute alpha = (doubleTypeCompute)1.1f;
    doubleTypeCompute beta  = (doubleTypeCompute)1.0f;

#if !NDEBUG
    std::cout << "RAND_MAX value is " << RAND_MAX << std::endl;
#endif

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
        c_ms_ns_lengths.push_back(extent[mode]);
    std::vector<int64_t> a_ms_ks_lengths;
    for(auto mode : modeA)
        a_ms_ks_lengths.push_back(extent[mode]);
    std::vector<int64_t> b_ks_ns_lengths;
    for(auto mode : modeB)
        b_ks_ns_lengths.push_back(extent[mode]);

    hiptensorHandle_t** handle;
    hiptensorCreate(handle);

    /********************************************
   * Intialise Tensors with the input lengths *
   ********************************************/
    hiptensorTensorDescriptor_t a_ms_ks;

    hiptensorInitTensorDescriptor(handle,
                                  &a_ms_ks,
                                  nmodeA,
                                  a_ms_ks_lengths.data(),
                                  NULL, /*stride*/
                                  typeA,
                                  HIPTENSOR_OP_IDENTITY);
#if !NDEBUG
    std::cout << "a_ms_ks: " << a_ms_ks << std::endl;
#endif

    hiptensorTensorDescriptor_t b_ks_ns;
    hiptensorInitTensorDescriptor(handle,
                                  &b_ks_ns,
                                  nmodeB,
                                  b_ks_ns_lengths.data(),
                                  NULL, /*stride*/
                                  typeB,
                                  HIPTENSOR_OP_IDENTITY);

#if !NDEBUG
    std::cout << "b_ks_ns: " << b_ks_ns << std::endl;
#endif

    hiptensorTensorDescriptor_t c_ms_ns;
    hiptensorInitTensorDescriptor(handle,
                                  &c_ms_ns,
                                  nmodeC,
                                  c_ms_ns_lengths.data(),
                                  NULL, /*stride*/
                                  typeC,
                                  HIPTENSOR_OP_IDENTITY);

#if !NDEBUG
    std::cout << "c_ms_ns: " << c_ms_ns << std::endl;
#endif

    /**********************
   * Allocating data
   **********************/

    size_t elementsA = a_ms_ks.hiptensorGetElementSpace();
    size_t elementsB = b_ks_ns.hiptensorGetElementSpace();
    size_t elementsC = c_ms_ns.hiptensorGetElementSpace();

    size_t sizeA = sizeof(ADataType) * elementsA;
    size_t sizeB = sizeof(BDataType) * elementsB;
    size_t sizeC = sizeof(CDataType) * elementsC;

    ADataType* A = (ADataType*)malloc(sizeA);
    BDataType* B = (BDataType*)malloc(sizeB);
    CDataType* C = (CDataType*)malloc(sizeC);

    void *A_d, *B_d, *C_d;

    hip_check_error(hipMalloc(static_cast<void**>(&A_d), sizeA));
    hip_check_error(hipMalloc(static_cast<void**>(&B_d), sizeB));
    hip_check_error(hipMalloc(static_cast<void**>(&C_d), sizeC));

    /*******************
   * Initialize data
   *******************/
    for(int64_t i = 0; i < elementsA; i++)
        A[i] = ((float(std::rand())) / float(RAND_MAX) - 0.5) * 100;
    for(int64_t i = 0; i < elementsB; i++)
        B[i] = ((float(std::rand())) / float(RAND_MAX) - 0.5) * 100;
    for(int64_t i = 0; i < elementsC; i++)
        C[i] = ((float(std::rand())) / float(RAND_MAX) - 0.5) * 100;

    /********************************************
   * Transfer the Host Tensor to Device Memory *
   ********************************************/
    hip_check_error(hipMemcpy(A_d, static_cast<const void*>(A), sizeA, hipMemcpyHostToDevice));
    hip_check_error(hipMemcpy(B_d, static_cast<const void*>(B), sizeB, hipMemcpyHostToDevice));
    hip_check_error(hipMemcpy(C_d, static_cast<const void*>(C), sizeC, hipMemcpyHostToDevice));

    /************************************************
   * Retrieve the memory alignment for each tensor
   ************************************************/

    uint32_t alignmentRequirementA;
    hiptensorGetAlignmentRequirement(handle, A_d, &a_ms_ks, &alignmentRequirementA);

    uint32_t alignmentRequirementB;
    hiptensorGetAlignmentRequirement(handle, B_d, &b_ks_ns, &alignmentRequirementB);

    uint32_t alignmentRequirementC;
    hiptensorGetAlignmentRequirement(handle, C_d, &c_ms_ns, &alignmentRequirementC);

    /*******************************
   * Create Contraction Descriptor
   *******************************/

    hiptensorContractionDescriptor_t desc;
    hiptensorInitContractionDescriptor(handle,
                                       &desc,
                                       &a_ms_ks,
                                       modeA.data(),
                                       alignmentRequirementA,
                                       &b_ks_ns,
                                       modeB.data(),
                                       alignmentRequirementB,
                                       &c_ms_ns,
                                       modeC.data(),
                                       alignmentRequirementC,
                                       &c_ms_ns,
                                       modeC.data(),
                                       alignmentRequirementC,
                                       typeCompute);
    /**************************
   * Set the algorithm to use
   ***************************/

    hiptensorContractionFind_t find;
    hiptensorInitContractionFind(handle, &find, HIPTENSOR_ALGO_DEFAULT);

    /**********************
   * Query workspace
   **********************/

    uint64_t worksize = 0;
    hiptensorContractionGetWorkspaceSize(
        handle, &desc, &find, HIPTENSOR_WORKSPACE_RECOMMENDED, &worksize);
    void* work = nullptr;

    /**************************
   * Create Contraction Plan
   **************************/

    hiptensorContractionPlan_t plan;
    hiptensorInitContractionPlan(handle, &plan, &desc, &find, worksize);

    hiptensorContraction(handle,
                         &plan,
                         (void*)&alpha,
                         A_d,
                         B_d,
                         (void*)&beta,
                         C_d,
                         C_d,
                         work,
                         worksize,
                         0 /* stream */);

    plan.hiptensorPrintContractionMetrics();
    hip_check_error(hipMemcpy(C, C_d, sizeC, hipMemcpyDeviceToHost));

#if !NDEBUG
    std::ofstream tensorA, tensorB, tensorC;
    if(elementsA < MAX_ELEMENTS_PRINT_COUNT)
    {
        std::cout << "Tensor A elements:\n";
        hiptensorPrintArrayElements(A, elementsA);
        std::cout << std::endl;
    }
    tensorA.open("tensor_A.txt");
    hiptensorPrintElementsToFile(tensorA, A, elementsA, ',');
    std::cout << std::endl;
    tensorA.close();
    if(elementsB < MAX_ELEMENTS_PRINT_COUNT)
    {
        std::cout << "Tensor B elements:\n";
        hiptensorPrintArrayElements(B, elementsB);
        std::cout << std::endl;
    }
    tensorB.open("tensor_B.txt");
    hiptensorPrintElementsToFile(tensorB, B, elementsB, ',');
    std::cout << std::endl;
    tensorB.close();
    if(elementsC < MAX_ELEMENTS_PRINT_COUNT)
    {
        std::cout << "Tensor C elements:\n";
        hiptensorPrintArrayElements(C, elementsC);
        std::cout << std::endl;
    }
    tensorC.open("tensor_C_bilinear_contraction_results.txt");
    hiptensorPrintElementsToFile(tensorC, C, elementsC, ',');
    std::cout << std::endl;
    tensorC.close();
#endif

    if(A)
        free(A);
    if(B)
        free(B);
    if(C)
        free(C);
    if(A_d)
        hip_check_error(hipFree(A_d));
    if(B_d)
        hip_check_error(hipFree(B_d));
    if(C_d)
        hip_check_error(hipFree(C_d));
    return 0;
}
