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

#define MAX_ELEMENTS_PRINT_COUNT 512

int main(int argc, char* argv[])
{
    typedef double ADataType;
    typedef double BDataType;
    typedef double DDataType;
    typedef double doubleTypeCompute;

    hipDataType            typeA       = HIP_R_64F;
    hipDataType            typeB       = HIP_R_64F;
    hipDataType            typeD       = HIP_R_64F;
    hiptensorComputeType_t typeCompute = HIPTENSOR_COMPUTE_64F;

    doubleTypeCompute alpha = (doubleTypeCompute)1.0;
    doubleTypeCompute beta  = (doubleTypeCompute)0.0;

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
        d_ms_ns_lengths.push_back(extent[mode]);
    std::vector<int64_t> a_ms_ks_lengths;
    for(auto mode : modeA)
        a_ms_ks_lengths.push_back(extent[mode]);
    std::vector<int64_t> b_ks_ns_lengths;
    for(auto mode : modeB)
        b_ks_ns_lengths.push_back(extent[mode]);

    hiptensorHandle_t* handle;
    hiptensorCreate(&handle);

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

    hiptensorTensorDescriptor_t d_ms_ns;
    hiptensorInitTensorDescriptor(&handle,
                                  &d_ms_ns,
                                  nmodeD,
                                  d_ms_ns_lengths.data(),
                                  NULL, /*stride*/
                                  typeD,
                                  HIPTENSOR_OP_IDENTITY);

#if !NDEBUG
    std::cout << "d_ms_ns: " << c_ms_ns << std::endl;
#endif

    /**********************
   * Allocating data
   **********************/

    size_t elementsA = a_ms_ks.hiptensorGetElementSpace();
    size_t elementsB = b_ks_ns.hiptensorGetElementSpace();
    size_t elementsD = d_ms_ns.hiptensorGetElementSpace();

    size_t sizeA = sizeof(ADataType) * elementsA;
    size_t sizeB = sizeof(BDataType) * elementsB;
    size_t sizeD = sizeof(DDataType) * elementsD;

    ADataType* A = (ADataType*)malloc(sizeA);
    BDataType* B = (BDataType*)malloc(sizeB);
    DDataType* D = (DDataType*)malloc(sizeD);

    void *A_d, *B_d, *D_d;

    hip_check_error(hipMalloc(static_cast<void**>(&A_d), sizeA));
    hip_check_error(hipMalloc(static_cast<void**>(&B_d), sizeB));
    hip_check_error(hipMalloc(static_cast<void**>(&D_d), sizeD));

    /*******************
   * Initialize data
   *******************/
    for(int64_t i = 0; i < elementsA; i++)
    {
        A[i] = ((double(std::rand())) / double(RAND_MAX) - 0.5) * 100;
    }

    for(int64_t i = 0; i < elementsB; i++)
    {
        B[i] = ((double(std::rand())) / double(RAND_MAX) - 0.5) * 100;
    }

    /********************************************
   * Transfer the Host Tensor to Device Memory *
   ********************************************/
    hip_check_error(hipMemcpy(A_d, static_cast<const void*>(A), sizeA, hipMemcpyHostToDevice));
    hip_check_error(hipMemcpy(B_d, static_cast<const void*>(B), sizeB, hipMemcpyHostToDevice));
    hip_check_error(hipMemset(D_d, 0, sizeD));

    /************************************************
   * Retrieve the memory alignment for each tensor
   ************************************************/
    uint32_t alignmentRequirementA;
    hiptensorGetAlignmentRequirement(handle, A_d, &a_ms_ks, &alignmentRequirementA);

    uint32_t alignmentRequirementB;
    hiptensorGetAlignmentRequirement(handle, B_d, &b_ks_ns, &alignmentRequirementB);

    uint32_t alignmentRequirementD;
    hiptensorGetAlignmentRequirement(&handle, D_d, &d_ms_ns, &alignmentRequirementD);

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
                                       nullptr,
                                       nullptr,
                                       0,
                                       &d_ms_ns,
                                       modeD.data(),
                                       alignmentRequirementD,
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
                         nullptr,
                         nullptr,
                         D_d,
                         work,
                         worksize,
                         0 /* stream */);

    plan.hiptensorPrintContractionMetrics();
    hip_check_error(hipMemcpy(D, D_d, sizeD, hipMemcpyDeviceToHost));

#if !NDEBUG
    std::ofstream tensorA, tensorB, tensorD;
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
    if(elementsD < MAX_ELEMENTS_PRINT_COUNT)
    {
        std::cout << "Tensor D elements:\n";
        hiptensorPrintArrayElements(D, elementsD);
        std::cout << std::endl;
    }
    tensorD.open("tensor_D_scale_contraction_results.txt");
    hiptensorPrintElementsToFile(tensorD, D, elementsD, ',');
    std::cout << std::endl;
    tensorD.close();
#endif

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
        hip_check_error(hipFree(A_d));
    }

    if(B_d)
    {
        hip_check_error(hipFree(B_d));
    }

    if(D_d)
    {
        hip_check_error(hipFree(D_d));
    }

    return 0;
}
