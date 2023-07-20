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
#include "contraction_resource.hpp"

#include "common.hpp"

#define MAX_ELEMENTS_PRINT_COUNT 512

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
    typedef float DDataType;
    typedef float floatTypeCompute;

    hipDataType            typeA       = HIP_R_32F;
    hipDataType            typeB       = HIP_R_32F;
    hipDataType            typeC       = HIP_R_32F;
    hipDataType            typeD       = HIP_R_32F;
    hiptensorComputeType_t typeCompute = HIPTENSOR_COMPUTE_32F;

    floatTypeCompute alpha = (floatTypeCompute)2.0f;
    floatTypeCompute beta  = (floatTypeCompute)2.0f;

    /**********************
   * Computing: C_{m,n,u,v} = alpha * A_{m,n,h,k} B_{u,v,h,k} + beta *
   *C_{m,n,u,v}
   **********************/

    std::vector<int> modeD{'m', 'n', 'u', 'v'};
    std::vector<int> modeC{'m', 'n', 'u', 'v'};
    std::vector<int> modeA{'m', 'n', 'h', 'k'};
    std::vector<int> modeB{'u', 'v', 'h', 'k'};

    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();
    int nmodeD = modeD.size();

    std::unordered_map<int, int64_t> extent;

    extent['m'] = 5;
    extent['n'] = 6;
    extent['u'] = 3;
    extent['v'] = 4;
    extent['h'] = 3;
    extent['k'] = 4;

    using DataStorage = hiptensor::ContractionResource;
    auto& dataInstance = DataStorage::instance();

    hiptensor::ContractionResource::ProblemDims problemSize = {5, 6, 3, 4, 3, 4};
    hiptensor::ContractionResource::ElementBytes bytesPerElement = 
        {sizeof(typeA), sizeof(typeB), sizeof(typeC), sizeof(typeD)};
    dataInstance->resizeStorage(problemSize, bytesPerElement);


    std::vector<int64_t> c_ms_ns_lengths;
    for(auto mode : modeC)
    {
        c_ms_ns_lengths.push_back(extent[mode]);
    }

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

    std::vector<int64_t> b_ns_ks_lengths;
    for(auto mode : modeB)
    {
        b_ns_ks_lengths.push_back(extent[mode]);
    }

    hiptensorHandle_t* handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

    CHECK_HIPTENSOR_ERROR(
        hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_ERROR | HIPTENSOR_LOG_LEVEL_PERF_TRACE));

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
        b_ns_ks_lengths.begin(), b_ns_ks_lengths.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsC = std::accumulate(
        c_ms_ns_lengths.begin(), c_ms_ns_lengths.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsD = std::accumulate(
        d_ms_ns_lengths.begin(), d_ms_ns_lengths.end(), size_t{1}, std::multiplies<size_t>());

    size_t sizeD = sizeof(DDataType) * elementsD;
    void* D_ref_d;
    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&D_ref_d), sizeD));

    /*******************
   * Initialize data
   *******************/

    for(int64_t i = 0; i < elementsA; i++)
    {
        static_cast<ADataType*>(dataInstance->hostA().get())[i] = ((float(std::rand())) / float(RAND_MAX) - 0.5) * 10;
    }

    for(int64_t i = 0; i < elementsB; i++)
    {
        static_cast<BDataType*>(dataInstance->hostB().get())[i] = ((float(std::rand())) / float(RAND_MAX) - 0.5) * 10;
    }

    for(int64_t i = 0; i < elementsC; i++)
    {
        static_cast<CDataType*>(dataInstance->hostC().get())[i] = ((float(std::rand())) / float(RAND_MAX) - 0.5) * 10;
    }

    for(int64_t i = 0; i < elementsD; i++)
    {
        static_cast<DDataType*>(dataInstance->hostD().get())[i] = std::numeric_limits<DDataType>::signaling_NaN();
    }

    /********************************************
   * Transfer the Host Tensor to Device Memory *
   ********************************************/
    std::cout << "Initializing device data..." << std::endl;

    dataInstance->copyHostToDeviceAll(bytesPerElement);

    /************************************************
   * Retrieve the memory alignment for each tensor
   ************************************************/

    uint32_t alignmentRequirementA;
    CHECK_HIPTENSOR_ERROR(
        hiptensorGetAlignmentRequirement(handle, dataInstance->deviceA().get(), &a_ms_ks, &alignmentRequirementA));

    uint32_t alignmentRequirementB;
    CHECK_HIPTENSOR_ERROR(
        hiptensorGetAlignmentRequirement(handle, dataInstance->deviceB().get(), &b_ns_ks, &alignmentRequirementB));

    uint32_t alignmentRequirementC;
    CHECK_HIPTENSOR_ERROR(
        hiptensorGetAlignmentRequirement(handle, dataInstance->deviceC().get(), &c_ms_ns, &alignmentRequirementC));

    uint32_t alignmentRequirementD;
    CHECK_HIPTENSOR_ERROR(
        hiptensorGetAlignmentRequirement(handle, dataInstance->deviceD().get(), &d_ms_ns, &alignmentRequirementD));

    /*******************************
   * Create Contraction Descriptor
   *******************************/

    std::cout << "a_ms_ks: " << a_ms_ks << std::endl;
    std::cout << "b_ns_ks: " << b_ns_ks << std::endl;
    std::cout << "c_ms_ns: " << c_ms_ns << std::endl;
    std::cout << "d_ms_ns: " << d_ms_ns << std::endl;

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
                                                             &d_ms_ns,
                                                             modeD.data(),
                                                             alignmentRequirementD,
                                                             typeCompute));
    /**************************
   * Set the algorithm to use
   ***************************/

    hiptensorContractionFind_t find;
    CHECK_HIPTENSOR_ERROR(hiptensorInitContractionFind(handle, &find, HIPTENSOR_ALGO_ACTOR_CRITIC));

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
                                               dataInstance->deviceA().get(),
                                               dataInstance->deviceB().get(),
                                               (void*)&beta,
                                               dataInstance->deviceC().get(),
                                               dataInstance->deviceD().get(),
                                               workspace,
                                               worksize,
                                               0 /* stream */));

    CHECK_HIPTENSOR_ERROR(hiptensorContractionReference((void*)&alpha,
                                                        dataInstance->hostA().get(),
                                                        dataInstance->hostB().get(),
                                                        (void*)&beta,
                                                        dataInstance->hostC().get(),
                                                        dataInstance->hostD().get(),
                                                        a_ms_ks.mLengths,
                                                        a_ms_ks.mStrides,
                                                        b_ns_ks.mLengths,
                                                        b_ns_ks.mStrides,
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

    CHECK_HIP_ERROR(hipMemcpy(D_ref_d, dataInstance->hostD().get(), sizeD, hipMemcpyHostToDevice));

    std::tie(mValidationResult, mMaxRelativeError) = compareEqualLaunchKernel<DDataType>(
        static_cast<DDataType*>(dataInstance->deviceD().get()), static_cast<DDataType*>(D_ref_d), elementsD);

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

    if(printElements || storeElements)
    {
        CHECK_HIP_ERROR(hipMemcpy(dataInstance->hostD().get(), dataInstance->deviceD().get(), sizeD, hipMemcpyDeviceToHost));
    }

    if(printElements)
    {
        if(elementsA < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor A elements:\n";
            hiptensorPrintArrayElements<ADataType>(static_cast<ADataType*>(dataInstance->hostA().get()), elementsA);
            std::cout << std::endl;
        }

        if(elementsB < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor B elements:\n";
            hiptensorPrintArrayElements(static_cast<BDataType*>(dataInstance->hostB().get()), elementsB);
            std::cout << std::endl;
        }

        if(elementsC < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor C elements:\n";
            hiptensorPrintArrayElements(static_cast<CDataType*>(dataInstance->hostC().get()), elementsC);
            std::cout << std::endl;
        }

        if(elementsD < MAX_ELEMENTS_PRINT_COUNT)
        {
            std::cout << "Tensor D Device elements:\n";
            hiptensorPrintArrayElements(static_cast<DDataType*>(dataInstance->hostD().get()), elementsD);
            std::cout << std::endl;
        }
    }

    if(storeElements)
    {
        std::ofstream tensorA, tensorB, tensorC, tensorD;

        tensorA.open("tensor_A.txt");
        hiptensorPrintElementsToFile(tensorA, static_cast<ADataType*>(dataInstance->hostA().get()), elementsA, ", ");
        tensorA.close();

        tensorB.open("tensor_B.txt");
        hiptensorPrintElementsToFile(tensorB, static_cast<ADataType*>(dataInstance->hostB().get()), elementsB, ", ");
        tensorB.close();

        tensorC.open("tensor_C.txt");
        hiptensorPrintElementsToFile(tensorC, static_cast<ADataType*>(dataInstance->hostC().get()), elementsC, ", ");
        tensorC.close();

        tensorD.open("tensor_D_bilinear_contraction_results.txt");
        hiptensorPrintElementsToFile(tensorD, static_cast<ADataType*>(dataInstance->hostD().get()), elementsD, ", ");
        tensorD.close();
    }

    HIPTENSOR_FREE_DEVICE(D_ref_d);

    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));

    HIPTENSOR_FREE_DEVICE(workspace);

    std::cout << "Finished!" << std::endl;

    return 0;
}
