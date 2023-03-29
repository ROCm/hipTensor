#include <fstream>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <unordered_map>

// hipTensor includes
#include <hiptensor/ht_types.hpp>
#include <hiptensor/ht_tensor.hpp>
#include <utility/ht_utility.hpp>

#define MAX_ELEMENTS_PRINT_COUNT 512

int main(int argc, char* argv[])
{
    typedef float ADataType;
    typedef float BDataType;
    typedef float CDataType;
    typedef float floatTypeCompute;


    hipTensorDataType_t typeA = hipTensor_R_32F;
    hipTensorDataType_t typeB = hipTensor_R_32F;
    hipTensorDataType_t typeC = hipTensor_R_32F;
    hipTensorComputeType_t typeCompute = hipTensor_COMPUTE_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.0f;
    floatTypeCompute beta  = (floatTypeCompute)0.0f;

#ifdef HT_PRINT_DEBUG
    std::cout << "RAND_MAX value is " << RAND_MAX << std::endl;
#endif
    /**********************
     * Computing: C_{m,n,u,v} = A_{m,n,h,k} B_{h,k,u,v}
     **********************/

    std::vector<int> modeC{'m','n','u','v'};
    std::vector<int> modeA{'m','n','h','k'};
    std::vector<int> modeB{'u','v','h','k'};


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
    for (auto mode : modeC)
    	c_ms_ns_lengths.push_back(extent[mode]);
    std::vector<int64_t> a_ms_ks_lengths;
    for (auto mode : modeA)
       	a_ms_ks_lengths.push_back(extent[mode]);
    std::vector<int64_t> b_ks_ns_lengths;
    for (auto mode : modeB)
        b_ks_ns_lengths.push_back(extent[mode]);

    hipTensorHandle_t handle;
    hipTensorInit(&handle);
    
    /********************************************
     * Intialise Tensors with the input lengths *
     ********************************************/
    hipTensorTensorDescriptor_t a_ms_ks;
    hipTensorInitTensorDescriptor(&handle, &a_ms_ks, nmodeA, 
				a_ms_ks_lengths.data(), NULL,/*stride*/
				typeA, hipTensor_OP_IDENTITY);

#if HT_PRINT_DEBUG
    std::cout << "a_ms_ks: " << a_ms_ks << std::endl;
#endif

    hipTensorTensorDescriptor_t b_ks_ns;
    hipTensorInitTensorDescriptor(&handle, &b_ks_ns, nmodeB,
               b_ks_ns_lengths.data(), NULL,/*stride*/
				typeB, hipTensor_OP_IDENTITY);

#ifdef HT_PRINT_DEBUG
    std::cout << "b_ks_ns: " << b_ks_ns <<  std::endl;
#endif

    hipTensorTensorDescriptor_t c_ms_ns;
    hipTensorInitTensorDescriptor(&handle, 
				&c_ms_ns, nmodeC,
				c_ms_ns_lengths.data(), NULL,/*stride*/
                		typeC, hipTensor_OP_IDENTITY);

#ifdef HT_PRINT_DEBUG
    std::cout << "c_ms_ns: " << c_ms_ns << std::endl;
#endif    

    /**********************
     * Allocating data
     **********************/

    size_t elementsA = a_ms_ks.hipTensorGetElementSpace();
    size_t elementsB = b_ks_ns.hipTensorGetElementSpace();
    size_t elementsC = c_ms_ns.hipTensorGetElementSpace();

    
    size_t sizeA = sizeof(ADataType) * elementsA;
    size_t sizeB = sizeof(BDataType) * elementsB;
    size_t sizeC = sizeof(CDataType) * elementsC;

    ADataType *A = (ADataType*) malloc(sizeA);
    BDataType *B = (BDataType*) malloc(sizeB);
    CDataType *C = (CDataType*) malloc(sizeC);
    
    void *A_d, *B_d, *C_d;
    
    hip_check_error(hipMalloc(static_cast<void**>(&A_d), sizeA));
    hip_check_error(hipMalloc(static_cast<void**>(&B_d), sizeB));
    hip_check_error(hipMalloc(static_cast<void**>(&C_d), sizeC));

    /*******************
     * Initialize data
     *******************/
    for (int64_t i = 0; i < elementsA; i++)
        A[i] = ((float(std::rand()))/float(RAND_MAX) - 0.5)*100;
    for (int64_t i = 0; i < elementsB; i++)
        B[i] = ((float(std::rand()))/float(RAND_MAX) - 0.5)*100;	

    /********************************************
     * Transfer the Host Tensor to Device Memory *
     ********************************************/
    hip_check_error(hipMemcpy(A_d, static_cast<const void*>(A), sizeA, hipMemcpyHostToDevice));
    hip_check_error(hipMemcpy(B_d, static_cast<const void*>(B), sizeB, hipMemcpyHostToDevice));
    hip_check_error(hipMemset(C_d, 0, sizeC));
    
    /************************************************
     * Retrieve the memory alignment for each tensor
     ************************************************/ 
    uint32_t alignmentRequirementA;
    hipTensorGetAlignmentRequirement(&handle,
                          A_d, &a_ms_ks,
                          &alignmentRequirementA);
    
    uint32_t alignmentRequirementB;
    hipTensorGetAlignmentRequirement(&handle,
                          B_d, &b_ks_ns,
                          &alignmentRequirementB);
    
    uint32_t alignmentRequirementC;
    hipTensorGetAlignmentRequirement(&handle,
                          C_d, &c_ms_ns,
                          &alignmentRequirementC);
    
    /*******************************
     * Create Contraction Descriptor
     *******************************/

    hipTensorContractionDescriptor_t desc;
    hipTensorInitContractionDescriptor(&handle,
                                    &desc,
                                    &a_ms_ks, modeA.data(), alignmentRequirementA,
                                    &b_ks_ns, modeB.data(), alignmentRequirementB,
                                    &c_ms_ns, modeC.data(), alignmentRequirementC,
                                    &c_ms_ns, modeC.data(), alignmentRequirementC,
                                    typeCompute);
    /**************************
    * Set the algorithm to use
    ***************************/

    hipTensorContractionFind_t find;
    hipTensorInitContractionFind(&handle, 
                                &find,
                                hipTensor_ALGO_DEFAULT);

   /**********************
    * Query workspace
    **********************/

    uint64_t worksize = 0;
    hipTensorContractionGetWorkspace(&handle,
                                    &desc,
                                    &find,
                                    hipTensor_WORKSPACE_RECOMMENDED, &worksize);
    void *work = nullptr;
	
   /**************************
    * Create Contraction Plan
    **************************/

    hipTensorContractionPlan_t plan;
    hipTensorInitContractionPlan(&handle,
                                &plan,
                                &desc,
                                &find,
                                worksize);

    hipTensorContraction(&handle,
                       &plan,
                       (void*) &alpha, A_d, B_d,
                       (void*) &beta,  C_d, C_d,
                       work, worksize, 0 /* stream */);
    
	plan.hipTensorPrintContractionMetrics();
    hip_check_error(hipMemcpy(C, C_d, sizeC, hipMemcpyDeviceToHost));
    
#if HT_PRINT_DEBUG
    std::ofstream tensorA, tensorB, tensorC;
    if (elementsA < MAX_ELEMENTS_PRINT_COUNT)
    {
        std::cout<<"Tensor A elements:\n";
        hipTensorPrintArrayElements(A, elementsA);    
        std::cout<<std::endl;
    }
    tensorA.open("tensor_A.txt");
    hipTensorPrintElementsToFile(tensorA, A, elementsA, ','); 
    std::cout<<std::endl;
    tensorA.close();
    if (elementsB < MAX_ELEMENTS_PRINT_COUNT)
    {
        std::cout<<"Tensor B elements:\n";
        hipTensorPrintArrayElements(B, elementsB);    
        std::cout<<std::endl;
    }
    tensorB.open("tensor_B.txt");
    hipTensorPrintElementsToFile(tensorB, B, elementsB, ','); 
    std::cout<<std::endl;
    tensorB.close();
    if (elementsC < MAX_ELEMENTS_PRINT_COUNT)
    {
        std::cout<<"Tensor C elements:\n";
        hipTensorPrintArrayElements(C, elementsC);    
        std::cout<<std::endl;
    }
    tensorC.open("tensor_C_scale_contraction_results.txt");
    hipTensorPrintElementsToFile(tensorC, C, elementsC, ','); 
    std::cout<<std::endl;
    tensorC.close();
#endif
	
    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
    if (A_d) hip_check_error(hipFree(A_d));
    if (B_d) hip_check_error(hipFree(B_d));
    if (C_d) hip_check_error(hipFree(C_d));
    
    return 0;
}
