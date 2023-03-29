#ifndef HT_TYPES_H_
#define HT_TYPES_H_

#include <vector>
#include <numeric>
#include <algorithm>
#include <utility>
#include <cassert>
#include <iostream>

/**
 * \brief hipTensor status type returns
 * \details The type is used for function status returns. All hipTensor library functions return their status, which can have the following values.
 *
*/
typedef enum
{
    /** The operation completed successfully.*/
    hipTensor_STATUS_SUCCESS                = 0,
    /** The opaque data structure was not initialized.*/
    hipTensor_STATUS_NOT_INITIALIZED        = 1,
    /** Resource allocation failed inside the hipTensor library.*/
    hipTensor_STATUS_ALLOC_FAILED           = 3,
    /** An unsupported value or parameter was passed to the function (indicates an user error).*/
    hipTensor_STATUS_INVALID_VALUE          = 7,
    /** Indicates that the device is either not ready, or the target architecture is not supported.*/
    hipTensor_STATUS_ARCH_MISMATCH          = 8,
    /** An access to GPU memory space failed, which is usually caused by a failure to bind a texture.*/
    hipTensor_STATUS_MAPPING_ERROR          = 11,
    /** The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.*/
    hipTensor_STATUS_EXECUTION_FAILED       = 13,
    /** An internal hipTensor error has occurred.*/
    hipTensor_STATUS_INTERNAL_ERROR         = 14,
    /** The requested operation is not supported.*/
    hipTensor_STATUS_NOT_SUPPORTED          = 15,
    /** The functionality requested requires some license and an error was detected when trying to check the current licensing.*/
    hipTensor_STATUS_LICENSE_ERROR          = 16,
    /** A call to CUBLAS did not succeed.*/
    hipTensor_STATUS_CK_ERROR           = 17,
    /** Some unknown hipTensor error has occurred.*/
    hipTensor_STATUS_ROCM_ERROR             = 18,
    /** The provided workspace was insufficient.*/
    hipTensor_STATUS_INSUFFICIENT_WORKSPACE = 19,
    /** Indicates that the driver version is insufficient.*/
    hipTensor_STATUS_INSUFFICIENT_DRIVER    = 20,
    /** Indicates an error related to file I/O.*/
    hipTensor_STATUS_IO_ERROR               = 21,
} hipTensorStatus_t;

/**
 * \brief hipTensorDataType_t is an enumeration of the types supported by hipTensor libraries. 
 * hipTensor supports real FP16, BF16, FP32 input types.
 * \note Only hipTensor_R_32F is supported.
 * \todo Other datatypes support to be added in the next hipTensor library releases.
 *
*/
typedef enum 
{
	hipTensor_R_16F  =  0,      ///<real as a half
	hipTensor_R_16BF =  1,      ///<real as a nv_bfloat16
	hipTensor_R_32F  =  2,      ///<real as a float 
	hipTensor_R_64F  =  3,      ///<real as a double 
} hipTensorDataType_t;

/**
 * \brief Encodes hipTensor’s compute type 
 * \note Only the hipTensor_COMPUTE_32F compute is supported.
 * \todo Needs to add support for the other computing types.
 *
*/
typedef enum
{
    hipTensor_COMPUTE_16F  = (1U<< 0U),  ///< floating-point: 5-bit exponent and 10-bit mantissa (aka half)
    hipTensor_COMPUTE_16BF = (1U<< 10U),  ///< floating-point: 8-bit exponent and 7-bit mantissa (aka bfloat)
    hipTensor_COMPUTE_TF32 = (1U<< 12U),  ///< floating-point: 8-bit exponent and 10-bit mantissa (aka tensor-float-32)
    hipTensor_COMPUTE_32F  = (1U<< 2U),  ///< floating-point: 8-bit exponent and 23-bit mantissa (aka float)
    hipTensor_COMPUTE_64F  = (1U<< 4U),  ///< floating-point: 11-bit exponent and 52-bit mantissa (aka double)
    hipTensor_COMPUTE_8U   = (1U<< 6U),  ///< 8-bit unsigned integer
    hipTensor_COMPUTE_8I   = (1U<< 8U),  ///< 8-bit signed integer
    hipTensor_COMPUTE_32U  = (1U<< 7U),  ///< 32-bit unsigned integer
    hipTensor_COMPUTE_32I  = (1U<< 9U),  ///< 32-bit signed integer
} hipTensorComputeType_t;

/**
 * \brief This enum captures the operations supported by the hipTensor library.
 * \todo Other operations supported in the cuTENSOR needs to be adapted in the hipTensor library.
*/
typedef enum
{
    hipTensor_OP_IDENTITY = 1,       ///< Identity operator (i.e., elements are not changed)
    hipTensor_OP_UNKNOWN = 126,      ///< reserved for internal use only
} hipTensorOperator_t;

/**
 * \brief Allows users to specify the algorithm to be used for performing the tensor contraction.
 * \details This enum gives users finer control over which algorithm should be executed by hipTensorContraction(); 
 * values >= 0 correspond to certain sub-algorithms of GETT.
 * \note Only the default algorithm(hipTensor_ALGO_DEFAULT) is supported by the hipTensor.
 * \todo need to add support for other algorithm in the hipTensor future releases.
*/
typedef enum
{
    hipTensor_ALGO_DEFAULT_PATIENT   = -6, ///< Uses the more accurate but also more time-consuming performance model
    hipTensor_ALGO_GETT              = -4, ///< Choose the GETT algorithm
    hipTensor_ALGO_TGETT             = -3, ///< Transpose (A or B) + GETT
    hipTensor_ALGO_TTGT              = -2, ///< Transpose-Transpose-GEMM-Transpose (requires additional memory)
    hipTensor_ALGO_DEFAULT           = -1, ///< Lets the internal heuristic choose
} hipTensorAlgo_t;

/**
 * \brief This enum gives users finer control over the suggested workspace
 * \details This enum gives users finer control over the amount of workspace that is suggested by hipTensorContractionGetWorkspace.
 * \warning Not supported by the current composable_kernel backend. Need to adapt in the future releases.
 *
*/
typedef enum
{
    hipTensor_WORKSPACE_MIN = 1,         ///< At least one algorithm will be available
    hipTensor_WORKSPACE_RECOMMENDED = 2, ///< The most suitable algorithm will be available
    hipTensor_WORKSPACE_MAX = 3,         ///< All algorithms will be available
} hipTensorWorksizePreference_t;


/**
 * \brief This enum decides the over the operation based on the inputs.
 * \details This enum decides the operation based on the in puts passed in the hipTensorContractionGetWorkspace
*/
typedef enum
{
    hipTensor_CONTRACTION_SCALE= 0,       ///< \f${C=\alpha\mathcal{A}\mathcal{B}}\f$
    hipTensor_CONTRACTION_BILINEAR = 1,   ///< \f${D=\alpha\mathcal{A}\mathcal{B}+\beta\mathcal{C}}\f$
}hiptesnorContractionOperation_t;


/**
 * \brief Opaque structure holding hipTensor's library context.
*/
typedef struct { /*TODO: Discuss the struct members */ }hipTensorHandle_t;

/**
 * \brief Structure representing a tensor descriptor with the given lengths, and strides.
 *
 * Constructs a descriptor for the input tensor with the given lengths, strides when passed
 * in the function hipTensorInitTensorDescriptor
*/
struct hipTensorTensorDescriptor_t{

    hipTensorTensorDescriptor_t() = default; /*!< Default Constructor of the structure hipTensorDescriptor_t */
    
    void hipTensorCalculateStrides(); /*!< Function that returns the size of the tensor based on the input length and strides */
    
    template <typename X>
    hipTensorTensorDescriptor_t(const std::vector<X>& lens) : mLens(lens.begin(), lens.end()) 
    {
        this->hipTensorCalculateStrides();
    } /*!< Function that initializes the tensor based on the input lengths*/

    template <typename X, typename Y>
    hipTensorTensorDescriptor_t(const std::vector<X>& lens, const std::vector<Y>& strides)
        : mLens(lens.begin(), lens.end()), mStrides(strides.begin(), strides.end())
    {

    }  /*!< Function that initializes the tensor based on the input length and strides */

    hipTensorDataType_t ht_type;/*!< Data type of the tensors enum selection */

    std::size_t hipTensorGetNumOfDimension() const; /*!< Function that returns the number of dimensions */
    std::size_t hipTensorGetElementSize() const; /*!< Function that returns the total elements size*/
    std::size_t hipTensorGetElementSpace() const; /*!< Function that returns the size of the tensor based on the input length and strides */

    const std::vector<std::size_t>& hipTensorGetLengths() const; /*!< Function that returns the lengths of the tensor */
    const std::vector<std::size_t>& hipTensorGetStrides() const; /*!< Function that returns the strides of the tensor */


    friend std::ostream& operator<<(std::ostream& os, const hipTensorTensorDescriptor_t& desc); /*!< Function that prints the length, strides tensor */

    private:
    std::vector<std::size_t> mLens; /*!< Lengths of the tensor */
    std::vector<std::size_t> mStrides; /*!< Strides of the tensor */
};

/**
 * \brief Structure used to store the tensor descriptor dimensions and strides
 * for the contraction operation.
 *
*/
struct tensor_attr{
    std::vector<std::size_t> lens; /*!< Represent the lengths of the descriptor */
    std::vector<std::size_t> strides; /*!< Represent the strides of the descriptor */
    std::size_t tensor_size; /*!< Represent the allocated size of the tensor*/
};

/**
 * \brief Structure representing a tensor contraction descriptor
 * 
 * Constructs a contraction descriptor with all the input tensor descriptors and updates the 
 * dimensions on to this structure when passed into the function hipTensorInitContractionDescriptor
 *
*/
struct hipTensorContractionDescriptor_t{
	hiptesnorContractionOperation_t ht_contract_op; /*!<Enum that has the contraction operation(scale/bilinear)*/
    std::vector<tensor_attr> ht_contract_attr_desc; /*!<Vector that represents the length,strides,and size of the input tensors*/
    void hipTensorContractionAttrUpdate(const hipTensorTensorDescriptor_t *desc[], 
                                    const uint32_t tensor_size[], const int tensor_desc_num); /*!< Function that updates the param  ht_contract_attr_desc vector*/
};

/**
 * \brief Opaque structure representing a candidate.
 * \todo  Needs to adapt the structure as per the GPU devices in the hipTensor future releases.
 *
*/
struct hipTensorContractionFind_t {/*TODO: Discuss the struct members */ };

/**
 * \brief structure representing a plan
 * 
 * Captures all the perf results: execution time, FLOPS, Transfer speed, and 
 * the CK's contraction instance name
 * 
*/
struct hipTensorContractionMetrics_t {
    float avg_time; /*!<Time to exectued the selected CK's contraction instance*/
    float tflops;  /*!<FLOPS returned by the selected CK's contraction instance*/
    float transfer_speed; /*!<Transfer speed returned by the CK's contraction instance*/
    std::string ht_instance;/*!<CK's contraction instance name */
};

/**
 * \brief structure representing a plan
 * 
 * Constructs a contraction plan with the contractions descriptor passed into the function
 * hipTensorInitContractionPlan
 *
*/
struct hipTensorContractionPlan_t{
    hipTensorContractionDescriptor_t ht_plan_desc; /*!< Represent the contraction descriptor */
    void hipTensorPrintContractionMetrics(); /*!< Function that prints all the perf results of the CK's contraction instance */
};

#endif
