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
#ifndef HT_TYPES_H_
#define HT_TYPES_H_

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include <hip/hip_common.h>
#include <hip/library_types.h>

/**
 * \brief hiptensor status type returns
 * \details The type is used for function status returns. All hiptensor library
 * functions return their status, which can have the following values.
 *
 */
typedef enum
{
    /** The operation completed successfully.*/
    HIPTENSOR_STATUS_SUCCESS = 0,
    /** The opaque data structure was not initialized.*/
    HIPTENSOR_STATUS_NOT_INITIALIZED = 1,
    /** Resource allocation failed inside the hiptensor library.*/
    HIPTENSOR_STATUS_ALLOC_FAILED = 3,
    /** An unsupported value or parameter was passed to the function (indicates an
     user error).*/
    HIPTENSOR_STATUS_INVALID_VALUE = 7,
    /** Indicates that the device is either not ready, or the target architecture
     is not supported.*/
    HIPTENSOR_STATUS_ARCH_MISMATCH = 8,
    /** The GPU program failed to execute. This is often caused by a launch
     failure of the kernel on the GPU, which can be caused by multiple
     reasons.*/
    HIPTENSOR_STATUS_EXECUTION_FAILED = 13,
    /** An internal hiptensor error has occurred.*/
    HIPTENSOR_STATUS_INTERNAL_ERROR = 14,
    /** The requested operation is not supported.*/
    HIPTENSOR_STATUS_NOT_SUPPORTED = 15,
    /** A call to CUBLAS did not succeed.*/
    HIPTENSOR_STATUS_CK_ERROR = 17,
    /** Some unknown hiptensor error has occurred.*/
    HIPTENSOR_STATUS_HIP_ERROR = 18,
    /** The provided workspace was insufficient.*/
    HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19,
    /** Indicates that the driver version is insufficient.*/
    HIPTENSOR_STATUS_INSUFFICIENT_DRIVER = 20,
    /** Indicates an error related to file I/O.*/
    HIPTENSOR_STATUS_IO_ERROR = 21,
} hiptensorStatus_t;

/**
 * \brief Encodes hiptensor's compute type
 * \note Only the HIPTENSOR_COMPUTE_32F compute is supported.
 * \todo Needs to add support for the other computing types.
 *
 */
typedef enum
{
    HIPTENSOR_COMPUTE_16F = (1U << 0U), ///< floating-point: 5-bit exponent and
    ///< 10-bit mantissa (aka half)
    HIPTENSOR_COMPUTE_16BF = (1U << 10U), ///< floating-point: 8-bit exponent and
    ///< 7-bit mantissa (aka bfloat)
    HIPTENSOR_COMPUTE_TF32
    = (1U << 12U), ///< floating-point: 8-bit exponent and 10-bit mantissa (aka
    ///< tensor-float-32)
    HIPTENSOR_COMPUTE_32F = (1U << 2U), ///< floating-point: 8-bit exponent and
    ///< 23-bit mantissa (aka float)
    HIPTENSOR_COMPUTE_64F = (1U << 4U), ///< floating-point: 11-bit exponent and
    ///< 52-bit mantissa (aka double)
    HIPTENSOR_COMPUTE_8U  = (1U << 6U), ///< 8-bit unsigned integer
    HIPTENSOR_COMPUTE_8I  = (1U << 8U), ///< 8-bit signed integer
    HIPTENSOR_COMPUTE_32U = (1U << 7U), ///< 32-bit unsigned integer
    HIPTENSOR_COMPUTE_32I = (1U << 9U), ///< 32-bit signed integer
} hiptensorComputeType_t;

/**
 * \brief This enum captures the operations supported by the hiptensor library.
 * \todo Other operations supported in the cuTENSOR needs to be adapted in the
 * hiptensor library.
 */
typedef enum
{
    HIPTENSOR_OP_IDENTITY = 1, ///< Identity operator (i.e., elements are not changed)
    HIPTENSOR_OP_UNKNOWN  = 126, ///< reserved for internal use only
} hiptensorOperator_t;

/**
 * \brief Allows users to specify the algorithm to be used for performing the
 * tensor contraction. \details This enum gives users finer control over which
 * algorithm should be executed by hiptensorContraction(); values >= 0
 * correspond to certain sub-algorithms of GETT. \note Only the default
 * algorithm(HIPTENSOR_ALGO_DEFAULT) is supported by the hiptensor. \todo need
 * to add support for other algorithm in the hiptensor future releases.
 */
typedef enum
{
    HIPTENSOR_ALGO_ACTOR_CRITIC    = -8, ///< Uses novel actor-critic selection model
    HIPTENSOR_ALGO_DEFAULT_PATIENT = -6, ///< Uses the more accurate but also more
    ///< time-consuming performance model
    HIPTENSOR_ALGO_GETT  = -4, ///< Choose the GETT algorithm
    HIPTENSOR_ALGO_TGETT = -3, ///< Transpose (A or B) + GETT
    HIPTENSOR_ALGO_TTGT  = -2, ///< Transpose-Transpose-GEMM-Transpose (requires additional memory)
    HIPTENSOR_ALGO_DEFAULT = -1, ///< Lets the internal heuristic choose
} hiptensorAlgo_t;

/**
 * \brief This enum gives users finer control over the suggested workspace
 * \details This enum gives users finer control over the amount of workspace
 * that is suggested by hipTensorContractionGetWorkspaceSize. \warning Not
 * supported by the current composable_kernel backend. Need to adapt in the
 * future releases.
 *
 */
typedef enum
{
    HIPTENSOR_WORKSPACE_MIN         = 1, ///< At least one algorithm will be available
    HIPTENSOR_WORKSPACE_RECOMMENDED = 2, ///< The most suitable algorithm will be available
    HIPTENSOR_WORKSPACE_MAX         = 3, ///< All algorithms will be available
} hiptensorWorksizePreference_t;

/**
 * \brief This enum decides the logging context.
 * \details The logger output of certain contexts maybe contrained to these levels.
 */
typedef enum
{
    HIPTENSOR_LOG_LEVEL_OFF              = 0,
    HIPTENSOR_LOG_LEVEL_ERROR            = 1,
    HIPTENSOR_LOG_LEVEL_PERF_TRACE       = 2,
    HIPTENSOR_LOG_LEVEL_PERF_HINT        = 4,
    HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE = 8,
    HIPTENSOR_LOG_LEVEL_API_TRACE        = 16
} hiptensorLogLevel_t;

/**
 * \brief Opaque structure holding hiptensor's library context.
 */
struct hiptensorHandle_t
{
    int32_t mHipDevice;
};

/**
 * \brief Structure representing a tensor descriptor with the given lengths, and
 * strides.
 *
 * Constructs a descriptor for the input tensor with the given lengths, strides
 * when passed in the function hiptensorInitTensorDescriptor
 */
struct hiptensorTensorDescriptor_t
{
    hipDataType              mType; /*!< Data type of the tensors enum selection */
    std::vector<std::size_t> mLengths; /*!< Lengths of the tensor */
    std::vector<std::size_t> mStrides; /*!< Strides of the tensor */
};

/**
 * \brief Structure representing a tensor contraction descriptor
 *
 * Constructs a contraction descriptor with all the input tensor descriptors and
 * updates the dimensions on to this structure when passed into the function
 * hiptensorInitContractionDescriptor
 *
 */
struct hiptensorContractionDescriptor_t
{
    int32_t mContractionOpId; /*!< Enum that differentiates the internal contraction operation*/
    hiptensorComputeType_t                   mComputeType; /*!<Compute type for the contraction*/
    std::vector<hiptensorTensorDescriptor_t> mTensorDesc; /*!<Cache of tensor descriptors*/
    std::vector<uint32_t>                    mAlignmentReq; /*!<Cache of alignment requirements*/
};

/**
 * \brief Opaque structure representing a candidate.
 * \todo  Needs to adapt the structure as per the GPU devices in the hiptensor
 * future releases.
 *
 */
struct hiptensorContractionFind_t
{
    hiptensorAlgo_t    mSelectionAlgorithm;
    std::vector<void*> mCandidates;
};

/**
 * \brief structure representing a plan
 *
 * Constructs a contraction plan with the contractions descriptor passed into
 * the function hiptensorInitContractionPlan
 *
 */
struct hiptensorContractionPlan_t
{
    void*                            mSolution;
    hiptensorContractionDescriptor_t mContractionDesc; /*!< Represent the contraction descriptor */
};

/**
 * \brief Logging callback
 *
 * The specified callback is invoked whenever logging is enabled and information
 * is logged.
 *
 */
typedef void (*hiptensorLoggerCallback_t)(int32_t     logContext,
                                          const char* funcName,
                                          const char* msg);

#endif
