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
#ifndef HIPTENSOR_TYPES_HPP
#define HIPTENSOR_TYPES_HPP

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
 * \brief hipTensor status type
 * \details The type is used to indicate the status of hipTensor library functions.
 * It can have the following values.
 */
typedef enum
{
    /** The operation is successful.*/
    HIPTENSOR_STATUS_SUCCESS = 0,
    /** The handle was not initialized.*/
    HIPTENSOR_STATUS_NOT_INITIALIZED = 1,
    /** Resource allocation failed inside the hipTensor library.*/
    HIPTENSOR_STATUS_ALLOC_FAILED = 3,
    /** Invalid value or parameter was passed to the function (indicates an
     user error).*/
    HIPTENSOR_STATUS_INVALID_VALUE = 7,
    /** Indicates that the target architecure is not supported, or the
     device is not ready.*/
    HIPTENSOR_STATUS_ARCH_MISMATCH = 8,
    /** Indicates the failure of a GPU program or a kernel, which can be caused by multiple
     reasons.*/
    HIPTENSOR_STATUS_EXECUTION_FAILED = 13,
    /** An internal error has occurred.*/
    HIPTENSOR_STATUS_INTERNAL_ERROR = 14,
    /** The requested operation is not supported.*/
    HIPTENSOR_STATUS_NOT_SUPPORTED = 15,
    /** A call to Composable Kernels did not succeed.*/
    HIPTENSOR_STATUS_CK_ERROR = 17,
    /** Unknown hipTensor error has occurred.*/
    HIPTENSOR_STATUS_HIP_ERROR = 18,
    /** The provided workspace was insufficient.*/
    HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19,
    /** Indicates that the driver version is insufficient.*/
    HIPTENSOR_STATUS_INSUFFICIENT_DRIVER = 20,
    /** Indicates an error related to file I/O.*/
    HIPTENSOR_STATUS_IO_ERROR = 21,
} hiptensorStatus_t;

/**
 * \brief hipTensor's compute type
 *
 */
typedef enum
{
    HIPTENSOR_COMPUTE_32F = (1U << 2U),
    HIPTENSOR_COMPUTE_64F = (1U << 4U),
    /*!< Following types to be added (TBA) */
    HIPTENSOR_COMPUTE_16F  = (1U << 0U),
    HIPTENSOR_COMPUTE_16BF = (1U << 10U),
    HIPTENSOR_COMPUTE_8U   = (1U << 6U),
    HIPTENSOR_COMPUTE_8I   = (1U << 8U),
    HIPTENSOR_COMPUTE_32U  = (1U << 7U),
    HIPTENSOR_COMPUTE_32I  = (1U << 9U),
    HIPTENSOR_COMPUTE_NONE = 0
} hiptensorComputeType_t;

/**
 * \brief This enum captures the operations supported by the hipTensor library.
 */
typedef enum
{
    HIPTENSOR_OP_IDENTITY = 1, /*!< Identity operator  */
    HIPTENSOR_OP_UNKNOWN  = 126, /*!< reserved */
} hiptensorOperator_t;

/**
 * \brief This captures the algorithm to be used to perform the tensor contraction.
 */
typedef enum
{
    HIPTENSOR_ALGO_ACTOR_CRITIC = -8, /*!< Uses novel actor-critic selection model (To be Added) */
    HIPTENSOR_ALGO_DEFAULT      = -1, /*!< Lets the internal heuristic choose */
    HIPTENSOR_ALGO_DEFAULT_PATIENT = -6, /*!< Uses the more accurate and time-consuming model */
} hiptensorAlgo_t;

/**
 * \brief This enum gives control over the workspace selection
 */
typedef enum
{
    HIPTENSOR_WORKSPACE_MIN         = 1, /*!< At least one algorithm will be available */
    HIPTENSOR_WORKSPACE_RECOMMENDED = 2, /*!< The most suitable algorithm will be available */
    HIPTENSOR_WORKSPACE_MAX         = 3, /*!< All algorithms will be available */
} hiptensorWorksizePreference_t;

/**
 * \brief This enum decides the logging context.
 * \details The logger output of certain contexts maybe constrained to these levels.
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
 * \brief hipTensor's library context contained in a opaque handle
 */
struct hiptensorHandle_t
{
    int64_t fields[512];
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
    int32_t mContractionOpId; /*!< Enum that differentiates the internal contraction operation */
    hiptensorComputeType_t                   mComputeType; /*!<Compute type for the contraction */
    std::vector<hiptensorTensorDescriptor_t> mTensorDesc; /*!<Cache of tensor descriptors */
    std::vector<uint32_t>                    mAlignmentReq; /*!<Cache of alignment requirements */
};

/**
 * \brief hipTensor structure representing the algorithm candidates.
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

#endif // HIPTENSOR_TYPES_HPP
