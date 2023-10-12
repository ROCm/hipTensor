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
#ifndef HIPTENSOR_API_HPP
#define HIPTENSOR_API_HPP

#include "hiptensor_types.hpp"
#include "internal/hiptensor_utility.hpp"

/**
 * \brief Allocates an instance of hiptensorHandle_t on the heap and updates the handle pointer
 *
 * \details Creates hipTensor handle for the associated device.
 * In order for the  hipTensor library to use a different device, set the new
 * device to be used by calling hipInit(0) and then create another hipTensor
 * handle, which will be associated with the new device, by calling
 * hiptensorCreate().
 *
 * \param[out] handle Pointer to hiptensorHandle_t pointer
 *
 * \returns HIPTENSOR_STATUS_SUCCESS on success and an error code otherwise
 */

hiptensorStatus_t hiptensorCreate(hiptensorHandle_t** handle);

/**
 * \brief De-allocates the instance of hiptensorHandle_t
 *
 * \param[out] handle Pointer to hiptensorHandle_t
 *
 * \returns HIPTENSOR_STATUS_SUCCESS on success and an error code otherwise
 */

hiptensorStatus_t hiptensorDestroy(hiptensorHandle_t* handle);

/**
 * \brief Initializes a tensor descriptor
 *
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[out] desc Pointer to the allocated tensor descriptor object.
 * \param[in] numModes Number of modes.
 * \param[in] lens Extent of each mode(lengths) (must be larger than zero).
 * \param[in] strides stride[i] denotes the displacement (stride) between two consecutive
 * elements in the ith-mode. If stride is NULL, generalized packed column-major memory
 * layout is assumed (i.e., the strides increase monotonically from left to right).
 * \param[in] dataType Data type of the stored entries.
 * \param[in] unaryOp Unary operator that will be applied to the tensor
 * \retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 */

hiptensorStatus_t hiptensorInitTensorDescriptor(const hiptensorHandle_t*     handle,
                                                hiptensorTensorDescriptor_t* desc,
                                                const uint32_t               numModes,
                                                const int64_t                lens[],
                                                const int64_t                strides[],
                                                hipDataType                  dataType,
                                                hiptensorOperator_t          unaryOp);

/**
 * \brief Returns the description string for an error code
 * \param[in] error Error code to convert to string.
 * \retval the error string.
 */
const char* hiptensorGetErrorString(const hiptensorStatus_t error);

/**
 * \brief Computes the alignment requirement for a given pointer and descriptor.
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[in] ptr Pointer to the respective tensor data.
 * \param[in] desc Tensor descriptor for ptr data.
 * \param[out] alignmentRequirement Largest alignment requirement that ptr can
 * fulfill (in bytes).
 * \retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \retval HIPTENSOR_STATUS_INVALID_VALUE  if the unsupported parameter is passed.
 */

hiptensorStatus_t hiptensorGetAlignmentRequirement(const hiptensorHandle_t*           handle,
                                                   const void*                        ptr,
                                                   const hiptensorTensorDescriptor_t* desc,
                                                   uint32_t* alignmentRequirement);

/**
 * \brief Initializes a contraction descriptor for the tensor contraction problem.
 *
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[out] desc Tensor contraction problem descriptor.
 * \param[in] descA A descriptor that holds information about tensor A.
 * \param[in] modeA Array with 'nmodeA' entries that represent the modes of A.
 * \param[in] alignmentRequirementA Alignment reqirement for A's pointer (in bytes);
 * \param[in] descB A descriptor that holds information about tensor B.
 * \param[in] modeB Array with 'nmodeB' entries that represent the modes of B.
 * \param[in] alignmentRequirementB Alignment reqirement for B's pointer (in bytes);
 * \param[in] modeC Array with 'nmodeC' entries that represent the modes of C.
 * \param[in] descC A descriptor that holds information about tensor C.
 * \param[in] alignmentRequirementC Alignment requirement for C's pointer (in bytes);
 * \param[in] modeD Array with 'nmodeD' entries that represent the modes of D
 * (must be identical to modeC).
 * \param[in] descD A descriptor that holds information about tensor D
 * (must be identical to descC).
 * \param[in] alignmentRequirementD Alignment requirement for D's pointer (in bytes);
 * \param[in] typeCompute Datatype for the intermediate computation  T = A * B.
 * \retval HIPTENSOR_STATUS_SUCCESS Successful completion of the operation.
 * \retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle or tensor descriptors are not initialized.
 */

hiptensorStatus_t hiptensorInitContractionDescriptor(const hiptensorHandle_t*           handle,
                                                     hiptensorContractionDescriptor_t*  desc,
                                                     const hiptensorTensorDescriptor_t* descA,
                                                     const int32_t                      modeA[],
                                                     const uint32_t alignmentRequirementA,
                                                     const hiptensorTensorDescriptor_t* descB,
                                                     const int32_t                      modeB[],
                                                     const uint32_t alignmentRequirementB,
                                                     const hiptensorTensorDescriptor_t* descC,
                                                     const int32_t                      modeC[],
                                                     const uint32_t alignmentRequirementC,
                                                     const hiptensorTensorDescriptor_t* descD,
                                                     const int32_t                      modeD[],
                                                     const uint32_t         alignmentRequirementD,
                                                     hiptensorComputeType_t typeCompute);

/**
 * \brief Narrows down the candidates for the contraction problem.
 *
 * \details This function gives the user finer control over the candidates that
 * the subsequent call to \ref hiptensorInitContractionPlan is allowed to
 * evaluate. Currently, the backend provides few set of algorithms(DEFAULT).
 *
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[out] find Narrowed set of candidates for the contraction problem.
 * \param[in] algo Allows users to select a specific algorithm.
 * \retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval HIPTENSOR_STATUS_NOT_SUPPORTED If a specified algorithm is not supported
 * \retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle or find is not initialized.
 */

hiptensorStatus_t hiptensorInitContractionFind(const hiptensorHandle_t*    handle,
                                               hiptensorContractionFind_t* find,
                                               const hiptensorAlgo_t       algo);

/**
 * \brief Computes the size of workspace for a given tensor contraction
 *
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[in] desc Tensor contraction descriptor.
 * \param[in] find Narrowed set of candidates for the contraction problem.
 * \param[in] pref Preference to choose the workspace size.
 * \param[out] workspaceSize Size of the workspace (in bytes).
 * \retval HIPTENSOR_STATUS_SUCCESS Successful completion of the operation.
 * \retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \retval HIPTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
 */
hiptensorStatus_t hiptensorContractionGetWorkspaceSize(const hiptensorHandle_t* handle,
                                                       const hiptensorContractionDescriptor_t* desc,
                                                       const hiptensorContractionFind_t*       find,
                                                       const hiptensorWorksizePreference_t     pref,
                                                       uint64_t* workspaceSize);

/**
 * \brief Initializes the contraction plan for a given tensor contraction problem
 *
 * \details This function creates a contraction plan for the problem by applying
 * hipTensor's heuristics to select a candidate. The creaated plan can be reused
 * multiple times for the same tensor contraction problem. The plan is created for
 * the active HIP device.
 *
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[out] plan Opaque handle holding the contraction plan (i.e.,
 * the algorithm that will be executed, its runtime parameters for the given
 * tensor contraction problem).
 * \param[in] desc Tensor contraction descriptor.
 * \param[in] find Narrows down the candidates for the contraction problem.
 * \param[in] workspaceSize Available workspace size (in bytes).
 *
 * \retval HIPTENSOR_STATUS_SUCCESS If a viable candidate has been found.
 * \retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle or find or desc is not
 * initialized.
 */

hiptensorStatus_t hiptensorInitContractionPlan(const hiptensorHandle_t*                handle,
                                               hiptensorContractionPlan_t*             plan,
                                               const hiptensorContractionDescriptor_t* desc,
                                               const hiptensorContractionFind_t*       find,
                                               const uint64_t workspaceSize);
/**
 * \brief Computes the tensor contraction \f[ D = alpha * A * B + beta * C \f]
 *
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * HIP Device associated with the handle must be same/active at the time,0
 * the plan was created.
 * \param[in] plan Opaque handle holding the contraction plan (i.e.,
 * the algorithm that will be executed, its runtime parameters for the given
 * tensor contraction problem).
 * \param[in] alpha Scaling parameter for A*B of data type 'typeCompute'.
 * \param[in] A Pointer to A's data in device memory.
 * \param[in] B Pointer to B's data in device memory.
 * \param[in] beta Scaling parameter for C of data type 'typeCompute'.
 * \param[in] C Pointer to C's data in device memory.
 * \param[out] D Pointer to D's data in device memory.
 * \param[out] workspace Workspace pointer in device memory
 * \param[in] workspaceSize Available workspace size.
 * \param[in] stream HIP stream to perform all operations.
 *
 * Supported data-type combinations are:
 *
 * \verbatim embed:rst:leading-asterisk
 * +---------------+---------------+---------------+-------------------------+
 * |     typeA     |     typeB     |     typeC     |        typeCompute      |
 * +===============+===============+===============+=========================+
 * |    HIP_R_32F  |    HIP_R_32F  |   HIP_R_32F   |    HIPENSOR_COMPUTE_32F |
 * +---------------+---------------+---------------+-------------------------+
 * |    HIP_R_64F  |    HIP_R_64F  |   HIP_R_64F   |    HIPENSOR_COMPUTE_64F |
 * +---------------+---------------+---------------+-------------------------+
 * \endverbatim

 * \retval HIPTENSOR_STATUS_SUCCESS Successful completion of the operation.
 * \retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle or pointers are not
 initialized.
 * \retval HIPTENSOR_STATUS_CK_ERROR if some unknown composable_kernel (CK)
 error has occurred (e.g., no instance supported by inputs).
 */

hiptensorStatus_t hiptensorContraction(const hiptensorHandle_t*          handle,
                                       const hiptensorContractionPlan_t* plan,
                                       const void*                       alpha,
                                       const void*                       A,
                                       const void*                       B,
                                       const void*                       beta,
                                       const void*                       C,
                                       void*                             D,
                                       void*                             workspace,
                                       uint64_t                          workspaceSize,
                                       hipStream_t                       stream);

/**
 * \brief Registers a callback function that will be invoked by logger calls.
 * Note: Functionally additive to existing logging functionality.
 *
 * \param[in] callback This parameter is the callback function pointer provided to the logger.
 * \retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
 * \retval HIPTENSOR_STATUS_INVALID_VALUE if the given callback is invalid.
 */
hiptensorStatus_t hiptensorLoggerSetCallback(hiptensorLoggerCallback_t callback);

/**
 * \brief Registers a file output stream to redirect logging output to.
 * Note: File stream must be open and writable in text mode.
 *
 * \param[in] file This parameter is a file stream pointer provided to the logger.
 * \retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
 * \retval HIPTENSOR_STATUS_IO_ERROR if the output file is not valid (defaults back to stdout).
 */
hiptensorStatus_t hiptensorLoggerSetFile(FILE* file);

/**
 * \brief Redirects log output to a file given by the user.
 *
 * \param[in] logFile This parameter is a file name (relative to binary) or full path to redirect logger output.
 * \retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
 * \retval HIPTENSOR_STATUS_IO_ERROR if the output file is not valid (defaults back to stdout).
 */
hiptensorStatus_t hiptensorLoggerOpenFile(const char* logFile);

/**
 * \brief User-specified logging level. Logs in other contexts will not be recorded.
 *
 * \param[in] level This parameter is the logging level to be enforced.
 * \retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
 * \retval HIPTENSOR_STATUS_INVALID_VALUE if the given log level is invalid.
 */
hiptensorStatus_t hiptensorLoggerSetLevel(hiptensorLogLevel_t level);

/**
 * \brief User-specified logging mask. A mask may be a binary OR combination of
 * several log levels together. Logs in other contexts will not be recorded.
 *
 * \param[in] mask This parameter is the logging mask to be enforced.
 * \retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
 * \retval HIPTENSOR_STATUS_INVALID_VALUE if the given log mask is invalid.
 */
hiptensorStatus_t hiptensorLoggerSetMask(int32_t mask);

/**
 * \brief Disables logging.
 *
 * \retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
 */
hiptensorStatus_t hiptensorLoggerForceDisable();

/**
 * \brief Query HIP runtime version.
 *
 * \retval -1 if the operation failed.
 * \retval Integer HIP runtime version if the operation succeeded.
 */
int hiptensorGetHiprtVersion();

#endif // HIPTENSOR_API_HPP
