/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

//! @brief Allocates an instance of hiptensorHandle_t on the heap and updates the handle pointer
//!
//! @details Creates hipTensor handle for the associated device.
//! In order for the  hipTensor library to use a different device, set the new
//! device to be used by calling hipInit(0) and then create another hipTensor
//! handle, which will be associated with the new device, by calling
//! hiptensorCreate().
//! @param[out] handle Pointer to hiptensorHandle_t pointer
//! @returns HIPTENSOR_STATUS_SUCCESS on success and an error code otherwise
hiptensorStatus_t hiptensorCreate(hiptensorHandle_t* handle);

//! @brief De-allocates the instance of hiptensorHandle_t
//! @param[out] handle Pointer to hiptensorHandle_t
//! @returns HIPTENSOR_STATUS_SUCCESS on success and an error code otherwise
hiptensorStatus_t hiptensorDestroy(hiptensorHandle_t handle);

hiptensorStatus_t hiptensorHandleResizePlanCache(hiptensorHandle_t handle,
                                                 const uint32_t    numEntries);

hiptensorStatus_t hiptensorHandleWritePlanCacheToFile(const hiptensorHandle_t handle,
                                                      const char              filename[]);

hiptensorStatus_t hiptensorHandleReadPlanCacheFromFile(hiptensorHandle_t handle,
                                                       const char        filename[],
                                                       uint32_t*         numCachelinesRead);

hiptensorStatus_t hiptensorWriteKernelCacheToFile(const hiptensorHandle_t handle,
                                                  const char              filename[]);

hiptensorStatus_t hiptensorReadKernelCacheFromFile(hiptensorHandle_t handle, const char filename[]);
//! @brief Initializes a tensor descriptor
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[out] desc Pointer to the allocated tensor descriptor object.
//! @param[in] numModes Number of modes.
//! @param[in] lens Extent of each mode(lengths) (must be larger than zero).
//! @param[in] strides stride[i] denotes the displacement (stride) between two consecutive
//! elements in the ith-mode. If stride is NULL, generalized packed column-major memory
//! layout is assumed (i.e., the strides increase monotonically from left to right).
//! @param[in] dataType Data type of the stored entries.
//! @param[in] unaryOp Unary operator that will be applied to the tensor.
//! @retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully.
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
hiptensorStatus_t hiptensorCreateTensorDescriptor(const hiptensorHandle_t      handle,
                                                  hiptensorTensorDescriptor_t* desc,
                                                  const uint32_t               numModes,
                                                  const int64_t                lens[],
                                                  const int64_t                strides[],
                                                  hiptensorDataType_t          dataType,
                                                  uint32_t alignmentRequirement);
hiptensorStatus_t hiptensorDestroyTensorDescriptor(hiptensorTensorDescriptor_t desc);

hiptensorStatus_t hiptensorCreateContraction(const hiptensorHandle_t            handle,
                                             hiptensorOperationDescriptor_t*    desc,
                                             const hiptensorTensorDescriptor_t  descA,
                                             const int32_t                      modeA[],
                                             hiptensorOperator_t                opA,
                                             const hiptensorTensorDescriptor_t  descB,
                                             const int32_t                      modeB[],
                                             hiptensorOperator_t                opB,
                                             const hiptensorTensorDescriptor_t  descC,
                                             const int32_t                      modeC[],
                                             hiptensorOperator_t                opC,
                                             const hiptensorTensorDescriptor_t  descD,
                                             const int32_t                      modeD[],
                                             const hiptensorComputeDescriptor_t descCompute);

hiptensorStatus_t hiptensorDestroyOperationDescriptor(hiptensorOperationDescriptor_t desc);

hiptensorStatus_t
    hiptensorOperationDescriptorSetAttribute(const hiptensorHandle_t                 handle,
                                             hiptensorOperationDescriptor_t          desc,
                                             hiptensorOperationDescriptorAttribute_t attr,
                                             const void*                             buf,
                                             size_t                                  sizeInBytes);

hiptensorStatus_t
    hiptensorOperationDescriptorGetAttribute(const hiptensorHandle_t                 handle,
                                             hiptensorOperationDescriptor_t          desc,
                                             hiptensorOperationDescriptorAttribute_t attr,
                                             void*                                   buf,
                                             size_t                                  sizeInBytes);

/**
 * \brief This function creates an operation descriptor for a tensor permutation.
 *
 * \details The tensor permutation has the following general form:
 * \f[ B_{\Pi^B(i_0,i_1,...,i_n)} = \alpha op_A(A_{\Pi^A(i_0,i_1,...,i_n)}) \f]
 *
 * Consequently, this function performs an out-of-place tensor permutation and is a specialization of \ref hiptensorCreateElementwiseBinary.
 *
 * Where
 *    - A and B are multi-mode tensors (of arbitrary data types),
 *    - \f$\Pi^A, \Pi^B\f$ are permutation operators that permute the modes of A, B respectively,
 *    - \f$op_A\f$ is an unary element-wise operators (e.g., IDENTITY, SQR, CONJUGATE), and
 *    - \f$\Psi\f$ is specified in the tensor descriptor descA.
 *
 * Broadcasting (of a mode) can be achieved by simply omitting that mode from the respective tensor.
 *
 * Modes may appear in any order. The only <b>restrictions</b> are:
 *    - modes that appear in A _must_ also appear in the output tensor.
 *    - each mode may appear in each tensor at most once.
 *
 * Supported data-type combinations are:
 *
 * \verbatim embed:rst:leading-asterisk
 * +--------------------+--------------------+-------------------------------+
 * |      typeA         |      typeB         |   descCompute                 |
 * +====================+====================+===============================+
 * |  HIPTENSOR_R_16F   |  HIPTENSOR_R_16F   |  HIPTENSOR_COMPUTE_DESC_16F   |
 * +--------------------+--------------------+-------------------------------+
 * |  HIPTENSOR_R_16F   |  HIPTENSOR_R_16F   |  HIPTENSOR_COMPUTE_DESC_32F   |
 * +--------------------+--------------------+-------------------------------+
 * |  HIPTENSOR_R_16F   |  HIPTENSOR_R_32F   |  HIPTENSOR_COMPUTE_DESC_32F   |
 * +--------------------+--------------------+-------------------------------+
 * |  HIPTENSOR_R_32F   |  HIPTENSOR_R_16F   |  HIPTENSOR_COMPUTE_DESC_32F   |
 * +--------------------+--------------------+-------------------------------+
 * |  HIPTENSOR_R_16BF  |  HIPTENSOR_R_16BF  |  HIPTENSOR_COMPUTE_DESC_16BF  |
 * +--------------------+--------------------+-------------------------------+
 * |  HIPTENSOR_R_16BF  |  HIPTENSOR_R_16BF  |  HIPTENSOR_COMPUTE_DESC_32F   |
 * +--------------------+--------------------+-------------------------------+
 * |  HIPTENSOR_R_32F   |  HIPTENSOR_R_32F   |  HIPTENSOR_COMPUTE_DESC_32F   |
 * +--------------------+--------------------+-------------------------------+
 * |  HIPTENSOR_R_64F   |  HIPTENSOR_R_64F   |  HIPTENSOR_COMPUTE_DESC_64F   |
 * +--------------------+--------------------+-------------------------------+
 * |  HIPTENSOR_R_32F   |  HIPTENSOR_R_64F   |  HIPTENSOR_COMPUTE_DESC_64F   |
 * +--------------------+--------------------+-------------------------------+
 * |  HIPTENSOR_R_64F   |  HIPTENSOR_R_32F   |  HIPTENSOR_COMPUTE_DESC_64F   |
 * +--------------------+--------------------+-------------------------------+
 * |  HIPTENSOR_C_32F   |  HIPTENSOR_C_32F   |  HIPTENSOR_COMPUTE_DESC_32F   |
 * +--------------------+--------------------+-------------------------------+
 * |  HIPTENSOR_C_64F   |  HIPTENSOR_C_64F   |  HIPTENSOR_COMPUTE_DESC_64F   |
 * +--------------------+--------------------+-------------------------------+
 * |  HIPTENSOR_C_32F   |  HIPTENSOR_C_64F   |  HIPTENSOR_COMPUTE_DESC_64F   |
 * +--------------------+--------------------+-------------------------------+
 * |  HIPTENSOR_C_64F   |  HIPTENSOR_C_32F   |  HIPTENSOR_COMPUTE_DESC_64F   |
 * +--------------------+--------------------+-------------------------------+
 * \endverbatim
 *
 * \param[in] handle Opaque handle holding hipTENSOR's library context.
 * \param[out] desc This opaque struct gets allocated and filled with the information that encodes the requested permutation.
 * \param[in] descA The descriptor that holds information about the data type, modes, and strides of A.
 * \param[in] modeA Array of size descA->numModes that holds the names of the modes of A (e.g., if A_{a,b,c} => modeA = {'a','b','c'})
 * \param[in] opA Unary operator that will be applied to each element of A before it is further processed. The original data of this tensor remains unchanged.
 * \param[in] descB The descriptor that holds information about the data type, modes, and strides of B.
 * \param[in] modeB Array of size descB->numModes that holds the names of the modes of B
 * \param[in] descCompute Determines the precision in which this operations is performed.
 * \retval HIPTENSOR_STATUS_NOT_SUPPORTED if the combination of data types or operations is not supported
 * \retval HIPTENSOR_STATUS_INVALID_VALUE if tensor dimensions or modes have an illegal value
 * \retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully without error
 * \retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \remarks calls asynchronous functions, no reentrant, and thread-safe
 */
hiptensorStatus_t hiptensorCreatePermutation(
                 const hiptensorHandle_t handle, hiptensorOperationDescriptor_t* desc,
                 const hiptensorTensorDescriptor_t descA, const int32_t modeA[], hiptensorOperator_t opA,
                 const hiptensorTensorDescriptor_t descB, const int32_t modeB[],
                 const hiptensorComputeDescriptor_t descCompute);

hiptensorStatus_t hiptensorCreatePlanPreference(const hiptensorHandle_t    handle,
                                                hiptensorPlanPreference_t* pref,
                                                hiptensorAlgo_t            algo,
                                                hiptensorJitMode_t         jitMode);

hiptensorStatus_t hiptensorDestroyPlanPreference(hiptensorPlanPreference_t pref);

hiptensorStatus_t hiptensorPlanPreferenceSetAttribute(const hiptensorHandle_t            handle,
                                                      hiptensorPlanPreference_t          pref,
                                                      hiptensorPlanPreferenceAttribute_t attr,
                                                      const void*                        buf,
                                                      size_t sizeInBytes);

hiptensorStatus_t hiptensorPlanGetAttribute(const hiptensorHandle_t  handle,
                                            const hiptensorPlan_t    plan,
                                            hiptensorPlanAttribute_t attr,
                                            void*                    buf,
                                            size_t                   sizeInBytes);

hiptensorStatus_t hiptensorEstimateWorkspaceSize(const hiptensorHandle_t              handle,
                                                 const hiptensorOperationDescriptor_t desc,
                                                 const hiptensorPlanPreference_t      planPref,
                                                 const hiptensorWorksizePreference_t  workspacePref,
                                                 uint64_t* workspaceSizeEstimate);

hiptensorStatus_t hiptensorCreatePlan(const hiptensorHandle_t              handle,
                                      hiptensorPlan_t*                     plan,
                                      const hiptensorOperationDescriptor_t desc,
                                      const hiptensorPlanPreference_t      pref,
                                      uint64_t                             workspaceSizeLimit);

hiptensorStatus_t hiptensorDestroyPlan(hiptensorPlan_t plan);

hiptensorStatus_t hiptensorContract(const hiptensorHandle_t handle,
                                    const hiptensorPlan_t   plan,
                                    const void*             alpha,
                                    const void*             A,
                                    const void*             B,
                                    const void*             beta,
                                    const void*             C,
                                    void*                   D,
                                    void*                   workspace,
                                    uint64_t                workspaceSize,
                                    hipStream_t             stream);

//! @brief Returns the description string for an error code
//! @param[in] error Error code to convert to string.
//! @retval the error string.
const char* hiptensorGetErrorString(const hiptensorStatus_t error);

size_t hiptensorGetVersion();

//! @brief Tensor permutation
//! @details This function computes the permuation operation:
//! \f[
//! B_{\Pi^B(i_0,i_1,...,i_n)} = \alpha \Psi(A_{\Pi^A(i_0,i_1,...,i_n)})
//! \f]
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[in] plan Opaque handle holding all information about the desired tensor permutation (created by \ref hiptensorCreatePermutation followed by \ref hiptensorCreatePlan).
//! @param[in] alpha Scaling factor for A of the type typeScalar. Pointer to the host memory.
//! If alpha is zero, A is not read and the corresponding unary operator is not applied.
//! @param[in] A Multi-mode tensor of type typeA with nmodeA modes. Pointer to the GPU-accessible memory.
//! @param[in,out] B Multi-mode tensor of type typeB with nmodeB modes. Pointer to the GPU-accessible memory.
//! @param[in] stream HIP stream to perform all operations.
//! @retval HIPTENSOR_STATUS_NOT_SUPPORTED if the combination of data types or operations is not supported
//! @retval HIPTENSOR_STATUS_INVALID_VALUE if tensor dimensions or modes have an illegal value
//! @retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully without error
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
hiptensorStatus_t  hiptensorPermute(const hiptensorHandle_t handle, 
                                    const hiptensorPlan_t plan,
                                    const void* alpha, 
                                    const void* A,
                                    void* B, 
                                    const hipStream_t stream);

/**
 * \brief This function creates an operation descriptor for an elementwise binary operation.
 *
 * \details The binary operation has the following general form:
 * \f[ D_{\Pi^C(i_0,i_1,...,i_n)} = \Phi_{AC}(\alpha \Psi_A(A_{\Pi^A(i_0,i_1,...,i_n)}), \gamma \Psi_C(C_{\Pi^C(i_0,i_1,...,i_n)})) \f]
 *
 * Call \ref hiptensorElementwiseBinaryExecute to perform the actual operation.
 *
 * Supported data-type combinations are:
 *
 * \verbatim embed:rst:leading-asterisk
 * +-------------------+-------------------+------------------------------+
 * |     typeA         |     typeC         |  descCompute                 |
 * +===================+===================+==============================+
 * |  HIPTENSOR_R_16F   |  HIPTENSOR_R_16F   |  HIPTENSOR_COMPUTE_DESC_16F   |
 * +-------------------+-------------------+------------------------------+
 * |  HIPTENSOR_R_16F   |  HIPTENSOR_R_16F   |  HIPTENSOR_COMPUTE_DESC_32F   |
 * +-------------------+-------------------+------------------------------+
 * |  HIPTENSOR_R_16BF  |  HIPTENSOR_R_16BF  |  HIPTENSOR_COMPUTE_DESC_16BF  |
 * +-------------------+-------------------+------------------------------+
 * |  HIPTENSOR_R_16BF  |  HIPTENSOR_R_16BF  |  HIPTENSOR_COMPUTE_DESC_32F   |
 * +-------------------+-------------------+------------------------------+
 * |  HIPTENSOR_R_32F   |  HIPTENSOR_R_32F   |  HIPTENSOR_COMPUTE_DESC_32F   |
 * +-------------------+-------------------+------------------------------+
 * |  HIPTENSOR_R_64F   |  HIPTENSOR_R_64F   |  HIPTENSOR_COMPUTE_DESC_64F   |
 * +-------------------+-------------------+------------------------------+
 * |  HIPTENSOR_C_32F   |  HIPTENSOR_C_32F   |  HIPTENSOR_COMPUTE_DESC_32F   |
 * +-------------------+-------------------+------------------------------+
 * |  HIPTENSOR_C_64F   |  HIPTENSOR_C_64F   |  HIPTENSOR_COMPUTE_DESC_64F   |
 * +-------------------+-------------------+------------------------------+
 * |  HIPTENSOR_R_32F   |  HIPTENSOR_R_16F   |  HIPTENSOR_COMPUTE_DESC_32F   |
 * +-------------------+-------------------+------------------------------+
 * |  HIPTENSOR_R_64F   |  HIPTENSOR_R_32F   |  HIPTENSOR_COMPUTE_DESC_64F   |
 * +-------------------+-------------------+------------------------------+
 * |  HIPTENSOR_C_64F   |  HIPTENSOR_C_32F   |  HIPTENSOR_COMPUTE_DESC_64F   |
 * +-------------------+-------------------+------------------------------+
 * \endverbatim
 *
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[out] desc This opaque struct gets allocated and filled with the information that encodes the requested elementwise operation.
 * \param[in] descA The descriptor that holds the information about the data type, modes, and strides of A.
 * \param[in] modeA Array (in host memory) of size descA->numModes that holds the names of the modes of A (e.g., if A_{a,b,c} => modeA = {'a','b','c'}). The modeA[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to \ref hiptensorCreateTensorDescriptor.
 * \param[in] opA Unary operator that will be applied to each element of A before it is further processed. The original data of this tensor remains unchanged.
 * \param[in] descC The descriptor that holds information about the data type, modes, and strides of C.
 * \param[in] modeC Array (in host memory) of size descC->numModes that holds the names of the modes of C. The modeC[i] corresponds to extent[i] and stride[i] of the \ref hiptensorCreateTensorDescriptor.
 * \param[in] opC Unary operator that will be applied to each element of C before it is further processed. The original data of this tensor remains unchanged.
 * \param[in] descD The descriptor that holds information about the data type, modes, and strides of D. Notice that we currently request descD and descC to be identical.
 * \param[in] modeD Array (in host memory) of size descD->numModes that holds the names of the modes of D. The modeD[i] corresponds to extent[i] and stride[i] of the \ref hiptensorCreateTensorDescriptor.
 * \param[in] opAC Element-wise binary operator (see \f$\Phi_{AC}\f$ above).
 * \param[in] descCompute Determines the precision in which this operations is performed.
 * \retval HIPTENSOR_STATUS_NOT_SUPPORTED if the combination of data types or operations is not supported
 * \retval HIPTENSOR_STATUS_INVALID_VALUE if tensor dimensions or modes have an illegal value
 * \retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully without error
 * \retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \remarks calls asynchronous functions, no reentrant, and thread-safe
 */
hiptensorStatus_t hiptensorCreateElementwiseBinary(
                 const hiptensorHandle_t handle, hiptensorOperationDescriptor_t* desc,
                 const hiptensorTensorDescriptor_t descA, const int32_t modeA[], hiptensorOperator_t opA,
                 const hiptensorTensorDescriptor_t descC, const int32_t modeC[], hiptensorOperator_t opC,
                 const hiptensorTensorDescriptor_t descD, const int32_t modeD[],
                 hiptensorOperator_t opAC,
                 const hiptensorComputeDescriptor_t descCompute);

                                            
/**
 * \brief Performs an element-wise tensor operation for two input tensors (see \ref hiptensorCreateElementwiseBinary)
 *
 * \details This function performs a element-wise tensor operation of the form:
 * \f[ D_{\Pi^C(i_0,i_1,...,i_n)} = \Phi_{AC}(\alpha \Psi_A(A_{\Pi^A(i_0,i_1,...,i_n)}), \gamma \Psi_C(C_{\Pi^C(i_0,i_1,...,i_n)})) \f]
 *
 * See \ref hiptensorCreateElementwiseBinary() for details.
 *
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[in] plan Opaque handle holding all information about the desired elementwise operation (created by \ref hiptensorCreateElementwiseBinary followed by \ref hiptensorCreatePlan).
 * \param[in] alpha Scaling factor for A (see \ref hiptensorOperationDescriptorGetAttribute(desc, HIPTENSOR_OPERATION_SCALAR_TYPE) to query the expected data type). Pointer to the host memory. If alpha is zero, A is not read and the corresponding unary operator is not applied.
 * \param[in] A Multi-mode tensor (described by `descA` as part of \ref hiptensorCreateElementwiseBinary). Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to D.
 * \param[in] gamma Scaling factor for C (see \ref hiptensorOperationDescriptorGetAttribute(desc, HIPTENSOR_OPERATION_SCALAR_TYPE) to query the expected data type). Pointer to the host memory. If gamma is zero, C is not read and the corresponding unary operator is not applied.
 * \param[in] C Multi-mode tensor (described by `descC` as part of \ref hiptensorCreateElementwiseBinary). Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to D.
 * \param[out] D Multi-mode tensor (described by `descD` as part of \ref hiptensorCreateElementwiseBinary). Pointer to the GPU-accessible memory (`C` and `D` may be identical, if and only if `descC == descD`).
 * \param[in] stream The CUDA stream used to perform the operation.
 * \retval HIPTENSOR_STATUS_NOT_SUPPORTED if the combination of data types or operations is not supported
 * \retval HIPTENSOR_STATUS_INVALID_VALUE if tensor dimensions or modes have an illegal value
 * \retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully without error
 * \retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \remarks calls asynchronous functions, no reentrant, and thread-safe
 */
hiptensorStatus_t hiptensorElementwiseBinaryExecute(
                 const hiptensorHandle_t handle, const hiptensorPlan_t plan,
                 const void* alpha, const void* A,
                 const void* gamma, const void* C,
                                          void* D, hipStream_t stream);

/**
 * \brief This function creates an operation descriptor that encodes an elementwise trinary operation.
 *
 * \details Said trinary operation has the following general form:
 * \f[ D_{\Pi^C(i_0,i_1,...,i_n)} = \Phi_{ABC}(\Phi_{AB}(\alpha op_A(A_{\Pi^A(i_0,i_1,...,i_n)}), \beta op_B(B_{\Pi^B(i_0,i_1,...,i_n)})), \gamma op_C(C_{\Pi^C(i_0,i_1,...,i_n)})) \f]
 *
 * Where
 *    - A,B,C,D are multi-mode tensors (of arbitrary data types).
 *    - \f$\Pi^A, \Pi^B, \Pi^C \f$ are permutation operators that permute the modes of A, B, and C respectively.
 *    - \f$op_{A},op_{B},op_{C}\f$ are unary element-wise operators (e.g., IDENTITY, CONJUGATE).
 *    - \f$\Phi_{ABC}, \Phi_{AB}\f$ are binary element-wise operators (e.g., ADD, MUL, MAX, MIN).
 *
 * Notice that the broadcasting (of a mode) can be achieved by simply omitting that mode from the respective tensor.
 *
 * Moreover, modes may appear in any order, giving users a greater flexibility. The only <b>restrictions</b> are:
 *    - modes that appear in A or B _must_ also appear in the output tensor; a mode that only appears in the input would be contracted and such an operation would be covered by either \ref hiptensorContract or \ref hiptensorReduce.
 *    - each mode may appear in each tensor at most once.
 *
 * Input tensors may be read even if the value
 * of the corresponding scalar is zero.
 *
 * Examples:
 *    - \f$ D_{a,b,c,d} = A_{b,d,a,c}\f$
 *    - \f$ D_{a,b,c,d} = 2.2 * A_{b,d,a,c} + 1.3 * B_{c,b,d,a}\f$
 *    - \f$ D_{a,b,c,d} = 2.2 * A_{b,d,a,c} + 1.3 * B_{c,b,d,a} + C_{a,b,c,d}\f$
 *    - \f$ D_{a,b,c,d} = min((2.2 * A_{b,d,a,c} + 1.3 * B_{c,b,d,a}), C_{a,b,c,d})\f$
 *
 * Call \ref hiptensorElementwiseTrinaryExecute to perform the actual operation.
 *
 * Please use \ref hiptensorDestroyOperationDescriptor to deallocated the descriptor once it is no longer used.
 *
 * Supported data-type combinations are:
 *
 * \verbatim embed:rst:leading-asterisk
 * +-------------------+-------------------+-------------------+----------------------------+
 * |     typeA         |     typeB         |     typeC         |  descCompute               |
 * +===================+===================+===================+============================+
 * |  HIPTENSOR_R_16F   |  HIPTENSOR_R_16F   |  HIPTENSOR_R_16F   |  HIPTENSOR_COMPUTE_DESC_16F |
 * +-------------------+-------------------+-------------------+----------------------------+
 * |  HIPTENSOR_R_16F   |  HIPTENSOR_R_16F   |  HIPTENSOR_R_16F   |  HIPTENSOR_COMPUTE_DESC_32F |
 * +-------------------+-------------------+-------------------+----------------------------+
 * |  HIPTENSOR_R_16BF  |  HIPTENSOR_R_16BF  |  HIPTENSOR_R_16BF  |  HIPTENSOR_COMPUTE_DESC_16BF|
 * +-------------------+-------------------+-------------------+----------------------------+
 * |  HIPTENSOR_R_16BF  |  HIPTENSOR_R_16BF  |  HIPTENSOR_R_16BF  |  HIPTENSOR_COMPUTE_DESC_32F |
 * +-------------------+-------------------+-------------------+----------------------------+
 * |  HIPTENSOR_R_32F   |  HIPTENSOR_R_32F   |  HIPTENSOR_R_32F   |  HIPTENSOR_COMPUTE_DESC_32F |
 * +-------------------+-------------------+-------------------+----------------------------+
 * |  HIPTENSOR_R_64F   |  HIPTENSOR_R_64F   |  HIPTENSOR_R_64F   |  HIPTENSOR_COMPUTE_DESC_64F |
 * +-------------------+-------------------+-------------------+----------------------------+
 * |  HIPTENSOR_C_32F   |  HIPTENSOR_C_32F   |  HIPTENSOR_C_32F   |  HIPTENSOR_COMPUTE_DESC_32F |
 * +-------------------+-------------------+-------------------+----------------------------+
 * |  HIPTENSOR_C_64F   |  HIPTENSOR_C_64F   |  HIPTENSOR_C_64F   |  HIPTENSOR_COMPUTE_DESC_64F |
 * +-------------------+-------------------+-------------------+----------------------------+
 * |  HIPTENSOR_R_32F   |  HIPTENSOR_R_32F   |  HIPTENSOR_R_16F   |  HIPTENSOR_COMPUTE_DESC_32F |
 * +-------------------+-------------------+-------------------+----------------------------+
 * |  HIPTENSOR_R_64F   |  HIPTENSOR_R_64F   |  HIPTENSOR_R_32F   |  HIPTENSOR_COMPUTE_DESC_64F |
 * +-------------------+-------------------+-------------------+----------------------------+
 * |  HIPTENSOR_C_64F   |  HIPTENSOR_C_64F   |  HIPTENSOR_C_32F   |  HIPTENSOR_COMPUTE_DESC_64F |
 * +-------------------+-------------------+-------------------+----------------------------+
 * \endverbatim
 *
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[out] desc This opaque struct gets allocated and filled with the information that encodes the requested elementwise operation.
 * \param[in] descA A descriptor that holds the information about the data type, modes, and strides of A.
 * \param[in] modeA Array (in host memory) of size descA->numModes that holds the names of the modes of A (e.g., if \f$A_{a,b,c}\f$ then modeA = {'a','b','c'}). The modeA[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to \ref hiptensorCreateTensorDescriptor.
 * \param[in] opA Unary operator that will be applied to each element of A before it is further processed. The original data of this tensor remains unchanged.
 * \param[in] descB A descriptor that holds information about the data type, modes, and strides of B.
 * \param[in] modeB Array (in host memory) of size descB->numModes that holds the names of the modes of B. modeB[i] corresponds to extent[i] and stride[i] of the \ref hiptensorCreateTensorDescriptor
 * \param[in] opB Unary operator that will be applied to each element of B before it is further processed. The original data of this tensor remains unchanged.
 * \param[in] descC A descriptor that holds information about the data type, modes, and strides of C.
 * \param[in] modeC Array (in host memory) of size descC->numModes that holds the names of the modes of C. The modeC[i] corresponds to extent[i] and stride[i] of the \ref hiptensorCreateTensorDescriptor.
 * \param[in] opC Unary operator that will be applied to each element of C before it is further processed. The original data of this tensor remains unchanged.
 * \param[in] descD A descriptor that holds information about the data type, modes, and strides of D. Notice that we currently request descD and descC to be identical.
 * \param[in] modeD Array (in host memory) of size descD->numModes that holds the names of the modes of D. The modeD[i] corresponds to extent[i] and stride[i] of the \ref hiptensorCreateTensorDescriptor.
 * \param[in] opAB Element-wise binary operator (see \f$\Phi_{AB}\f$ above).
 * \param[in] opABC Element-wise binary operator (see \f$\Phi_{ABC}\f$ above).
 * \param[in] descCompute Determines the precision in which this operations is performed.
 * \retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \retval HIPTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
 * \retval HIPTENSOR_STATUS_ARCH_MISMATCH if the device is either not ready, or the target architecture is not supported.
 * \remarks calls asynchronous functions, no reentrant, and thread-safe
 */
hiptensorStatus_t hiptensorCreateElementwiseTrinary(
                 const hiptensorHandle_t handle, hiptensorOperationDescriptor_t* desc,
                 const hiptensorTensorDescriptor_t descA, const int32_t modeA[], hiptensorOperator_t opA,
                 const hiptensorTensorDescriptor_t descB, const int32_t modeB[], hiptensorOperator_t opB,
                 const hiptensorTensorDescriptor_t descC, const int32_t modeC[], hiptensorOperator_t opC,
                 const hiptensorTensorDescriptor_t descD, const int32_t modeD[],
                 hiptensorOperator_t opAB, hiptensorOperator_t opABC,
                 const hiptensorComputeDescriptor_t descCompute);

/**
 * \brief Performs an element-wise tensor operation for three input tensors (see \ref hiptensorCreateElementwiseTrinary)
 *
 * \details This function performs a element-wise tensor operation of the form:
 * \f[ D_{\Pi^C(i_0,i_1,...,i_n)} = \Phi_{ABC}(\Phi_{AB}(\alpha op_A(A_{\Pi^A(i_0,i_1,...,i_n)}), \beta op_B(B_{\Pi^B(i_0,i_1,...,i_n)})), \gamma op_C(C_{\Pi^C(i_0,i_1,...,i_n)})) \f]
 *
 * See \ref hiptensorCreateElementwiseTrinary() for details.
 *
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[in] plan Opaque handle holding all information about the desired elementwise operation (created by \ref hiptensorCreateElementwiseTrinary followed by \ref hiptensorCreatePlan).
 * \param[in] alpha Scaling factor for A (see \ref hiptensorOperationDescriptorGetAttribute(desc, HIPTENSOR_OPERATION_SCALAR_TYPE) to query the expected data type). Pointer to the host memory. If alpha is zero, A is not read and the corresponding unary operator is not applied.
 * \param[in] A Multi-mode tensor (described by `descA` as part of \ref hiptensorCreateElementwiseTrinary). Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to D.
 * \param[in] beta Scaling factor for B (see \ref hiptensorOperationDescriptorGetAttribute(desc, HIPTENSOR_OPERATION_SCALAR_TYPE) to query the expected data type). Pointer to the host memory. If beta is zero, B is not read and the corresponding unary operator is not applied.
 * \param[in] B Multi-mode tensor (described by `descB` as part of \ref hiptensorCreateElementwiseTrinary). Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to D.
 * \param[in] gamma Scaling factor for C (see \ref hiptensorOperationDescriptorGetAttribute(desc, HIPTENSOR_OPERATION_SCALAR_TYPE) to query the expected data type). Pointer to the host memory. If gamma is zero, C is not read and the corresponding unary operator is not applied.
 * \param[in] C Multi-mode tensor (described by `descC` as part of \ref hiptensorCreateElementwiseTrinary). Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to D.
 * \param[out] D Multi-mode tensor (described by `descD` as part of \ref hiptensorCreateElementwiseTrinary). Pointer to the GPU-accessible memory (`C` and `D` may be identical, if and only if `descC == descD`).
 * \param[in] stream The CUDA stream used to perform the operation.
 * \retval HIPTENSOR_STATUS_NOT_SUPPORTED if the combination of data types or operations is not supported
 * \retval HIPTENSOR_STATUS_INVALID_VALUE if tensor dimensions or modes have an illegal value
 * \retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully without error
 * \retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \remarks calls asynchronous functions, no reentrant, and thread-safe
 */
hiptensorStatus_t hiptensorElementwiseTrinaryExecute(
                 const hiptensorHandle_t handle, const hiptensorPlan_t plan,
                 const void* alpha, const void* A,
                 const void* beta,  const void* B,
                 const void* gamma, const void* C,
                                          void* D, hipStream_t stream);


//! @brief Computes the alignment requirement for a given pointer and descriptor.
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[in] ptr Pointer to the respective tensor data.
//! @param[in] desc Tensor descriptor for ptr data.
//! @param[out] alignmentRequirement Largest alignment requirement that ptr can fulfill (in bytes).
//! @retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully.
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
//! @retval HIPTENSOR_STATUS_INVALID_VALUE  if the unsupported parameter is passed.
hiptensorStatus_t hiptensorGetAlignmentRequirement(const hiptensorHandle_t           handle,
                                                   const void*                       ptr,
                                                   const hiptensorTensorDescriptor_t desc,
                                                   uint32_t* alignmentRequirement);

//! @brief Initializes a contraction descriptor for the tensor contraction problem.
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[out] desc Tensor contraction problem descriptor.
//! @param[in] descA A descriptor that holds information about tensor A.
//! @param[in] modeA Array with 'nmodeA' entries that represent the modes of A.
//! @param[in] alignmentRequirementA Alignment reqirement for A's pointer (in bytes);
//! @param[in] descB A descriptor that holds information about tensor B.
//! @param[in] modeB Array with 'nmodeB' entries that represent the modes of B.
//! @param[in] alignmentRequirementB Alignment reqirement for B's pointer (in bytes);
//! @param[in] modeC Array with 'nmodeC' entries that represent the modes of C.
//! @param[in] descC A descriptor that holds information about tensor C.
//! @param[in] alignmentRequirementC Alignment requirement for C's pointer (in bytes);
//! @param[in] modeD Array with 'nmodeD' entries that represent the modes of D (must be identical to modeC).
//! @param[in] descD A descriptor that holds information about tensor D (must be identical to descC).
//! @param[in] alignmentRequirementD Alignment requirement for D's pointer (in bytes);
//! @param[in] typeCompute Datatype for the intermediate computation  T = A * B.
//! @retval HIPTENSOR_STATUS_SUCCESS Successful completion of the operation.
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle or tensor descriptors are not initialized.
hiptensorStatus_t hiptensorInitContractionDescriptor(const hiptensorHandle_t           handle,
                                                     hiptensorContractionDescriptor_t* desc,
                                                     const hiptensorTensorDescriptor_t descA,
                                                     const int32_t                     modeA[],
                                                     const uint32_t alignmentRequirementA,
                                                     const hiptensorTensorDescriptor_t descB,
                                                     const int32_t                     modeB[],
                                                     const uint32_t alignmentRequirementB,
                                                     const hiptensorTensorDescriptor_t descC,
                                                     const int32_t                     modeC[],
                                                     const uint32_t alignmentRequirementC,
                                                     const hiptensorTensorDescriptor_t descD,
                                                     const int32_t                     modeD[],
                                                     const uint32_t alignmentRequirementD,
                                                     hiptensorComputeDescriptor_t typeCompute);

//! @brief Narrows down the candidates for the contraction problem.
//! @details This function gives the user finer control over the candidates that
//! the subsequent call to @ref hiptensorInitContractionPlan is allowed to
//! evaluate. Currently, the backend provides few set of algorithms(DEFAULT).
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[out] find Narrowed set of candidates for the contraction problem.
//! @param[in] algo Allows users to select a specific algorithm.
//! @retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully.
//! @retval HIPTENSOR_STATUS_NOT_SUPPORTED If a specified algorithm is not supported
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle or find is not initialized.
hiptensorStatus_t hiptensorInitContractionFind(const hiptensorHandle_t     handle,
                                               hiptensorContractionFind_t* find,
                                               const hiptensorAlgo_t       algo);

//! @brief Computes the size of workspace for a given tensor contraction
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[in] desc Tensor contraction descriptor.
//! @param[in] find Narrowed set of candidates for the contraction problem.
//! @param[in] pref Preference to choose the workspace size.
//! @param[out] workspaceSize Size of the workspace (in bytes).
//! @retval HIPTENSOR_STATUS_SUCCESS Successful completion of the operation.
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
//! @retval HIPTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
hiptensorStatus_t hiptensorContractionGetWorkspaceSize(const hiptensorHandle_t handle,
                                                       const hiptensorContractionDescriptor_t* desc,
                                                       const hiptensorContractionFind_t*       find,
                                                       const hiptensorWorksizePreference_t     pref,
                                                       uint64_t* workspaceSize);

//! @brief Initializes the contraction plan for a given tensor contraction problem
//! @details This function creates a contraction plan for the problem by applying
//! hipTensor's heuristics to select a candidate. The creaated plan can be reused
//! multiple times for the same tensor contraction problem. The plan is created for
//! the active HIP device.
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[out] plan Opaque handle holding the contraction plan (i.e.,
//! the algorithm that will be executed, its runtime parameters for the given
//! tensor contraction problem).
//! @param[in] desc Tensor contraction descriptor.
//! @param[in] find Narrows down the candidates for the contraction problem.
//! @param[in] workspaceSize Available workspace size (in bytes).
//! @retval HIPTENSOR_STATUS_SUCCESS If a viable candidate has been found.
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle or find or desc is not
//! initialized.
hiptensorStatus_t hiptensorInitContractionPlan(const hiptensorHandle_t                 handle,
                                               hiptensorContractionPlan_t*             plan,
                                               const hiptensorContractionDescriptor_t* desc,
                                               const hiptensorContractionFind_t*       find,
                                               const uint64_t workspaceSize);

//! @brief Computes the tensor contraction \f[ D = alpha * A * B + beta * C \f]
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! HIP Device associated with the handle must be same/active at the time,0
//! the plan was created.
//! @param[in] plan Opaque handle holding the contraction plan (i.e.,
//! the algorithm that will be executed, its runtime parameters for the given
//! tensor contraction problem).
//! @param[in] alpha Scaling parameter for A*B of data type 'typeCompute'.
//! @param[in] A Pointer to A's data in device memory.
//! @param[in] B Pointer to B's data in device memory.
//! @param[in] beta Scaling parameter for C of data type 'typeCompute'.
//! @param[in] C Pointer to C's data in device memory.
//! @param[out] D Pointer to D's data in device memory.
//! @param[out] workspace Workspace pointer in device memory
//! @param[in] workspaceSize Available workspace size.
//! @param[in] stream HIP stream to perform all operations.
//! @retval HIPTENSOR_STATUS_SUCCESS Successful completion of the operation.
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle or pointers are not
//! initialized.
//! @retval HIPTENSOR_STATUS_CK_ERROR if some unknown composable_kernel (CK)
//! error has occurred (e.g., no instance supported by inputs).
hiptensorStatus_t hiptensorContraction(const hiptensorHandle_t           handle,
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
 * \brief Creates a hiptensorOperatorDescriptor_t object that encodes a tensor reduction of the form \f$ D = alpha * opReduce(opA(A)) + beta * opC(C) \f$.
 *
 * \details
 * For example this function enables users to reduce an entire tensor to a scalar: C[] = alpha * A[i,j,k];
 *
 * This function is also able to perform partial reductions; for instance: C[i,j] = alpha * A[k,j,i]; in this case only elements along the k-mode are contracted.
 *
 * The binary opReduce operator provides extra control over what kind of a reduction
 * ought to be performed. For instance, setting opReduce to `HIPTENSOR_OP_ADD` reduces element of A
 * via a summation while `HIPTENSOR_OP_MAX` would find the largest element in A.
 *
 * Supported data-type combinations are:
 *
 * \verbatim embed:rst:leading-asterisk
 * +-------------------+-------------------+-------------------+-----------------------------+
 * |     typeA         |     typeB         |     typeC         |       typeCompute           |
 * +===================+===================+===================+=============================+
 * | `HIPTENSOR_R_16F`  | `HIPTENSOR_R_16F`  | `HIPTENSOR_R_16F`  | `HIPTENSOR_COMPUTE_DESC_16F` |
 * +-------------------+-------------------+-------------------+-----------------------------+
 * | `HIPTENSOR_R_16F`  | `HIPTENSOR_R_16F`  | `HIPTENSOR_R_16F`  | `HIPTENSOR_COMPUTE_DESC_32F` |
 * +-------------------+-------------------+-------------------+-----------------------------+
 * | `HIPTENSOR_R_16BF` | `HIPTENSOR_R_16BF` | `HIPTENSOR_R_16BF` | `HIPTENSOR_COMPUTE_DESC_16BF`|
 * +-------------------+-------------------+-------------------+-----------------------------+
 * | `HIPTENSOR_R_16BF` | `HIPTENSOR_R_16BF` | `HIPTENSOR_R_16BF` | `HIPTENSOR_COMPUTE_DESC_32F` |
 * +-------------------+-------------------+-------------------+-----------------------------+
 * | `HIPTENSOR_R_32F`  | `HIPTENSOR_R_32F`  | `HIPTENSOR_R_32F`  | `HIPTENSOR_COMPUTE_DESC_32F` |
 * +-------------------+-------------------+-------------------+-----------------------------+
 * | `HIPTENSOR_R_64F`  | `HIPTENSOR_R_64F`  | `HIPTENSOR_R_64F`  | `HIPTENSOR_COMPUTE_DESC_64F` |
 * +-------------------+-------------------+-------------------+-----------------------------+
 * | `HIPTENSOR_C_32F`  | `HIPTENSOR_C_32F`  | `HIPTENSOR_C_32F`  | `HIPTENSOR_COMPUTE_DESC_32F` |
 * +-------------------+-------------------+-------------------+-----------------------------+
 * | `HIPTENSOR_C_64F`  | `HIPTENSOR_C_64F`  | `HIPTENSOR_C_64F`  | `HIPTENSOR_COMPUTE_DESC_64F` |
 * +-------------------+-------------------+-------------------+-----------------------------+
 * \endverbatim
 *
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[out] desc This opaque struct gets allocated and filled with the information that encodes
 * the requested tensor reduction operation.
 * \param[in] descA The descriptor that holds the information about the data type, modes and strides of A.
 * \param[in] modeA Array with 'nmodeA' entries that represent the modes of A. modeA[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to \ref hiptensorCreateTensorDescriptor. Modes that only appear in modeA but not in modeC are reduced (contracted).
 * \param[in] opA Unary operator that will be applied to each element of A before it is further processed. The original data of this tensor remains unchanged.
 * \param[in] descC The descriptor that holds the information about the data type, modes and strides of C.
 * \param[in] modeC Array with 'nmodeC' entries that represent the modes of C. modeC[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to \ref hiptensorCreateTensorDescriptor.
 * \param[in] opC Unary operator that will be applied to each element of C before it is further processed. The original data of this tensor remains unchanged.
 * \param[in] descD Must be identical to descC for now.
 * \param[in] modeD Must be identical to modeC for now.
 * \param[in] opReduce binary operator used to reduce elements of A.
 * \param[in] typeCompute All arithmetic is performed using this data type (i.e., it affects the accuracy and performance).
 *
 * \retval HIPTENSOR_STATUS_NOT_SUPPORTED if operation is not supported.
 * \retval HIPTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
 * \retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 */
hiptensorStatus_t hiptensorCreateReduction(
                 const hiptensorHandle_t handle, hiptensorOperationDescriptor_t* desc,
                 const hiptensorTensorDescriptor_t descA, const int32_t modeA[], hiptensorOperator_t opA,
                 const hiptensorTensorDescriptor_t descC, const int32_t modeC[], hiptensorOperator_t opC,
                 const hiptensorTensorDescriptor_t descD, const int32_t modeD[],
                 hiptensorOperator_t opReduce, const hiptensorComputeDescriptor_t descCompute);

/**
 * \brief Performs the tensor reduction that is encoded by `plan` (see \ref hiptensorCreateReduction).
 *
 * \param[in] alpha Scaling for A. Its data type is determined by 'descCompute' (see \ref hiptensorOperationDescriptorGetAttribute(desc, CUTENSOR_OPERATION_SCALAR_TYPE)). Pointer to the host memory.
 * \param[in] A Pointer to the data corresponding to A in device memory. Pointer to the GPU-accessible memory. The data accessed via this pointer must not overlap with the elements written to D.
 * \param[in] beta Scaling for C. Its data type is determined by 'descCompute' (see \ref hiptensorOperationDescriptorGetAttribute(desc, CUTENSOR_OPERATION_SCALAR_TYPE)). Pointer to the host memory.
 * \param[in] C Pointer to the data corresponding to C in device memory. Pointer to the GPU-accessible memory.
 * \param[out] D Pointer to the data corresponding to C in device memory. Pointer to the GPU-accessible memory.
 * \param[out] workspace Scratchpad (device) memory of size --at least-- `workspaceSize` bytes; the workspace must be aligned to 256 bytes (i.e., the default alignment of cudaMalloc).
 * \param[in] workspaceSize Please use \ref hiptensorEstimateWorkspaceSize() to query the required workspace.
 * \param[in] stream The CUDA stream in which all the computation is performed.
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully.
 */
hiptensorStatus_t hiptensorReduce(
                 const hiptensorHandle_t handle, const hiptensorPlan_t plan,
                 const void* alpha, const void* A,
                 const void* beta,  const void* C,
                                          void* D,
                 void* workspace, uint64_t workspaceSize,
                 hipStream_t stream);                                     

//! @brief Determines the required workspaceSize for a given tensor reduction (see \ref hiptensorReduce)
//! @param[in] handle Opaque handle holding hipTensor's library context.
//! @param[in] A same as in hiptensorReduce
//! @param[in] descA same as in hiptensorReduce
//! @param[in] modeA same as in hiptensorReduce
//! @param[in] C same as in hiptensorReduce
//! @param[in] descC same as in hiptensorReduce
//! @param[in] modeC same as in hiptensorReduce
//! @param[in] D same as in hiptensorReduce
//! @param[in] descD same as in hiptensorReduce
//! @param[in] modeD same as in hiptensorReduce
//! @param[in] opReduce same as in hiptensorReduce
//! @param[in] typeCompute same as in hiptensorReduce
//! @param[out] workspaceSize The workspace size (in bytes) that is required for the given tensor reduction.
//! @retval HIPTENSOR_STATUS_SUCCESS The operation completed successfully.
//! @retval HIPTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
//! @retval HIPTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
hiptensorStatus_t hiptensorReductionGetWorkspaceSize(const hiptensorHandle_t           handle,
                                                     const void*                       A,
                                                     const hiptensorTensorDescriptor_t descA,
                                                     const int32_t                     modeA[],
                                                     const void*                       C,
                                                     const hiptensorTensorDescriptor_t descC,
                                                     const int32_t                     modeC[],
                                                     const void*                       D,
                                                     const hiptensorTensorDescriptor_t descD,
                                                     const int32_t                     modeD[],
                                                     hiptensorOperator_t               opReduce,
                                                     hiptensorComputeDescriptor_t      typeCompute,
                                                     uint64_t* workspaceSize);

//! @brief Registers a callback function that will be invoked by logger calls.
//! @param[in] callback This parameter is the callback function pointer provided to the logger.
//! @retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
//! @retval HIPTENSOR_STATUS_INVALID_VALUE if the given callback is invalid.
hiptensorStatus_t hiptensorLoggerSetCallback(hiptensorLoggerCallback_t callback);

//! @brief Registers a file output stream to redirect logging output to.
//! @note File stream must be open and writable in text mode.
//! @param[in] file This parameter is a file stream pointer provided to the logger.
//! @retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
//! @retval HIPTENSOR_STATUS_IO_ERROR if the output file is not valid (defaults back to stdout).
hiptensorStatus_t hiptensorLoggerSetFile(FILE* file);

//! @brief Redirects log output to a file given by the user.
//! @param[in] logFile This parameter is a file name (relative to binary) or full path to redirect logger output.
//! @retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
//! @retval HIPTENSOR_STATUS_IO_ERROR if the output file is not valid (defaults back to stdout).
hiptensorStatus_t hiptensorLoggerOpenFile(const char* logFile);

//! @brief User-specified logging level. Logs in other contexts will not be recorded.
//! @param[in] level This parameter is the logging level to be enforced.
//! @retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
//! @retval HIPTENSOR_STATUS_INVALID_VALUE if the given log level is invalid.
hiptensorStatus_t hiptensorLoggerSetLevel(hiptensorLogLevel_t level);

//! @brief User-specified logging mask. A mask may be a binary OR combination of
//! several log levels together. Logs in other contexts will not be recorded.
//! @param[in] mask This parameter is the logging mask to be enforced.
//! @retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
//! @retval HIPTENSOR_STATUS_INVALID_VALUE if the given log mask is invalid.
hiptensorStatus_t hiptensorLoggerSetMask(int32_t mask);

//! @brief Disables logging.
//! @retval HIPTENSOR_STATUS_SUCCESS if the operation completed successfully.
hiptensorStatus_t hiptensorLoggerForceDisable();

//! @brief Query HIP runtime version.
//! @retval -1 if the operation failed.
//! @retval Integer HIP runtime version if the operation succeeded.
int hiptensorGetHiprtVersion();

#endif // HIPTENSOR_API_HPP
