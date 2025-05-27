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

//! @brief Allocates and initializes a hipTensor library handle.
//!
//! @details This function creates a hipTensor handle associated with the current device. To use a different device, call `hipInit(0)` to set the new device, then create another hipTensor handle with `hiptensorCreate()`.
//! @param[out] handle A pointer to the `hiptensorHandle_t` pointer that will store the newly created handle.
//! @returns `HIPTENSOR_STATUS_SUCCESS` if the handle is created successfully, otherwise an error code.
hiptensorStatus_t hiptensorCreate(hiptensorHandle_t* handle);

//! @brief Deallocates a hipTensor library handle.
//! @param[out] handle The `hiptensorHandle_t` to be deallocated.
//! @returns `HIPTENSOR_STATUS_SUCCESS` on successful deallocation, otherwise an error code.
hiptensorStatus_t hiptensorDestroy(hiptensorHandle_t handle);

//! @brief Resizes the plan cache associated with a hipTensor handle.
//! @param[in] handle The hipTensor handle.
//! @param[in] numEntries The new number of entries for the plan cache.
//! @returns `HIPTENSOR_STATUS_SUCCESS` on success, or an error code otherwise.
hiptensorStatus_t hiptensorHandleResizePlanCache(hiptensorHandle_t handle,
                                                 const uint32_t    numEntries);

//! @brief Writes the plan cache of a hipTensor handle to a file.
//! @param[in] handle The hipTensor handle whose plan cache will be written.
//! @param[in] filename The name of the file to write the cache to.
//! @returns `HIPTENSOR_STATUS_SUCCESS` on success, or an error code otherwise.
hiptensorStatus_t hiptensorHandleWritePlanCacheToFile(const hiptensorHandle_t handle,
                                                      const char              filename[]);

//! @brief Reads a plan cache from a file into a hipTensor handle.
//! @param[in] handle The hipTensor handle to populate with the plan cache.
//! @param[in] filename The name of the file to read the cache from.
//! @param[out] numCachelinesRead On exit, this variable will hold the number of successfully-read cachelines.
//! @returns `HIPTENSOR_STATUS_SUCCESS` on success, or an error code otherwise.
hiptensorStatus_t hiptensorHandleReadPlanCacheFromFile(hiptensorHandle_t handle,
                                                       const char        filename[],
                                                       uint32_t*         numCachelinesRead);

//! @brief Writes the kernel cache of a hipTensor handle to a file.
//! @param[in] handle The hipTensor handle whose kernel cache will be written.
//! @param[in] filename The name of the file to write the cache to.
//! @returns `HIPTENSOR_STATUS_SUCCESS` on success, or an error code otherwise.
hiptensorStatus_t hiptensorWriteKernelCacheToFile(const hiptensorHandle_t handle,
                                                  const char              filename[]);

//! @brief Reads a kernel cache from a file into a hipTensor handle.
//! @param[in] handle The hipTensor handle to populate with the kernel cache.
//! @param[in] filename The name of the file to read the cache from.
//! @returns `HIPTENSOR_STATUS_SUCCESS` on success, or an error code otherwise.
hiptensorStatus_t hiptensorReadKernelCacheFromFile(hiptensorHandle_t handle, const char filename[]);

//! @brief Creates and initializes a tensor descriptor.
//! @details This function allocates an instance of `hiptensorTensorDescriptor_t`. Call `hiptensorDestroyTensorDescriptor()` to free this instance.
//! @param[in] handle An opaque handle representing the hipTensor library context.
//! @param[out] desc A pointer to the `hiptensorTensorDescriptor_t` object to be allocated.
//! @param[in] numModes The number of modes (dimensions) for the tensor.
//! @param[in] lens An array specifying the extent (length) of each mode; all values must be greater than zero.
//! @param[in] strides An array where `strides[i]` is the displacement between consecutive elements in the i-th mode. If `NULL`, a generalized packed column-major memory layout is assumed (strides increase monotonically from left to right).
//! @param[in] dataType The data type of the tensor elements.
//! @param[in] alignmentRequirement The memory alignment requirement for the tensor.
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completes successfully.
//! @retval `HIPTENSOR_STATUS_NOT_INITIALIZED` if the handle is not initialized.
//! @retval `HIPTENSOR_STATUS_ARCH_MISMATCH` if the data type is not supported.
//! @retval `HIPTENSOR_STATUS_INVALID_VALUE` if any parameters are invalid.
hiptensorStatus_t hiptensorCreateTensorDescriptor(const hiptensorHandle_t      handle,
                                                  hiptensorTensorDescriptor_t* desc,
                                                  const uint32_t               numModes,
                                                  const int64_t                lens[],
                                                  const int64_t                strides[],
                                                  hiptensorDataType_t          dataType,
                                                  uint32_t alignmentRequirement);

//! @brief Destroys a tensor descriptor.
//!
//! @param[in] desc A pointer to the tensor descriptor object to be deallocated.
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completes successfully.
hiptensorStatus_t hiptensorDestroyTensorDescriptor(hiptensorTensorDescriptor_t desc);

//! @brief Allocates and initializes a `hiptensorOperationDescriptor` object for a tensor contraction of the form \f$D = \alpha \mathcal{A} \mathcal{B} + \beta \mathcal{C}\f$.
//!
//! @details Call `hiptensorDestroyOperationDescriptor()` to deallocate this object.
//! @param[in] handle An opaque handle representing the hipTensor library context.
//! @param[out] desc A pointer to the `hiptensorOperationDescriptor_t` object, which will be allocated and filled with contraction operation information.
//! @param[in] descA The tensor descriptor for tensor A, including data type, modes, and strides.
//! @param[in] modeA An array of `nmodeA` entries representing the modes of tensor A. `modeA[i]` corresponds to `extent[i]` and `stride[i]` from `hiptensorInitTensorDescriptor`.
//! @param[in] opA The unary operator to apply to each element of A before further processing. The original data of A remains unchanged.
//! @param[in] descB The tensor descriptor for tensor B.
//! @param[in] modeB An array of `nmodeB` entries representing the modes of tensor B.
//! @param[in] opB The unary operator to apply to each element of B.
//! @param[in] descC The tensor descriptor for tensor C.
//! @param[in] modeC An array of `nmodeC` entries representing the modes of tensor C.
//! @param[in] opC The unary operator to apply to each element of C.
//! @param[in] descD The tensor descriptor for tensor D (currently must be identical to `descC`).
//! @param[in] modeD An array of `nmodeD` entries representing the modes of tensor D (currently must be identical to `modeC`).
//! @param[in] descCompute The data type for the intermediate computation of \f$T = A * B\f$.
//! @retval `HIPTENSOR_STATUS_NOT_SUPPORTED` if the combination of data types or operations is not supported.
//! @retval `HIPTENSOR_STATUS_INVALID_VALUE` if tensor dimensions or modes have an illegal value.
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completes successfully.
//! @retval `HIPTENSOR_STATUS_NOT_INITIALIZED` if the handle is not initialized.
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

//! @brief Frees all resources associated with a `hiptensorOperationDescriptor` object.
//! @param[in,out] desc The `hiptensorOperationDescriptor_t` object to be deallocated.
//! @retval `HIPTENSOR_STATUS_SUCCESS` on success, otherwise an error code.
//! @remarks This function is blocking, not reentrant, and thread-safe.
hiptensorStatus_t hiptensorDestroyOperationDescriptor(hiptensorOperationDescriptor_t desc);

//! @brief Sets an attribute for a `hiptensorOperationDescriptor_t` object.
//! @param[in] handle An opaque handle representing the hipTensor library context.
//! @param[in] desc The `hiptensorOperationDescriptor_t` object to modify.
//! @param[in] attr The attribute to set.
//! @param[in] buf A pointer to the buffer containing the value for the attribute.
//! @param[in] sizeInBytes The size of `buf` in bytes.
//! @returns `HIPTENSOR_STATUS_SUCCESS` on success, or an error code otherwise.
hiptensorStatus_t
    hiptensorOperationDescriptorSetAttribute(const hiptensorHandle_t                 handle,
                                             hiptensorOperationDescriptor_t          desc,
                                             hiptensorOperationDescriptorAttribute_t attr,
                                             const void*                             buf,
                                             size_t                                  sizeInBytes);

//! @brief Retrieves an attribute from a `hiptensorOperationDescriptor_t` object.
//! @param[in] handle An opaque handle representing the hipTensor library context.
//! @param[in] desc The `hiptensorOperationDescriptor_t` object to query.
//! @param[in] attr The attribute to retrieve.
//! @param[out] buf A pointer to the buffer where the attribute value will be stored.
//! @param[in] sizeInBytes The size of `buf` in bytes.
//! @returns `HIPTENSOR_STATUS_SUCCESS` on success, or an error code otherwise.
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

//! @brief Allocates a `hiptensorPlanPreference_t` object, allowing users to restrict the kernels applicable for a plan/operation.
//! @param[in] handle An opaque handle representing the hipTensor library context.
//! @param[out] pref A pointer to the allocated `hiptensorPlanPreference_t` structure.
//! @param[in] algo Allows selection of a specific algorithm. `HIPTENSOR_ALGO_DEFAULT` (default) enables the heuristic to choose; any value `>= 0` selects a specific GEMM-like algorithm and disables the heuristic. If the specified algorithm is not supported, `HIPTENSOR_STATUS_NOT_SUPPORTED` is returned.
//! @param[in] jitMode Determines if hipTensor can use JIT-compiled kernels, which may lengthen the plan creation phase.
hiptensorStatus_t hiptensorCreatePlanPreference(const hiptensorHandle_t    handle,
                                                hiptensorPlanPreference_t* pref,
                                                hiptensorAlgo_t            algo,
                                                hiptensorJitMode_t         jitMode);

//! @brief Frees all resources related to the provided `hiptensorPlanPreference_t` object.
//! @param[in,out] pref The `hiptensorPlanPreference_t` object to be deallocated.
//! @retval `HIPTENSOR_STATUS_SUCCESS` on success, otherwise an error code.
//! @remarks This function is blocking, not reentrant, and thread-safe.
hiptensorStatus_t hiptensorDestroyPlanPreference(hiptensorPlanPreference_t pref);

//! @brief Sets an attribute for a `hiptensorPlanPreference_t` object.
//! @param[in] handle An opaque handle representing the hipTensor library context.
//! @param[in,out] pref The opaque structure that restricts the search space of viable kernel candidates.
//! @param[in] attr Specifies the attribute to be set.
//! @param[in] buf This buffer (of size `sizeInBytes`) contains the value to which `attr` will be set.
//! @param[in] sizeInBytes The size of `buf` in bytes.
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completes successfully.
//! @retval `HIPTENSOR_STATUS_NOT_INITIALIZED` if the handle is not initialized.
//! @retval `HIPTENSOR_STATUS_INVALID_VALUE` if some input data is invalid (typically a user error).
hiptensorStatus_t hiptensorPlanPreferenceSetAttribute(const hiptensorHandle_t            handle,
                                                      hiptensorPlanPreference_t          pref,
                                                      hiptensorPlanPreferenceAttribute_t attr,
                                                      const void*                        buf,
                                                      size_t sizeInBytes);

//! @brief Retrieves information about an already-created plan.
//! @param[in] handle An opaque handle representing the hipTensor library context.
//! @param[in] plan The already-created plan (e.g., from `hiptensorCreatePlan` or `hiptensorCreatePlanAutotuned`).
//! @param[in] attr The requested attribute.
//! @param[out] buf On successful exit, this buffer holds the information of the requested attribute.
//! @param[in] sizeInBytes The size of `buf` in bytes.
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completes successfully.
//! @retval `HIPTENSOR_STATUS_INVALID_VALUE` if some input data is invalid (typically a user error).
hiptensorStatus_t hiptensorPlanGetAttribute(const hiptensorHandle_t  handle,
                                            const hiptensorPlan_t    plan,
                                            hiptensorPlanAttribute_t attr,
                                            void*                    buf,
                                            size_t                   sizeInBytes);

//! @brief Determines the required workspace size for a given operation.
//! @param[in] handle An opaque handle representing the hipTensor library context.
//! @param[in] desc The opaque structure encoding the operation.
//! @param[in] planPref The opaque structure restricting the space of viable candidates.
//! @param[in] workspacePref This parameter influences the size of the workspace.
//! @param[out] workspaceSizeEstimate The estimated workspace size (in bytes) required for the operation.
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completes successfully.
//! @retval `HIPTENSOR_STATUS_NOT_INITIALIZED` if the handle is not initialized.
//! @retval `HIPTENSOR_STATUS_INVALID_VALUE` if some input data is invalid (typically a user error).
hiptensorStatus_t hiptensorEstimateWorkspaceSize(const hiptensorHandle_t              handle,
                                                 const hiptensorOperationDescriptor_t desc,
                                                 const hiptensorPlanPreference_t      planPref,
                                                 const hiptensorWorksizePreference_t  workspacePref,
                                                 uint64_t* workspaceSizeEstimate);

//! @brief Allocates a `hiptensorPlan_t` object, selects an appropriate kernel for a given operation, and prepares an execution plan.
//!
//! @details This function applies hipTensor's heuristic to select a kernel for an operation created by functions like `hiptensorCreateContraction`, `hiptensorCreateReduction`, `hiptensorCreatePermutation`, `hiptensorCreateElementwiseBinary`, or `hiptensorCreateElementwiseTrinary`. The resulting plan can then be passed to the corresponding `hiptensor*Execute` function to perform the actual operation.
//! The plan is created for the active HIP device.
//! Note: `hiptensorCreatePlan` must not be captured via HIP graphs if Just-In-Time compilation is enabled (i.e., `hiptensorJitMode_t` is not `HIPTENSOR_JIT_MODE_NONE`).
//! @param[in] handle An opaque handle representing the hipTensor library context.
//! @param[out] plan A pointer to the `hiptensorPlan_t` data structure created by this function, containing all necessary execution information (e.g., selected kernel).
//! @param[in] desc The opaque structure encoding the operation.
//! @param[in] pref An opaque structure used to restrict the space of applicable kernels. May be `nullptr` to use default choices.
//! @param[in] workspaceSizeLimit The maximum workspace size (in bytes) the operation is allowed to use.
//! @retval `HIPTENSOR_STATUS_SUCCESS` if a viable candidate kernel is found.
//! @retval `HIPTENSOR_STATUS_NOT_SUPPORTED` if no viable candidate kernel could be found.
//! @retval `HIPTENSOR_STATUS_NOT_INITIALIZED` if the handle is not initialized.
//! @retval `HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE` if the provided workspace was insufficient.
//! @retval `HIPTENSOR_STATUS_INVALID_VALUE` if some input data is invalid (typically a user error).
hiptensorStatus_t hiptensorCreatePlan(const hiptensorHandle_t              handle,
                                      hiptensorPlan_t*                     plan,
                                      const hiptensorOperationDescriptor_t desc,
                                      const hiptensorPlanPreference_t      pref,
                                      uint64_t                             workspaceSizeLimit);

//! @brief Frees all resources related to the provided plan.
//! @param[in,out] plan The `hiptensorPlan_t` object to be deallocated.
//! @retval `HIPTENSOR_STATUS_SUCCESS` on success, otherwise an error code.
//! @remarks This function is blocking, not reentrant, and thread-safe.
hiptensorStatus_t hiptensorDestroyPlan(hiptensorPlan_t plan);

//! @brief Computes the tensor contraction \f$D = \alpha \mathcal{A} \mathcal{B} + \beta \mathcal{C}\f$.
//!
//! @details The equation is given by: \f$\mathcal{D}_{{modes}_\mathcal{D}} \gets \alpha * \mathcal{A}_{{modes}_\mathcal{A}} B_{{modes}_\mathcal{B}} + \beta \mathcal{C}_{{modes}_\mathcal{C}}\f$.
//! The active HIP device must match the device that was active when the plan was created.
//! @param[in] handle An opaque handle representing the hipTensor library context.
//! @param[in] plan An opaque handle holding the contraction execution plan.
//! @param[in] alpha The scaling factor for \f$A*B\f$. Its data type is determined by `descCompute`. Pointer to host memory.
//! @param[in] A A pointer to the data for tensor A in GPU-accessible memory. This data must not overlap with elements written to D.
//! @param[in] B A pointer to the data for tensor B in GPU-accessible memory. This data must not overlap with elements written to D.
//! @param[in] beta The scaling factor for C. Its data type is determined by `descCompute`. Pointer to host memory.
//! @param[in] C A pointer to the data for tensor C in GPU-accessible memory.
//! @param[out] D A pointer to the data for tensor D in GPU-accessible memory.
//! @param[out] workspace An optional parameter (may be `NULL`). This pointer provides additional device memory workspace for library optimizations.
//! @param[in] workspaceSize The size of the `workspace` array in bytes.
//! @param[in] stream The HIP stream on which all computations are performed.
//! @retval `HIPTENSOR_STATUS_NOT_SUPPORTED` if the operation is not supported.
//! @retval `HIPTENSOR_STATUS_INVALID_VALUE` if some input data is invalid (typically a user error).
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completes successfully.
//! @retval `HIPTENSOR_STATUS_NOT_INITIALIZED` if the handle is not initialized.
//! @retval `HIPTENSOR_STATUS_ARCH_MISMATCH` if the plan was created for a different device than the currently active one.
//! @retval `HIPTENSOR_STATUS_INSUFFICIENT_DRIVER` if the driver is insufficient.
//! @retval `HIPTENSOR_STATUS_CUDA_ERROR` if an unknown HIP error occurs (e.g., out of memory).
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

//! @brief Returns a descriptive string for a given error code.
//! @param[in] error The error code to convert to a string.
//! @retval A string describing the error.
const char* hiptensorGetErrorString(const hiptensorStatus_t error);

//! @brief Returns the hipTensor library version.
//! @retval The hipTensor library version as a size_t.
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


//! @brief Computes the alignment requirement for a given pointer and tensor descriptor.
//! @param[in] handle An opaque handle representing the hipTensor library context.
//! @param[in] ptr A pointer to the respective tensor data.
//! @param[in] desc The tensor descriptor for the `ptr` data.
//! @param[out] alignmentRequirement The largest alignment requirement (in bytes) that `ptr` can fulfill.
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completes successfully.
//! @retval `HIPTENSOR_STATUS_NOT_INITIALIZED` if the handle is not initialized.
//! @retval `HIPTENSOR_STATUS_INVALID_VALUE` if an unsupported parameter is passed.
hiptensorStatus_t hiptensorGetAlignmentRequirement(const hiptensorHandle_t           handle,
                                                   const void*                       ptr,
                                                   const hiptensorTensorDescriptor_t desc,
                                                   uint32_t* alignmentRequirement);

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

//! @brief Determines the required workspace size for a tensor reduction.
//! @param[in] handle An opaque handle representing the hipTensor library context.
//! @param[in] A Same as in `hiptensorReduction`.
//! @param[in] descA Same as in `hiptensorReduction`.
//! @param[in] modeA Same as in `hiptensorReduction`.
//! @param[in] C Same as in `hiptensorReduction`.
//! @param[in] descC Same as in `hiptensorReduction`.
//! @param[in] modeC Same as in `hiptensorReduction`.
//! @param[in] D Same as in `hiptensorReduction`.
//! @param[in] descD Same as in `hiptensorReduction`.
//! @param[in] modeD Same as in `hiptensorReduction`.
//! @param[in] opReduce Same as in `hiptensorReduction`.
//! @param[in] typeCompute Same as in `hiptensorReduction`.
//! @param[out] workspaceSize The workspace size (in bytes) required for the tensor reduction.
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completes successfully.
//! @retval `HIPTENSOR_STATUS_NOT_INITIALIZED` if the handle is not initialized.
//! @retval `HIPTENSOR_STATUS_INVALID_VALUE` if some input data is invalid (typically a user error).
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

//! @brief Registers a callback function to be invoked by logger calls.
//! @param[in] callback The callback function pointer to provide to the logger.
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completed successfully.
//! @retval `HIPTENSOR_STATUS_INVALID_VALUE` if the given callback is invalid.
hiptensorStatus_t hiptensorLoggerSetCallback(hiptensorLoggerCallback_t callback);

//! @brief Registers a file output stream to redirect logging output.
//! @note The file stream must be open and writable in text mode.
//! @param[in] file A file stream pointer to provide to the logger.
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completed successfully.
//! @retval `HIPTENSOR_STATUS_IO_ERROR` if the output file is not valid (defaults back to stdout).
hiptensorStatus_t hiptensorLoggerSetFile(FILE* file);

//! @brief Redirects log output to a user-specified file.
//! @param[in] logFile The file name (relative to binary) or full path to redirect logger output.
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completed successfully.
//! @retval `HIPTENSOR_STATUS_IO_ERROR` if the output file is not valid (defaults back to stdout).
hiptensorStatus_t hiptensorLoggerOpenFile(const char* logFile);

//! @brief Sets the user-specified logging level. Logs in other contexts will not be recorded.
//! @param[in] level The logging level to enforce.
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completed successfully.
//! @retval `HIPTENSOR_STATUS_INVALID_VALUE` if the given log level is invalid.
hiptensorStatus_t hiptensorLoggerSetLevel(hiptensorLogLevel_t level);

//! @brief Sets the user-specified logging mask. A mask can be a binary OR combination of several log levels. Logs in other contexts will not be recorded.
//! @param[in] mask The logging mask to enforce.
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completed successfully.
//! @retval `HIPTENSOR_STATUS_INVALID_VALUE` if the given log mask is invalid.
hiptensorStatus_t hiptensorLoggerSetMask(int32_t mask);

//! @brief Disables logging.
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completed successfully.
hiptensorStatus_t hiptensorLoggerForceDisable();

//! @brief Queries the HIP runtime version.
//! @retval -1 if the operation failed.
//! @retval An integer representing the HIP runtime version if the operation succeeded.
int hiptensorGetHiprtVersion();

#endif // HIPTENSOR_API_HPP
