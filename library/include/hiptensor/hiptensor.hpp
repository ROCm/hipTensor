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

//! @brief Performs a tensor permutation operation.
//! @details This function computes the permutation operation: \f$B_{\Pi^B(i_0,i_1,...,i_n)} = \alpha \Psi(A_{\Pi^A(i_0,i_1,...,i_n)})\f$.
//! @param[in] handle An opaque handle representing the hipTensor library context.
//! @param[in] alpha The scaling factor for A, with data type `typeScalar`. Pointer to host memory. If alpha is zero, A is not read, and the unary operator is not applied.
//! @param[in] A A multi-mode tensor of type `typeA` with `nmodeA` modes. Pointer to GPU-accessible memory.
//! @param[in] descA A descriptor holding information about A's data type, modes, and strides.
//! @param[in] modeA An array of size `descA->numModes` holding the names of A's modes.
//! @param[in,out] B A multi-mode tensor of type `typeB` with `nmodeB` modes. Pointer to GPU-accessible memory.
//! @param[in] descB A descriptor holding information about B's data type, modes, and strides.
//! @param[in] modeB An array of size `descB->numModes` holding the names of B's modes.
//! @param[in] typeScalar The data type of `alpha`.
//! @param[in] stream The HIP stream to perform all operations.
//! @retval `HIPTENSOR_STATUS_NOT_SUPPORTED` if the combination of data types or operations is not supported.
//! @retval `HIPTENSOR_STATUS_INVALID_VALUE` if tensor dimensions or modes have an illegal value.
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completes successfully.
//! @retval `HIPTENSOR_STATUS_NOT_INITIALIZED` if the handle is not initialized.
hiptensorStatus_t hiptensorPermutation(const hiptensorHandle_t           handle,
                                       const void*                       alpha,
                                       const void*                       A,
                                       const hiptensorTensorDescriptor_t descA,
                                       const int32_t                     modeA[],
                                       void*                             B,
                                       const hiptensorTensorDescriptor_t descB,
                                       const int32_t                     modeB[],
                                       const hiptensorDataType_t         typeScalar,
                                       const hipStream_t                 stream);

//! @brief Performs an element-wise tensor operation on two input tensors.
//!
//! @details This function computes the element-wise operation: \f$D_{\Pi^C(i_0,i_1,...,i_n)} = \Phi_{AC}(\alpha \Psi_A(A_{\Pi^A(i_0,i_1,...,i_n)}), \gamma \Psi_C(C_{\Pi^C(i_0,i_1,...,i_n)}))\f$.
//! Where:
//!   - \f$D\f$ is the output tensor.
//!   - \f$A\f$ and \f$C\f$ are the input tensors.
//!   - \f$\alpha\f$ and \f$\gamma\f$ are scalar scaling factors.
//!   - \f$\Psi_A\f$ and \f$\Psi_C\f$ are unary operators (applied only if \f$\alpha\f$ and \f$\gamma\f$ are non-zero).
//!   - \f$\Phi_{AC}\f$ is a binary element-wise operator.
//!   - \f$\Pi^A\f$ and \f$\Pi^C\f$ represent mode permutations.
//!
//! @param[in] handle An opaque handle to the hipTensor library context.
//! @param[in] alpha Scaling factor for tensor A (host memory).
//! @param[in] A Input tensor A (GPU memory).
//! @param[in] descA Descriptor for tensor A, including data type, modes, and strides.
//! @param[in] modeA Array of mode names for tensor A (host memory).
//! @param[in] gamma Scaling factor for tensor C (host memory).
//! @param[in] C Input tensor C (GPU memory).
//! @param[in] descC Descriptor for tensor C, including data type, modes, and strides.
//! @param[in] modeC Array of mode names for tensor C (host memory).
//! @param[out] D Output tensor D (GPU memory).
//! @param[in] descD Descriptor for tensor D (must match descC).
//! @param[in] modeD Array of mode names for tensor D (host memory).
//! @param[in] opAC Element-wise binary operator \f$\Phi_{AC}\f$.
//! @param[in] typeScalar Scalar data type for intermediate computations.
//! @param[in] stream The HIP stream for execution.
//! @return `HIPTENSOR_STATUS_NOT_SUPPORTED` if data type or operation combination is unsupported.
//! @return `HIPTENSOR_STATUS_INVALID_VALUE` if tensor dimensions or modes are invalid.
//! @return `HIPTENSOR_STATUS_SUCCESS` if the operation completes successfully.
//! @return `HIPTENSOR_STATUS_NOT_INITIALIZED` if the handle is not initialized.
hiptensorStatus_t hiptensorElementwiseBinary(const hiptensorHandle_t           handle,
                                             const void*                       alpha,
                                             const void*                       A,
                                             const hiptensorTensorDescriptor_t descA,
                                             const int32_t                     modeA[],
                                             const void*                       gamma,
                                             const void*                       C,
                                             const hiptensorTensorDescriptor_t descC,
                                             const int32_t                     modeC[],
                                             void*                             D,
                                             const hiptensorTensorDescriptor_t descD,
                                             const int32_t                     modeD[],
                                             hiptensorOperator_t               opAC,
                                             hiptensorDataType_t               typeScalar,
                                             hipStream_t                       stream);

//! @brief Performs an element-wise tensor operation with three input tensors.
//!
//! @details This function computes the element-wise operation: \f$D_{\Pi^C(i_0,i_1,...,i_n)} = \Phi_{ABC}(\Phi_{AB}(\alpha \Psi_A(A_{\Pi^A(i_0,i_1,...,i_n)}), \beta \Psi_B(B_{\Pi^B(i_0,i_1,...,i_n)})), \gamma \Psi_C(C_{\Pi^C(i_0,i_1,...,i_n)}))\f$.
//!
//! Tensor modes can appear in any order, providing flexibility. However, the following restrictions apply:
//!   - Modes present in \f$A\f$ or \f$B\f$ must also be present in the output tensor \f$D\f$. Modes only in inputs would imply contraction, handled by `hiptensorContraction` or `hiptensorReduction`.
//!   - Each mode can appear at most once in each tensor.
//!
//! @param[in] handle An opaque handle to the hipTensor library context.
//! @param[in] alpha Scaling factor for tensor \f$A\f$ (host memory).
//! @param[in] A Input tensor \f$A\f$ (GPU memory).
//! @param[in] descA Descriptor for tensor \f$A\f$, including data type, modes, and strides.
//! @param[in] modeA Array of mode names for tensor \f$A\f$ (host memory).
//! @param[in] beta Scaling factor for tensor \f$B\f$ (host memory).
//! @param[in] B Input tensor \f$B\f$ (GPU memory).
//! @param[in] descB Descriptor for tensor \f$B\f$, including data type, modes, and strides.
//! @param[in] modeB Array of mode names for tensor \f$B\f$ (host memory).
//! @param[in] gamma Scaling factor for tensor \f$C\f$ (host memory).
//! @param[in] C Input tensor \f$C\f$ (GPU memory).
//! @param[in] descC Descriptor for tensor \f$C\f$, including data type, modes, and strides.
//! @param[in] modeC Array of mode names for tensor \f$C\f$ (host memory).
//! @param[out] D Output tensor \f$D\f$ (GPU memory). May alias input tensors if memory layouts match.
//! @param[in] descD Descriptor for tensor \f$D\f$ (must match descC).
//! @param[in] modeD Array of mode names for tensor \f$D\f$ (host memory).
//! @param[in] opAB Element-wise binary operator \f$\Phi_{AB}\f$.
//! @param[in] opABC Element-wise binary operator \f$\Phi_{ABC}\f$.
//! @param[in] typeScalar Data type for scalars alpha, beta, and gamma, and for intermediate computations.
//! @param[in] stream The HIP stream for execution.
//! @return `HIPTENSOR_STATUS_SUCCESS` if the operation completes successfully.
//! @return `HIPTENSOR_STATUS_NOT_INITIALIZED` if the handle is not initialized.
//! @return `HIPTENSOR_STATUS_INVALID_VALUE` if input data is invalid.
//! @return `HIPTENSOR_STATUS_ARCH_MISMATCH` if the device is not ready or the architecture is unsupported.
hiptensorStatus_t hiptensorElementwiseTrinary(const hiptensorHandle_t           handle,
                                              const void*                       alpha,
                                              const void*                       A,
                                              const hiptensorTensorDescriptor_t descA,
                                              const int32_t                     modeA[],
                                              const void*                       beta,
                                              const void*                       B,
                                              const hiptensorTensorDescriptor_t descB,
                                              const int32_t                     modeB[],
                                              const void*                       gamma,
                                              const void*                       C,
                                              const hiptensorTensorDescriptor_t descC,
                                              const int32_t                     modeC[],
                                              void*                             D,
                                              const hiptensorTensorDescriptor_t descD,
                                              const int32_t                     modeD[],
                                              hiptensorOperator_t               opAB,
                                              hiptensorOperator_t               opABC,
                                              hiptensorDataType_t               typeScalar,
                                              const hipStream_t                 stream);

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

//! @brief Implements a tensor reduction of the form \f$D = \alpha \cdot \text{opReduce}(\text{opA}(A)) + \beta \cdot \text{opC}(C) \f$.
//!
//! @param[in] handle An opaque handle representing the hipTensor library context.
//! @param[in] alpha Scaling factor for A; its data type is determined by `typeCompute`. Pointer to host memory.
//! @param[in] A Pointer to the data for tensor A in device memory. Pointer to GPU-accessible memory.
//! @param[in] descA A descriptor holding information about A's data type, modes, and strides.
//! @param[in] modeA An array with `nmodeA` entries representing the modes of A.
//! @param[in] beta Scaling factor for C; its data type is determined by `typeCompute`. Pointer to host memory.
//! @param[in] C Pointer to the data for tensor C in device memory. Pointer to GPU-accessible memory.
//! @param[in] descC A descriptor holding information about C's data type, modes, and strides.
//! @param[in] modeC An array with `nmodeC` entries representing the modes of C.
//! @param[out] D Pointer to the data for tensor C in device memory. Pointer to GPU-accessible memory.
//! @param[in] descD Currently must be identical to `descC`.
//! @param[in] modeD Currently must be identical to `modeC`.
//! @param[in] opReduce The binary operator used to reduce elements of A.
//! @param[in] typeCompute All arithmetic is performed using this data type, affecting accuracy and performance.
//! @param[out] workspace Scratchpad (device) memory.
//! @param[in] workspaceSize The size of the workspace array in bytes. Use `hiptensorReductionGetWorkspaceSize()` to query the required workspace. Lower values, including zero, are valid but may lead to suboptimal performance.
//! @param[in] stream The stream on which all computations are performed.
//! @retval `HIPTENSOR_STATUS_NOT_SUPPORTED` if the operation is not supported.
//! @retval `HIPTENSOR_STATUS_INVALID_VALUE` if some input data is invalid (typically a user error).
//! @retval `HIPTENSOR_STATUS_SUCCESS` if the operation completes successfully.
//! @retval `HIPTENSOR_STATUS_NOT_INITIALIZED` if the handle is not initialized.
hiptensorStatus_t hiptensorReduction(const hiptensorHandle_t           handle,
                                     const void*                       alpha,
                                     const void*                       A,
                                     const hiptensorTensorDescriptor_t descA,
                                     const int32_t                     modeA[],
                                     const void*                       beta,
                                     const void*                       C,
                                     const hiptensorTensorDescriptor_t descC,
                                     const int32_t                     modeC[],
                                     void*                             D,
                                     const hiptensorTensorDescriptor_t descD,
                                     const int32_t                     modeD[],
                                     hiptensorOperator_t               opReduce,
                                     hiptensorComputeDescriptor_t      typeCompute,
                                     void*                             workspace,
                                     uint64_t                          workspaceSize,
                                     hipStream_t                       stream);

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
