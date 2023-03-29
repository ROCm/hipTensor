#ifndef HT_TENSOR_HPP
#define HT_TENSOR_HPP

#include "ht_types.hpp"
#include "../utility/ht_utility.hpp"

/**
 * \brief Initializes the hipTensor library
 *
 * \details The device associated with a particular hipTensor handle is assumed to remain
 * unchanged after the hipTensorInit() call. In order for the hipTensor library to
 * use a different device, the application must set the new device to be used by
 * calling hipInit(0) and then create another hipTensor handle, which will
 * be associated with the new device, by calling hipTensorInit().
 *
 * \param[out] handle Pointer to hipTensorHandle_t
 *
 * \returns hipTensor_STATUS_SUCCESS on success and an error code otherwise
 * \remarks blocking, no reentrant, and thread-safe
 */

hipTensorStatus_t hipTensorInit(hipTensorHandle_t* handle);

/**
 * \brief Initializes a tensor descriptor
 *
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[out] desc Pointer to the address where the allocated tensor descriptor object is stored.
 * \param[in] numModes Number of modes/dimensions.
 * \param[in] lens Extent of each mode(lengths) (must be larger than zero).
 * \param[in] strides stride[i] denotes the displacement (stride) between two consecutive elements in the ith-mode.
 *            If stride is NULL, a packed generalized column-major memory
 *            layout is assumed (i.e., the strides increase monotonically from left to right)
 * \param[in] dataType Data type of the stored entries.
 * \param[in] unaryOp Unary operator that will be applied to each element of the corresponding
 *            tensor in a lazy fashion (i.e., the algorithm uses this tensor as its operand only once).
 *            The original data of this tensor remains unchanged.
 * \pre extent and stride arrays must each contain at least sizeof(int64_t) * numModes bytes
 * \retval hipTensor_STATUS_SUCCESS The operation completed successfully.
 * \retval hipTensor_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \remarks non-blocking, no reentrant, and thread-safe
 */

hipTensorStatus_t hipTensorInitTensorDescriptor(const hipTensorHandle_t* handle,
                                            hipTensorTensorDescriptor_t* desc, const uint32_t numModes,
                                            const int64_t lens[], const int64_t strides[],
                                            hipTensorDataType_t dataType, hipTensorOperator_t unaryOp);
/**
 * \brief Computes the minimal alignment requirement for a given pointer and descriptor
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[in] ptr Raw pointer to the data of the respective tensor.
 * \param[in] desc Tensor descriptor for ptr.
 * \param[out] alignmentRequirement Largest alignment requirement that ptr can fulfill (in bytes).
 * \retval hipTensor_STATUS_SUCCESS The operation completed successfully.
 * \retval hipTensor_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \retval hipTensor_STATUS_INVALID_VALUE  if the unsupported datatype is passed.
 */

hipTensorStatus_t hipTensorGetAlignmentRequirement(const hipTensorHandle_t* handle,
                                                const void *ptr, const hipTensorTensorDescriptor_t* desc, 
                                                uint32_t* alignmentRequirement);

/**
 * \brief Describes the tensor contraction problem of the form: \f[ D = \alpha \mathcal{A}  \mathcal{B} + \beta \mathcal{C} \f]
 *
 * \details \f[ \mathcal{D}_{{modes}_\mathcal{D}} \gets \alpha \mathcal{A}_{{modes}_\mathcal{A}} B_{{modes}_\mathcal{B}} + \beta \mathcal{C}_{{modes}_\mathcal{C}} \f].
 *
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[out] desc This opaque struct gets filled with the information that encodes
 * the tensor contraction problem.
 * \param[in] descA A descriptor that holds the information about the data type, modes and strides of A.
 * \param[in] modeA Array with 'nmodeA' entries that represent the modes of A. The modeA[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to hipTensorInitTensorDescriptor.
 * \param[in] alignmentRequirementA Alignment that hipTensor may require for A's pointer (in bytes); you
 * can use the helper function \ref hipTensorGetAlignmentRequirement to determine the best value for a given pointer.
 * \param[in] descB The B descriptor that holds information about the data type, modes, and strides of B.
 * \param[in] modeB Array with 'nmodeB' entries that represent the modes of B. The modeB[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to hipTensorInitTensorDescriptor.
 * \param[in] alignmentRequirementB Alignment that hipTensor may require for B's pointer (in bytes); you
 * can use the helper function \ref hipTensorGetAlignmentRequirement to determine the best value for a given pointer.
 * \param[in] modeC Array with 'nmodeC' entries that represent the modes of C. The modeC[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to hipTensorInitTensorDescriptor.
 * \param[in] descC The C descriptor that holds information about the data type, modes, and strides of C.
 * \param[in] alignmentRequirementC Alignment that hipTensor may require for C's pointer (in bytes); you
 * can use the helper function \ref hipTensorGetAlignmentRequirement to determine the best value for a given pointer.
 * \param[in] modeD Array with 'nmodeD' entries that represent the modes of D (must be identical to modeC for now). The modeD[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to hipTensorInitTensorDescriptor.
 * \param[in] descD The D descriptor that holds information about the data type, modes, and strides of D (must be identical to descC for now).
 * \param[in] alignmentRequirementD Alignment that hipTensor may require for D's pointer (in bytes); you
 * can use the helper function \ref hipTensorGetAlignmentRequirement to determine the best value for a given pointer.
 * \param[in] typeCompute Datatype of for the intermediate computation of typeCompute T = A * B.
 * \retval hipTensor_STATUS_SUCCESS The operation completed successfully without error
 * \retval hipTensor_STATUS_NOT_INITIALIZED if the handle or tensor descriptors are not initialized.
 */

hipTensorStatus_t hipTensorInitContractionDescriptor(const hipTensorHandle_t* handle,
			                                    hipTensorContractionDescriptor_t* desc,
			                                    const hipTensorTensorDescriptor_t* descA, const int32_t modeA[], const uint32_t alignmentRequirementA,
			                                    const hipTensorTensorDescriptor_t* descB, const int32_t modeB[], const uint32_t alignmentRequirementB,
			                                    const hipTensorTensorDescriptor_t* descC, const int32_t modeC[], const uint32_t alignmentRequirementC,
			                                    const hipTensorTensorDescriptor_t* descD, const int32_t modeD[], const uint32_t alignmentRequirementD,
			                                    hipTensorComputeType_t typeCompute);

/**
 * \brief Limits the search space of viable candidates (a.k.a. algorithms)
 *
 * \details This function gives the user finer control over the candidates that the subsequent call to \ref hipTensorInitContractionPlan
 * is allowed to evaluate. Currently, the backend provides only one set of algorithms. Need to adapt for the future if multiple 
 * set of algorithms (based on different accelerators) are available.
 *
 *
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[out] find
 * \param[in] algo Allows users to select a specific algorithm. hipTensor_ALGO_DEFAULT only supprted by the CK backend.
 * \retval hipTensor_STATUS_SUCCESS The operation completed successfully.
 * \retval hipTensor_STATUS_NOT_SUPPORTED If a specified algorithm is not supported hipTensor_STATUS_NOT_SUPPORTED is returned.
 * \retval hipTensor_STATUS_NOT_INITIALIZED if the handle or find is not initialized.
 */

hipTensorStatus_t hipTensorInitContractionFind(const hipTensorHandle_t* handle,
                                             hipTensorContractionFind_t* find,
                                             const hipTensorAlgo_t algo);

/*TODO: Stub API. Not in use as per CK backend.Need to adapt based on the future implementations
 */
hipTensorStatus_t hipTensorContractionGetWorkspace(const hipTensorHandle_t* handle,
                                                 const hipTensorContractionDescriptor_t* desc,
                                                 const hipTensorContractionFind_t* find,
                                                 const hipTensorWorksizePreference_t pref,
                                                 uint64_t *workspaceSize);

/**
 * \brief Initializes the contraction plan for a given tensor contraction problem
 *
 * \details This function applies hipTensor's heuristic to select a candidate for a
 * given tensor contraction problem (encoded by desc). The resulting plan can be reused
 * multiple times as long as the tensor contraction problem remains the same.
 *
 * The plan is created for the active HIP device.
 *
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[out] plan Opaque handle holding the contraction execution plan (i.e., the
 * candidate that will be executed as well as all it's runtime parameters for the given
 * tensor contraction problem).
 * \param[in] desc This opaque struct encodes the given tensor contraction problem.
 * \param[in] find (unused) This opaque struct is used to restrict the search space of viable candidates.
 * \param[in] workspaceSize (unused) Available workspace size (in bytes).
 *
 * \retval hipTensor_STATUS_SUCCESS If a viable candidate has been found.
 * \retval hipTensor_STATUS_NOT_INITIALIZED if the handle or find or desc is not initialized.
 */

hipTensorStatus_t hipTensorInitContractionPlan(const hipTensorHandle_t* handle,
                                             hipTensorContractionPlan_t* plan,
                                             const hipTensorContractionDescriptor_t* desc,
                                             const hipTensorContractionFind_t* find,
                                             const uint64_t workspaceSize);
/**
 * \brief This routine computes the tensor contraction \f[ D = alpha * A * B + beta * C \f]
 *
 * \details \f[ \mathcal{D}_{{modes}_\mathcal{D}} \gets \alpha * \mathcal{A}_{{modes}_\mathcal{A}} B_{{modes}_\mathcal{B}} + \beta \mathcal{C}_{{modes}_\mathcal{C}} \f]
 *
 * The currently active HIP device must match the HIP device that was active at the time at which the plan was created.
 * \param[in] handle Opaque handle holding hipTensor's library context.
 * \param[in] plan Opaque handle holding the contraction execution plan.
 * \param[in] alpha Scaling for A*B. Its data type is determined by 'typeCompute'. Pointer to the host memory.
 * \param[in] A Pointer to the data corresponding to A in device memory. Pointer to the GPU-accessible memory.
 * \param[in] B Pointer to the data corresponding to B. Pointer to the GPU-accessible memory.
 * \param[in] beta Scaling for C. Its data type is determined by 'typeCompute'. Pointer to the host memory.
 * \param[in] C Pointer to the data corresponding to C. Pointer to the GPU-accessible memory.
 * \param[out] D Pointer to the data corresponding to D. Pointer to the GPU-accessible memory.
 * \param[out] workspace (unused in this context)
 * \param[in] workspaceSize (unused in this context)
 * \param[in] stream The HIP stream in which all the computation is performed.
 *
 * Supported data-type combinations are:
 *
 * \verbatim embed:rst:leading-asterisk
 * +---------------+---------------+---------------+-------------------------+
 * |     typeA     |     typeB     |     typeC     |        typeCompute      |
 * +===============+===============+===============+=========================+
 * |    HIP_R_32F  |    HIP_R_32F  |   HIP_R_32F   |    HIPENSOR_COMPUTE_32F |
 * +---------------+---------------+---------------+-------------------------+
 * \endverbatim

 * \par[Example]
 * See https://github.com/AMD-HPC/hipTensor/blob/develop/test/01_contraction/test_bilinear_contraction_xdl_fp32.cpp \n
 *     https://github.com/AMD-HPC/hipTensor/blob/develop/test/01_contraction/test_scale_contraction_xdl_fp32.cpp 
 * for the concrete examples.
 *
 * \retval hipTensor_STATUS_SUCCESS The operation completed successfully.
 * \retval hipTensor_STATUS_NOT_INITIALIZED if the handle or pointers are not initialized.
 * \retval hipTensor_STATUS_CK_ERROR if some unknown composable_kernel (CK) error has occurred (e.g., no instance supported by inputs).
 */

hipTensorStatus_t hipTensorContraction(const hipTensorHandle_t* handle,
			                        const hipTensorContractionPlan_t* plan,
			                        const void* alpha, const void* A, const void* B,
			                        const void* beta,  const void* C,       void* D,
			                        void *workspace, uint64_t workspaceSize, hipStream_t stream);

#endif
