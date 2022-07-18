#ifndef HT_TENSOR_HPP
#define HT_TENSOR_HPP

#include "ht_types.hpp"


/**
 * \brief Initializes the hipTENSOR library
 *
 * \details The device associated with a particular hipTENSOR handle is assumed to remain
 * unchanged after the hiptensorInit() call. In order for the cuTENSOR library to
 * use a different device, the application must set the new device to be used by
 * calling hipInit(0) and then create another hipTENSOR handle, which will
 * be associated with the new device, by calling hiptensorInit().
 *
 * \param[out] handle Pointer to hiptensorHandle_t
 *
 * \returns HIPTENSOR_STATUS_SUCCESS on success and an error code otherwise
 * \remarks blocking, no reentrant, and thread-safe
 */


/**
 * @brief Get a string representation of the function name.
 *
 * Again, a somewhat longer description with fancy notes etc.
 *
 * @param[out]  str  char* where string representation will be stored.
 *
 * @return 0 on success, or -1 if an error occurred.
 *
 * @pre Same as with @foo. Why?!
 * @post Yeah, really, why?!
 */

hiptensorStatus_t hiptensorInit(hiptensorHandle_t* handle);

hiptensorStatus_t hiptensorInitTensorDescriptor(const hiptensorHandle_t* handle,
                                            hiptensorTensorDescriptor_t* desc, const uint32_t numModes,
                                            const int64_t lens[], const int64_t strides[],
                                            hiptensorDataType_t dataType, hiptensorOperator_t unaryOp);

hiptensorStatus_t hiptensorGetAlignmentRequirement(const hiptensorHandle_t* handle,
                                                const void *ptr, const hiptensorTensorDescriptor_t* desc, 
                                                uint32_t* alignmentRequirement);

hiptensorStatus_t hiptensorInitContractionDescriptor(const hiptensorHandle_t* handle,
			                                    hiptensorContractionDescriptor_t* desc,
			                                    const hiptensorTensorDescriptor_t* descA, const int32_t modeA[], const uint32_t alignmentRequirementA,
			                                    const hiptensorTensorDescriptor_t* descB, const int32_t modeB[], const uint32_t alignmentRequirementB,
			                                    const hiptensorTensorDescriptor_t* descC, const int32_t modeC[], const uint32_t alignmentRequirementC,
			                                    const hiptensorTensorDescriptor_t* descD, const int32_t modeD[], const uint32_t alignmentRequirementD,
			                                    hiptensorComputeType_t typeCompute);

hiptensorStatus_t hiptensorInitContractionFind(const hiptensorHandle_t* handle,
                                             hiptensorContractionFind_t* find,
                                             const hiptensorAlgo_t algo);

hiptensorStatus_t hiptensorContractionGetWorkspace(const hiptensorHandle_t* handle,
                                                 const hiptensorContractionDescriptor_t* desc,
                                                 const hiptensorContractionFind_t* find,
                                                 const hiptensorWorksizePreference_t pref,
                                                 uint64_t *workspaceSize);

hiptensorStatus_t hiptensorInitContractionPlan(const hiptensorHandle_t* handle,
                                             hiptensorContractionPlan_t* plan,
                                             const hiptensorContractionDescriptor_t* desc,
                                             const hiptensorContractionFind_t* find,
                                             const uint64_t workspaceSize);

hiptensorStatus_t hiptensorContraction(const hiptensorHandle_t* handle,
			                        const hiptensorContractionPlan_t* plan,
			                        const void* alpha, const void* A, const void* B,
			                        const void* beta,  const void* C,       void* D,
			                        void *workspace, uint64_t workspaceSize, hipStream_t stream);

#endif
