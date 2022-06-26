#ifndef HT_TENSOR_HPP
#define HT_TENSOR_HPP

#include "ht_types.hpp"

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

hiptensorStatus_t hiptensorGetContractionMetrics( const hiptensorHandle_t* handle,
                                                const hiptensorContractionPlan_t* plan, 
				     	                        float* avgTime, float* flops, float* speed);


#endif
