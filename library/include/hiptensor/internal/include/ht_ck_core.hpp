#ifndef HT_CK_CORE_HPP_
#define HT_CK_CORE_HPP_

#include "../ht_types.hpp"


hipTensorStatus_t hipTensorCKScaleContraction(const hipTensorHandle_t* handle,
                                    const hipTensorContractionPlan_t* plan,
                                     hipTensorContractionMetrics_t *ht_contract_metrics,
                                    const void* alpha, const void* A, const void* B,
                                    const void* beta,  const void* C,       void* D,
                                    void *workspace, uint64_t workspaceSize, hipStream_t stream);

hipTensorStatus_t hipTensorCKBilinearContraction(const hipTensorHandle_t* handle,
                                    const hipTensorContractionPlan_t* plan,
                                    hipTensorContractionMetrics_t *ht_contract_metrics,
                                    const void* alpha, const void* A, const void* B,
                                    const void* beta,  const void* C,       void* D,
                                    void *workspace, uint64_t workspaceSize, hipStream_t stream);
/*
hipTensorStatus_t hipTensorDeriveLayoutFromInputs(hipTensorContractionDescriptor_t* desc, const int ndim);

hipTensorStatus_t hipTensorCKContraction(const hipTensorHandle_t* handle,
                                    const hipTensorContractionPlan_t* plan,
                                    hipTensorContractionMetrics_t *ht_contract_metrics,
                                    const void* alpha, const void* A, const void* B,
                                    const void* beta,  const void* C,       void* D,
                                    void *workspace, uint64_t workspaceSize, hipStream_t stream);


hipTensorStatus_t hipTensorInitCKContractionFind(const hipTensorHandle_t* handle,
                                             hipTensorContractionFind_t* find,
                                             const hipTensorAlgo_t algo);
*/
#endif
