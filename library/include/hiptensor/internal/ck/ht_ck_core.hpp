#ifndef HT_CK_CORE_HPP_
#define HT_CK_CORE_HPP_

#include "../../ht_types.hpp"
#include "../ht_utility.hpp"

hiptensorStatus_t hiptensorCKScaleContraction(
    const hiptensorHandle_t *handle, const hiptensorContractionPlan_t *plan,
    hiptensorContractionMetrics_t *ht_contract_metrics, const void *alpha,
    const void *A, const void *B, const void *beta, const void *C, void *D,
    void *workspace, uint64_t workspaceSize, hipStream_t stream);

hiptensorStatus_t hiptensorCKBilinearContraction(
    const hiptensorHandle_t *handle, const hiptensorContractionPlan_t *plan,
    hiptensorContractionMetrics_t *ht_contract_metrics, const void *alpha,
    const void *A, const void *B, const void *beta, const void *C, void *D,
    void *workspace, uint64_t workspaceSize, hipStream_t stream);
/*
hiptensorStatus_t
hiptensorDeriveLayoutFromInputs(hiptensorContractionDescriptor_t* desc, const
int ndim);

hiptensorStatus_t hiptensorCKContraction(const hiptensorHandle_t* handle,
                                    const hiptensorContractionPlan_t* plan,
                                    hiptensorContractionMetrics_t
*ht_contract_metrics, const void* alpha, const void* A, const void* B, const
void* beta,  const void* C,       void* D, void *workspace, uint64_t
workspaceSize, hipStream_t stream);


hiptensorStatus_t hiptensorInitCKContractionFind(const hiptensorHandle_t*
handle, hiptensorContractionFind_t* find, const hiptensorAlgo_t algo);
*/
#endif
