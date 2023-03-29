#include "ht_tensor.hpp"
#include "ht_ck_core.hpp"

hipTensorContractionMetrics_t ht_contract_metrics;

hipTensorStatus_t hipTensorInitContractionDescriptor(const hipTensorHandle_t* handle,
                                                    hipTensorContractionDescriptor_t* desc,
                                                    const hipTensorTensorDescriptor_t* descA, const int32_t modeA[], const uint32_t alignmentRequirementA,
                                                    const hipTensorTensorDescriptor_t* descB, const int32_t modeB[], const uint32_t alignmentRequirementB,
                                                    const hipTensorTensorDescriptor_t* descC, const int32_t modeC[], const uint32_t alignmentRequirementC,
                                                    const hipTensorTensorDescriptor_t* descD, const int32_t modeD[], const uint32_t alignmentRequirementD,
                                                    hipTensorComputeType_t typeCompute)

{
    if (!handle || !desc || !descA || !descB || !descC )
        return hipTensor_STATUS_NOT_INITIALIZED;

    const hipTensorTensorDescriptor_t *ht_input_descs[] = { descA, descB, descC };
    const uint32_t alignmentRequriement_arr[] = {alignmentRequirementA, alignmentRequirementB, alignmentRequirementC};
    desc->hipTensorContractionAttrUpdate(ht_input_descs, alignmentRequriement_arr, 3);
    
    if ( !descD )
        desc->ht_contract_op = hipTensor_CONTRACTION_SCALE;
    else
        desc->ht_contract_op = hipTensor_CONTRACTION_BILINEAR;

    return hipTensor_STATUS_SUCCESS;
}


hipTensorStatus_t hipTensorInitContractionFind(const hipTensorHandle_t* handle,
                                             hipTensorContractionFind_t* find,
                                             const hipTensorAlgo_t algo)
{
    if (!handle || !find)
        return hipTensor_STATUS_NOT_INITIALIZED;
    
    if (algo != hipTensor_ALGO_DEFAULT)
    {
        std::cout << "CK algorithm not supported" << std::endl;
        return hipTensor_STATUS_INTERNAL_ERROR;
    }

    return hipTensor_STATUS_SUCCESS;
}


hipTensorStatus_t hipTensorContractionGetWorkspace(const hipTensorHandle_t* handle,
                                                 const hipTensorContractionDescriptor_t* desc,
                                                 const hipTensorContractionFind_t* find,
                                                 const hipTensorWorksizePreference_t pref,
                                                 uint64_t *workspaceSize) 
{
    return hipTensor_STATUS_SUCCESS;
}

hipTensorStatus_t hipTensorInitContractionPlan(const hipTensorHandle_t* handle,
                                            hipTensorContractionPlan_t* plan, const hipTensorContractionDescriptor_t* desc,
                                            const hipTensorContractionFind_t* find,
                                            const uint64_t workspaceSize) 
{
    if (!handle || !plan || !desc)
        return hipTensor_STATUS_NOT_INITIALIZED;

    plan->ht_plan_desc = *desc;
    return hipTensor_STATUS_SUCCESS;
}

hipTensorStatus_t hipTensorContraction(const hipTensorHandle_t* handle,
                                     const hipTensorContractionPlan_t* plan,
                                     const void* alpha, const void* A, const void* B,
                                     const void* beta,  const void* C,       void* D,
                                     void *workspace, uint64_t workspaceSize, hipStream_t stream)
{
    if (!handle || !A || !B || !D)
	    return hipTensor_STATUS_NOT_INITIALIZED;

	if ( plan->ht_plan_desc.ht_contract_op == hipTensor_CONTRACTION_SCALE )
	{
		hipTensorCKScaleContraction( handle, plan, &ht_contract_metrics, alpha, A, B,
								NULL, NULL, D, workspace, workspaceSize, stream );
	}
	else if ( plan->ht_plan_desc.ht_contract_op == hipTensor_CONTRACTION_BILINEAR )
	{
		hipTensorCKBilinearContraction( handle, plan, &ht_contract_metrics, alpha, A, B,
								beta, C, D, workspace, workspaceSize, stream );
	}
	else
	{
		std::cout << "Contraction operation not permitted" << std::endl;
        return hipTensor_STATUS_CK_ERROR;	
	}
    return hipTensor_STATUS_SUCCESS;
}


void hipTensorContractionPlan_t:: hipTensorPrintContractionMetrics()
{
    std::cout << "Perf: " << ht_contract_metrics.avg_time << " ms, " <<  ht_contract_metrics.tflops << " TFlops, " 
              << ht_contract_metrics.transfer_speed << " GB/s, " << ht_contract_metrics.ht_instance << std::endl;
}
