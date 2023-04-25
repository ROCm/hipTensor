/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2023 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "ht_ck_core.hpp"
#include "ht_tensor.hpp"

hiptensorContractionMetrics_t ht_contract_metrics;

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
                                                     hiptensorComputeType_t typeCompute)

{
    if(!handle || !desc || !descA || !descB || !descC)
        return HIPTENSOR_STATUS_NOT_INITIALIZED;

    const hiptensorTensorDescriptor_t* ht_input_descs[] = {descA, descB, descC};
    const uint32_t                     alignmentRequriement_arr[]
        = {alignmentRequirementA, alignmentRequirementB, alignmentRequirementC};
    desc->hiptensorContractionAttrUpdate(ht_input_descs, alignmentRequriement_arr, 3);

    if(!descD)
        desc->ht_contract_op = HIPTENSOR_CONTRACTION_SCALE;
    else
        desc->ht_contract_op = HIPTENSOR_CONTRACTION_BILINEAR;

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorInitContractionFind(const hiptensorHandle_t*    handle,
                                               hiptensorContractionFind_t* find,
                                               const hiptensorAlgo_t       algo)
{
    if(!handle || !find)
        return HIPTENSOR_STATUS_NOT_INITIALIZED;

    if(algo != HIPTENSOR_ALGO_DEFAULT)
    {
        std::cout << "CK algorithm not supported" << std::endl;
        return HIPTENSOR_STATUS_INTERNAL_ERROR;
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorContractionGetWorkspace(const hiptensorHandle_t*                handle,
                                                   const hiptensorContractionDescriptor_t* desc,
                                                   const hiptensorContractionFind_t*       find,
                                                   const hiptensorWorksizePreference_t     pref,
                                                   uint64_t* workspaceSize)
{
    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorInitContractionPlan(const hiptensorHandle_t*                handle,
                                               hiptensorContractionPlan_t*             plan,
                                               const hiptensorContractionDescriptor_t* desc,
                                               const hiptensorContractionFind_t*       find,
                                               const uint64_t workspaceSize)
{
    if(!handle || !plan || !desc)
        return HIPTENSOR_STATUS_NOT_INITIALIZED;

    plan->ht_plan_desc = *desc;
    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorContraction(const hiptensorHandle_t *handle,
                                       const hiptensorContractionPlan_t *plan,
                                       const void *alpha, const void *A,
                                       const void *B, const void *beta,
                                       const void *C, void *D, void *workspace,
                                       uint64_t workspaceSize,
                                       hipStream_t stream) {
  if (!handle || !A || !B || !D) {
    return HIPTENSOR_STATUS_NOT_INITIALIZED;
  }

  hiptensorCKContraction(handle, plan, &ht_contract_metrics, alpha, A, B, beta,
                         C, D, workspace, workspaceSize, stream);

  //   if (plan->ht_plan_desc.ht_contract_op == hiptensor_CONTRACTION_SCALE) {
  //     hiptensorCKScaleContraction(handle, plan, &ht_contract_metrics, alpha,
  //     A, B,
  //                                 NULL, NULL, D, workspace, workspaceSize,
  //                                 stream);
  //   } else if (plan->ht_plan_desc.ht_contract_op ==
  //              hiptensor_CONTRACTION_BILINEAR) {
  //     hiptensorCKBilinearContraction(handle, plan, &ht_contract_metrics,
  //     alpha, A,
  //                                    B, beta, C, D, workspace, workspaceSize,
  //                                    stream);
  //   } else {
  //     std::cout << "Contraction operation not permitted" << std::endl;
  //     return hiptensor_STATUS_CK_ERROR;
  //   }
  return HIPTENSOR_STATUS_SUCCESS;
}

void hiptensorContractionPlan_t::hiptensorPrintContractionMetrics()
{
    std::cout << "Perf: " << ht_contract_metrics.avg_time << " ms, " << ht_contract_metrics.tflops
              << " TFlops, " << ht_contract_metrics.transfer_speed << " GB/s, "
              << ht_contract_metrics.ht_instance << std::endl;
}
