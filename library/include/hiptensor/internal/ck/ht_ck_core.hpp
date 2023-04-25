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
#ifndef HT_CK_CORE_HPP_
#define HT_CK_CORE_HPP_

#include "../../ht_types.hpp"
#include "../ht_utility.hpp"

hiptensorStatus_t hiptensorCKContraction(const hiptensorHandle_t*          handle,
                                         const hiptensorContractionPlan_t* plan,
                                         hiptensorContractionMetrics_t*    ht_contract_metrics,
                                         const void*                       alpha,
                                         const void*                       A,
                                         const void*                       B,
                                         const void*                       beta,
                                         const void*                       C,
                                         void*                             D,
                                         void*                             workspace,
                                         uint64_t                          workspaceSize,
                                         hipStream_t                       stream);

// hiptensorStatus_t hiptensorCKBilinearContraction(
//     const hiptensorHandle_t *handle, const hiptensorContractionPlan_t *plan,
//     hiptensorContractionMetrics_t *ht_contract_metrics, const void *alpha,
//     const void *A, const void *B, const void *beta, const void *C, void *D,
//     void *workspace, uint64_t workspaceSize, hipStream_t stream);
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
