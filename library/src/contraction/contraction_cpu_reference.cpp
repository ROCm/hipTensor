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

#include "contraction_cpu_reference.hpp"
#include "contraction_cpu_reference_impl.hpp"
#include "contraction_cpu_reference_instances.hpp"

hiptensorStatus_t hiptensorContractionReference(void const*                alpha,
                                                void const*                A,
                                                void const*                B,
                                                void const*                beta,
                                                void const*                C,
                                                void*                      D,
                                                std::vector<size_t> const& a_ms_ks_lengths,
                                                std::vector<size_t> const& a_ms_ks_strides,
                                                std::vector<size_t> const& b_ns_ks_lengths,
                                                std::vector<size_t> const& b_ns_ks_strides,
                                                std::vector<size_t> const& c_ms_ns_lengths,
                                                std::vector<size_t> const& c_ms_ns_strides,
                                                std::vector<size_t> const& d_ms_ns_lengths,
                                                std::vector<size_t> const& d_ms_ns_strides,
                                                hipDataType                typeA,
                                                hipDataType                typeB,
                                                hipDataType                typeC,
                                                hipDataType                typeD,
                                                void*                      workspace)
{
    auto& instances  = hiptensor::ContractionCpuReferenceInstances::instance();
    auto  candidates = instances->allSolutions().query(typeA, typeB, typeC, typeD);

    auto toCKVec
        = [](auto& inputVec) { return std::vector<ck::index_t>(inputVec.begin(), inputVec.end()); };

    if(candidates.solutionCount() != 1)
    {
        return HIPTENSOR_STATUS_INTERNAL_ERROR;
    }
    else
    {
        auto refCandidate = candidates.solutions().begin()->second;
        if(refCandidate->initArgs(alpha,
                                  A,
                                  B,
                                  beta,
                                  C,
                                  D,
                                  toCKVec(a_ms_ks_lengths),
                                  toCKVec(a_ms_ks_strides),
                                  toCKVec(b_ns_ks_lengths),
                                  toCKVec(b_ns_ks_strides),
                                  std::vector<std::vector<ck::index_t>>{toCKVec(c_ms_ns_lengths)},
                                  std::vector<std::vector<ck::index_t>>{toCKVec(c_ms_ns_strides)},
                                  toCKVec(d_ms_ns_lengths),
                                  toCKVec(d_ms_ns_strides),
                                  workspace))
        {
            (*refCandidate)();
        }
        return HIPTENSOR_STATUS_SUCCESS;
    }
}
