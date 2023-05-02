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

#include "contraction_solution.hpp"

namespace hiptensor
{

    ContractionSolution::ContractionSolution(ContractionSolution&& other)
        : mM(other.mM)
        , mN(other.mN)
        , mK(other.mK)
        , mBytes(other.mBytes)
        , mValid(other.mValid)
        , mKernelName(other.mKernelName)
        , mOpId(other.mOpId)
        , mInitArgs(other.mInitArgs)
        , mDeviceOp(std::move(other.mDeviceOp))
        , mArgPtr(std::move(other.mArgPtr))
        , mInvokerPtr(std::move(other.mInvokerPtr))
    {
    }

    ContractionSolution& ContractionSolution::operator=(ContractionSolution&& other)
    {
        if(this != &other)
        {
            mM = other.mM;
            mN = other.mN;
            mK = other.mK;

            mBytes      = other.mBytes;
            mValid      = other.mValid;
            mKernelName = other.mKernelName;
            mOpId       = other.mOpId;

            mInitArgs   = other.mInitArgs;
            mDeviceOp   = std::move(other.mDeviceOp);
            mArgPtr     = std::move(other.mArgPtr);
            mInvokerPtr = std::move(other.mInvokerPtr);
        }
        return *this;
    }

    bool
        ContractionSolution::initArgs(void const*                                  alpha,
                                      void const*                                  A,
                                      void const*                                  B,
                                      void const*                                  beta,
                                      void const*                                  D,
                                      void*                                        E,
                                      std::vector<ck::index_t> const&              a_ms_ns_lengths,
                                      std::vector<ck::index_t> const&              a_ms_ks_strides,
                                      std::vector<ck::index_t> const&              b_ns_ks_lengths,
                                      std::vector<ck::index_t> const&              b_ns_ks_strides,
                                      std::vector<std::vector<ck::index_t>> const& ds_ms_ns_lengths,
                                      std::vector<std::vector<ck::index_t>> const& ds_ms_ns_strides,
                                      std::vector<ck::index_t> const&              e_ms_ns_lengths,
                                      std::vector<ck::index_t> const&              e_ms_ns_strides)
    {
        if(mDeviceOp)
        {
            mInitArgs(*this,
                      alpha,
                      A,
                      B,
                      beta,
                      D,
                      E,
                      a_ms_ns_lengths,
                      a_ms_ks_strides,
                      b_ns_ks_lengths,
                      b_ns_ks_strides,
                      ds_ms_ns_lengths,
                      ds_ms_ns_strides,
                      e_ms_ns_lengths,
                      e_ms_ns_strides);

            return mValid;
        }
        return false;
    }

    float ContractionSolution::operator()(StreamConfig const& streamConfig /*= StreamConfig{}*/)
    {
        if(!mArgPtr || !mInvokerPtr || mOpId == ContractionOpId_t::UNKNOWN)
        {
#if !NDEBUG
            std::cout << deviceOp->GetTypeString() << " is not initialized" << std::endl;
#endif // !NDEBUG
            return -1.0f;
        }

        if(!mValid)
        {
#if !NDEBUG
            std::cout << mKernelName << " does not support this problem" << std::endl;
#endif // !NDEBUG
            return -1.0f;
        }

        return mInvokerPtr->Run(mArgPtr.get(), streamConfig);
    }

    float ContractionSolution::operator()(
        void const*                                  alpha,
        void const*                                  A,
        void const*                                  B,
        void const*                                  beta,
        void const*                                  D,
        void*                                        E,
        std::vector<ck::index_t> const&              a_ms_ns_lengths,
        std::vector<ck::index_t> const&              a_ms_ks_strides,
        std::vector<ck::index_t> const&              b_ns_ks_lengths,
        std::vector<ck::index_t> const&              b_ns_ks_strides,
        std::vector<std::vector<ck::index_t>> const& ds_ms_ns_lengths,
        std::vector<std::vector<ck::index_t>> const& ds_ms_ns_strides,
        std::vector<ck::index_t> const&              e_ms_ns_lengths,
        std::vector<ck::index_t> const&              e_ms_ns_strides,
        StreamConfig const&                          streamConfig /*= StreamConfig{}*/)
    {
        if(!initArgs(alpha,
                     A,
                     B,
                     beta,
                     D,
                     E,
                     a_ms_ns_lengths,
                     a_ms_ks_strides,
                     b_ns_ks_lengths,
                     b_ns_ks_strides,
                     ds_ms_ns_lengths,
                     ds_ms_ns_strides,
                     e_ms_ns_lengths,
                     e_ms_ns_strides))
        {
#if !NDEBUG
            std::cout << mKernelName << " does not support this problem" << std::endl;
#endif // !NDEBUG
            return -1.0f;
        }

        return mInvokerPtr->Run(mArgPtr.get(), streamConfig);
    }

    bool ContractionSolution::isValid() const
    {
        return mValid;
    }

} // namespace hiptensor
