/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include "contraction_solution.hpp"

namespace hiptensor
{

    ContractionSolution::ContractionSolution(
        std::unique_ptr<ck::tensor_operation::device::BaseOperator>&& deviceOp,
        std::unique_ptr<ContractionSolutionParams>&&                  params)
        : mM(0)
        , mN(0)
        , mK(0)
        , mBytes(0)
        , mValid(false)
        , mDeviceOp(std::move(deviceOp))
        , mParams(std::move(params))
    {
    }

    ContractionSolution::ContractionSolution(ContractionSolution&& other)
        : mM(other.mM)
        , mN(other.mN)
        , mK(other.mK)
        , mBytes(other.mBytes)
        , mValid(other.mValid)
        , mDeviceOp(std::move(other.mDeviceOp))
        , mParams(std::move(other.mParams))
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

            mBytes = other.mBytes;
            mValid = other.mValid;

            mParams     = std::move(other.mParams);
            mDeviceOp   = std::move(other.mDeviceOp);
            mArgPtr     = std::move(other.mArgPtr);
            mInvokerPtr = std::move(other.mInvokerPtr);
        }
        return *this;
    }

    float ContractionSolution::operator()(StreamConfig const& streamConfig /*= StreamConfig{}*/)
    {
        if(!mArgPtr || !mInvokerPtr || !mParams || mParams->opCDE() == ContractionOpId_t::UNKNOWN)
        {
#if !NDEBUG
            std::cout << mDeviceOp->GetTypeString() << " is not initialized" << std::endl;
#endif // !NDEBUG
            return -1.0f;
        }

        if(!mValid)
        {
#if !NDEBUG
            std::cout << kernelName() << " does not support this problem" << std::endl;
#endif // !NDEBUG
            return -1.0f;
        }

        return mInvokerPtr->Run(mArgPtr.get(), streamConfig);
    }

    float ContractionSolution::operator()(void const*                     alpha,
                                          void const*                     A,
                                          void const*                     B,
                                          void const*                     beta,
                                          void const*                     D,
                                          void*                           E,
                                          std::vector<std::size_t> const& a_ms_ns_lengths,
                                          std::vector<std::size_t> const& a_ms_ks_strides,
                                          std::vector<std::size_t> const& b_ns_ks_lengths,
                                          std::vector<std::size_t> const& b_ns_ks_strides,
                                          std::vector<std::size_t> const& ds_ms_ns_lengths,
                                          std::vector<std::size_t> const& ds_ms_ns_strides,
                                          std::vector<std::size_t> const& e_ms_ns_lengths,
                                          std::vector<std::size_t> const& e_ms_ns_strides,
                                          void*                           workspacePtr,
                                          StreamConfig const& streamConfig /*= StreamConfig{}*/)
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
                     e_ms_ns_strides,
                     workspacePtr))
        {
#if !NDEBUG
            std::cout << kernelName() << " does not support this problem" << std::endl;
#endif // !NDEBUG
            return -1.0f;
        }

        return mInvokerPtr->Run(mArgPtr.get(), streamConfig);
    }

    bool ContractionSolution::isValid() const
    {
        return mValid;
    }

    std::unique_ptr<ContractionSolutionParams> const& ContractionSolution::params() const
    {
        return mParams;
    }

    size_t ContractionSolution::uid() const
    {
        // Convert CK uid string into binary.
        std::istringstream converter(mDeviceOp->GetTypeIdHashCode());
        size_t             value;
        converter >> std::hex >> value;
        return value;
    }

    std::tuple<ck::index_t, ck::index_t, ck::index_t> ContractionSolution::problemDims() const
    {
        return std::make_tuple(mM, mN, mK);
    }

    ck::index_t ContractionSolution::problemBytes() const
    {
        return mBytes;
    }

    std::string ContractionSolution::kernelName() const
    {
        return mDeviceOp->GetTypeString();
    }

    size_t ContractionSolution::workspaceSize() const
    {
        if(mValid)
        {
            return mDeviceOp->GetWorkSpaceSize(mArgPtr.get());
        }
        else
        {
            return 0;
        }
    }

    void ContractionSolution::resetArgs()
    {
        mM     = 0;
        mN     = 0;
        mK     = 0;
        mBytes = 0;

        mArgPtr.reset(nullptr);
        mInvokerPtr.reset(nullptr);

        mValid = false;
    }

} // namespace hiptensor
