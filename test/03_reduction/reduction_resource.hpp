/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_REDUCTION_RESOURCE_HPP
#define HIPTENSOR_REDUCTION_RESOURCE_HPP

#include <memory>
#include <tuple>

#include "hip_resource.hpp"
#include "singleton.hpp"

// ReductionResource class is intended to manage a shared pool of resources for
// testing hiptensor reduction kernels on the GPU.
//
// It minimizes the memory handling overhead for launching thousands of GPU
// kernels by allowing re-use of existing memory allocations. Memory is only
// re-allocated as necessary to satisfy minimum size requirements.
//
// The interface indicates memory ownership by this class and shall only be
// used to access for read/write purposes.
//
// Currently uses HIP as the backend for device allocation.

namespace hiptensor
{

    struct ReductionResource : public HipResource, public LazySingleton<ReductionResource>
    {
        // For static initialization
        friend std::unique_ptr<ReductionResource> std::make_unique<ReductionResource>();

        using Base = HipResource;

    public:
        using DevicePtrT = Base::DevicePtrT;
        using HostPtrT   = Base::HostPtrT;

        // N, C, W, H
        using ProblemDims = std::vector<std::size_t>;

    private: // No public instantiation except make_unique.
             // No copy
        ReductionResource();
        ReductionResource(const ReductionResource&)            = delete;
        ReductionResource& operator=(const ReductionResource&) = delete;

    public:
        ReductionResource(ReductionResource&&);
        virtual ~ReductionResource() = default;

        void setupStorage(ProblemDims const& dimSizes,
                          ProblemDims const& outputSizes,
                          hipDataType        dataType);
        void fillRand(HostPtrT& hostBuf, DevicePtrT& deviceBuf, size_t elementCount);
        void copyCToHost();
        void copyReferenceToDevice();

        HostPtrT& hostA();
        HostPtrT& hostC();
        HostPtrT& hostReference();

        DevicePtrT& deviceA();
        DevicePtrT& deviceC();
        DevicePtrT& deviceReference();

        size_t getCurrentMatrixAElement() const;
        size_t getCurrentMatrixAMemorySize() const;
        size_t getCurrentMatrixCElement() const;
        size_t getCurrentMatrixCMemorySize() const;
        void   reset() final;

    protected:
        DevicePtrT mDeviceA, mDeviceC, mDeviceReference;
        HostPtrT   mHostA, mHostC, mHostReference;

        hipDataType mCurrentDataType; /**< Type size of element of A/C */

        size_t mCurrentMatrixAElement; /**< Element count of A */
        size_t mCurrentAllocByteA; /**< Allocated size of memory */

        size_t mCurrentMatrixCElement; /**< Element count of C */
        size_t mCurrentAllocByteC; /**< Allocated size of memory */
    };

} // namespace hiptensor

#endif // HIPTENSOR_REDUCTION_RESOURCE_HPP
