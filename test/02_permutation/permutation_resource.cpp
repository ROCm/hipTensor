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

#ifndef HIPTENSOR_PERMUTATION_RESOURCE_IMPL_HPP
#define HIPTENSOR_PERMUTATION_RESOURCE_IMPL_HPP

#include "permutation_resource.hpp"
#include "data_types.hpp"
#include "utils.hpp"

namespace hiptensor
{

    PermutationResource::PermutationResource()
        : HipResource()
        , mDeviceA(Base::allocDevice(0))
        , mDeviceB(Base::allocDevice(0))
        , mHostA(Base::allocHost(0))
        , mHostB(Base::allocHost(0))
        , mCurrentMatrixElement(0)
        , mCurrentDataType(HIP_R_32F)
        , mCurrentAllocByte(0)
    {
    }

    PermutationResource::PermutationResource(PermutationResource&& rhs)
        : HipResource()
        , mDeviceA(std::move(rhs.mDeviceA))
        , mDeviceB(std::move(rhs.mDeviceB))
        , mHostA(std::move(rhs.mHostA))
        , mHostB(std::move(rhs.mHostB))
        , mCurrentMatrixElement(rhs.mCurrentMatrixElement)
        , mCurrentDataType(rhs.mCurrentDataType)
        , mCurrentAllocByte(rhs.mCurrentAllocByte)
    {
    }

    void PermutationResource::setupStorage(ProblemDims const& dimSizes, hipDataType dataType)
    {
        auto requiredElementCount = getProduct(dimSizes);
        auto requiredMemorySize   = requiredElementCount * hipDataTypeSize(dataType);

        bool needFillData = false;
        if(requiredMemorySize > mCurrentAllocByte)
        {
            Base::reallocDeviceHostPair(mDeviceA, mHostA, requiredMemorySize);
            Base::reallocDeviceHostPair(mDeviceB, mHostB, requiredMemorySize);
            Base::reallocDeviceHostPair(mDeviceReference, mHostReference, requiredMemorySize);
            mCurrentAllocByte = requiredMemorySize;
            needFillData      = true;
        }
        if(mCurrentDataType != dataType || mCurrentMatrixElement < requiredElementCount)
        {
            needFillData = true;
        }
        mCurrentMatrixElement = requiredElementCount;
        mCurrentDataType      = dataType;
        if(needFillData)
        {
            fillRandToA();
        }
    }

    void PermutationResource::reset()
    {
        Base::reallocDeviceHostPair(mDeviceA, mHostA, 0);
        Base::reallocDeviceHostPair(mDeviceB, mHostB, 0);
        Base::reallocDeviceHostPair(mDeviceReference, mHostReference, 0);
        mCurrentMatrixElement = 0;
        mCurrentDataType      = HIP_R_32F;
        mCurrentAllocByte     = 0;
    }

    void PermutationResource::fillRandToA()
    {
        uint32_t seed = static_cast<uint32_t>(std::time(nullptr));
        
        if(mCurrentDataType == HIP_R_32F)
        {
            fillLaunchKernel<float>((float*)deviceA().get(), mCurrentMatrixElement, seed);
        }
        else
        {
            fillLaunchKernel<_Float16>((_Float16*)deviceA().get(), mCurrentMatrixElement, seed);
        }
        Base::copyData(hostA(), deviceA(), getCurrentMatrixMemorySize());
    }

    void PermutationResource::copyBToHost()
    {
        Base::copyData(hostB(), deviceB(), getCurrentMatrixMemorySize());
    }

    void PermutationResource::copyReferenceToDevice()
    {
        Base::copyData(deviceReference(), hostReference(), getCurrentMatrixMemorySize());
    }

    size_t PermutationResource::getCurrentMatrixElement() const
    {
        return mCurrentMatrixElement;
    }

    size_t PermutationResource::getCurrentMatrixMemorySize() const
    {
        return mCurrentMatrixElement * hipDataTypeSize(mCurrentDataType);
    }

    auto PermutationResource::hostA() -> HostPtrT&
    {
        return mHostA;
    }

    auto PermutationResource::hostB() -> HostPtrT&
    {
        return mHostB;
    }

    auto PermutationResource::hostReference() -> HostPtrT&
    {
        return mHostReference;
    }

    auto PermutationResource::deviceA() -> DevicePtrT&
    {
        return mDeviceA;
    }

    auto PermutationResource::deviceB() -> DevicePtrT&
    {
        return mDeviceB;
    }

    auto PermutationResource::deviceReference() -> DevicePtrT&
    {
        return mDeviceReference;
    }
} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_RESOURCE_IMPL_HPP
