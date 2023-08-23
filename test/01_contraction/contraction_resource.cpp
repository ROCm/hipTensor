/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2023 Advanced Micro Devices, Inc.
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

#ifndef HIPTENSOR_CONTRACTION_RESOURCE_IMPL_HPP
#define HIPTENSOR_CONTRACTION_RESOURCE_IMPL_HPP

#include "contraction_resource.hpp"

namespace hiptensor
{

    ContractionResource::ContractionResource()
        : HipResource()
        , mDeviceA(Base::allocDevice(0))
        , mDeviceB(Base::allocDevice(0))
        , mDeviceC(Base::allocDevice(0))
        , mDeviceD(Base::allocDevice(0))
        , mHostA(Base::allocHost(0))
        , mHostB(Base::allocHost(0))
        , mHostC(Base::allocHost(0))
        , mHostD(Base::allocHost(0))
        , mCurrentMatrixElements({0, 0, 0, 0})
        , mCurrentAllocElements({0, 0, 0, 0})
    {
    }

    ContractionResource::ContractionResource(ContractionResource&& rhs)
        : HipResource()
        , mDeviceA(std::move(rhs.mDeviceA))
        , mDeviceB(std::move(rhs.mDeviceB))
        , mDeviceC(std::move(rhs.mDeviceC))
        , mDeviceD(std::move(rhs.mDeviceD))
        , mHostA(std::move(rhs.mHostA))
        , mHostB(std::move(rhs.mHostB))
        , mHostC(std::move(rhs.mHostC))
        , mHostD(std::move(rhs.mHostD))
        , mCurrentMatrixElements(rhs.mCurrentMatrixElements)
        , mCurrentAllocElements(rhs.mCurrentAllocElements)
    {
    }

    void ContractionResource::copyHostToDeviceAll(ElementBytes const& bytesPerElement)
    {
        Base::copyData(mDeviceA, mHostA, std::get<MatrixA>(mCurrentMatrixElements) * std::get<MatrixA>(bytesPerElement));
        Base::copyData(mDeviceB, mHostB, std::get<MatrixB>(mCurrentMatrixElements) * std::get<MatrixB>(bytesPerElement));
        Base::copyData(mDeviceC, mHostC, std::get<MatrixC>(mCurrentMatrixElements) * std::get<MatrixC>(bytesPerElement));
        Base::copyData(mDeviceD, mHostD, std::get<MatrixD>(mCurrentMatrixElements) * std::get<MatrixD>(bytesPerElement));
    }

    void ContractionResource::copyDeviceToHostAll(ElementBytes const& bytesPerElement)
    {
        Base::copyData(mHostA, mDeviceA, std::get<MatrixA>(mCurrentMatrixElements) * std::get<MatrixA>(bytesPerElement));
        Base::copyData(mHostB, mDeviceB, std::get<MatrixB>(mCurrentMatrixElements) * std::get<MatrixB>(bytesPerElement));
        Base::copyData(mHostC, mDeviceC, std::get<MatrixC>(mCurrentMatrixElements) * std::get<MatrixC>(bytesPerElement));
        Base::copyData(mHostD, mDeviceD, std::get<MatrixD>(mCurrentMatrixElements) * std::get<MatrixD>(bytesPerElement));
    }

    void ContractionResource::resizeStorage(ProblemDims const& size, ElementBytes bytesPerElement)
    {
        // elements MatrixA = M * N * H * K
        // elements MatrixB = U * V * H * K
        // elements MatrixC = M * N * U * V
        // elements MatrixD = M * N * U * V
        resizeStorage(
            std::make_tuple(size[M] * size[N] * size[H] * size[K],
                            size[U] * size[V] * size[H] * size[K],
                            size[M] * size[N] * size[U] * size[V],
                            size[M] * size[N] * size[U] * size[V]),
            bytesPerElement); 
    }

    void ContractionResource::resizeStorage(MatrixElements const& newMatrixElements, ElementBytes bytesPerElement)
    {
        auto conditionalReallocDeviceHostPair = [](auto&    devicePtr,
                                                   auto&    hostPtr,
                                                   int64_t& currentAllocElements,
                                                   int64_t  newAllocElements,
                                                   int32_t  elementBytes) {
            // Only realloc if required (e.g. current allocation won't fit new sizes)
            if(currentAllocElements < newAllocElements)
            {
                Base::reallocDeviceHostPair(devicePtr, hostPtr, newAllocElements * elementBytes);
                currentAllocElements = newAllocElements;
            }
        };

        conditionalReallocDeviceHostPair(mDeviceA,
                                         mHostA,
                                         std::get<MatrixA>(mCurrentAllocElements),
                                         std::get<MatrixA>(newMatrixElements),
                                         std::get<MatrixA>(bytesPerElement));
        conditionalReallocDeviceHostPair(mDeviceB,
                                         mHostB,
                                         std::get<MatrixB>(mCurrentAllocElements),
                                         std::get<MatrixB>(newMatrixElements),
                                         std::get<MatrixB>(bytesPerElement));
        conditionalReallocDeviceHostPair(mDeviceC,
                                         mHostC,
                                         std::get<MatrixC>(mCurrentAllocElements),
                                         std::get<MatrixC>(newMatrixElements),
                                         std::get<MatrixC>(bytesPerElement));
        conditionalReallocDeviceHostPair(mDeviceD,
                                         mHostD,
                                         std::get<MatrixD>(mCurrentAllocElements),
                                         std::get<MatrixD>(newMatrixElements),
                                         std::get<MatrixD>(bytesPerElement));

        // Always update the current matrix element count
        mCurrentMatrixElements = newMatrixElements;
        // std::cout << std::get<MatrixA>(mCurrentAllocElements) << ", " 
        //           << std::get<MatrixB>(mCurrentAllocElements) << ", " 
        //           << std::get<MatrixC>(mCurrentAllocElements) << ", " 
        //           << std::get<MatrixD>(mCurrentAllocElements) << std::endl;
    }

    void ContractionResource::reset()
    {
        Base::reallocDeviceHostPair(mDeviceA, mHostA, 0);
        Base::reallocDeviceHostPair(mDeviceB, mHostB, 0);
        Base::reallocDeviceHostPair(mDeviceC, mHostC, 0);
        Base::reallocDeviceHostPair(mDeviceD, mHostD, 0);
        mCurrentAllocElements  = {0, 0, 0, 0};
        mCurrentMatrixElements = {0, 0, 0, 0};
    }

    auto ContractionResource::hostA() -> HostPtrT&
    {
        return mHostA;
    }

    auto ContractionResource::hostB() -> HostPtrT&
    {
        return mHostB;
    }

    auto ContractionResource::hostC() -> HostPtrT&
    {
        return mHostC;
    }

    auto ContractionResource::hostD() -> HostPtrT&
    {
        return mHostD;
    }

    auto ContractionResource::deviceA() -> DevicePtrT&
    {
        return mDeviceA;
    }

    auto ContractionResource::deviceB() -> DevicePtrT&
    {
        return mDeviceB;
    }

    auto ContractionResource::deviceC() -> DevicePtrT&
    {
        return mDeviceC;
    }

    auto ContractionResource::deviceD() -> DevicePtrT&
    {
        return mDeviceD;
    }

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_RESOURCE_IMPL_HPP
