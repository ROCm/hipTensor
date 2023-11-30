/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_CONTRACTION_SOLUTION_IMPL_HPP
#define HIPTENSOR_CONTRACTION_SOLUTION_IMPL_HPP

#include <numeric>

#include "contraction_solution.hpp"
#include "hash.hpp"

namespace std
{
    template <>
    struct std::hash<hiptensor::ContractionSolution>
    {
        std::size_t operator()(hiptensor::ContractionSolution const& s) const noexcept
        {
            return std::hash<hiptensor::ContractionSolutionParams>{}(*s.params());
        }
    };
}

namespace hiptensor
{
    template <typename DeviceOp, typename DataType, typename Enabler = void>
    class ContractionSolutionImpl;

    template <typename DeviceOp, typename DataType>
    class ContractionSolutionImpl<
        DeviceOp,
        DataType,
        std::enable_if_t<!std::is_same<DataType, hipFloatComplex>{} &&
                          std::is_same_v<typename MetaTraits<DeviceOp>::CDEOp,
                                        ck::tensor_operation::element_wise::Bilinear>>>
        : public ContractionSolution
    {
    protected :
        // Kernel Params
        std::unique_ptr<ContractionSolutionParams>                  mParams;
        std::unique_ptr<ck::tensor_operation::device::BaseArgument> mArgPtr;
        std::unique_ptr<ck::tensor_operation::device::BaseInvoker>  mInvokerPtr;

    public:
        using Base   = ContractionSolution;

        ContractionSolutionImpl(ContractionSolutionImpl&& other)
        : mParams(std::move(other.mParams))
        , mArgPtr(std::move(other.mArgPtr))
        , mInvokerPtr(std::move(other.mInvokerPtr))
        {
        }

        ContractionSolutionImpl& operator=(ContractionSolutionImpl&& other)
        {
            if(this != &other)
            {
                mParams     = std::move(other.mParams);
                mArgPtr     = std::move(other.mArgPtr);
                mInvokerPtr = std::move(other.mInvokerPtr);
            }
            return *this;
        }

        std::unique_ptr<ContractionSolutionParams> const& params() const override
        {
            return mParams;
        }

        size_t workspaceSize() const override
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

        void resetArgs()
        {
            mArgPtr.reset(nullptr);
            mInvokerPtr.reset(nullptr);

            // Clear out the previous arguments
            Base::resetArgs();

        }

        ContractionSolutionImpl(std::unique_ptr<DeviceOp>&& deviceOp)
            : ContractionSolution(std::move(deviceOp),
                                 std::make_unique<ContractionSolutionParamsImpl<DeviceOp>>())
        {
        }

        bool initArgs(void const*                     alpha,
                      void const*                     A,
                      void const*                     B,
                      void const*                     beta,
                      void const*                     D,
                      void*                           E,
                      std::vector<std::size_t> const& a_ms_ks_lengths,
                      std::vector<std::size_t> const& a_ms_ks_strides,
                      std::vector<std::size_t> const& b_ns_ks_lengths,
                      std::vector<std::size_t> const& b_ns_ks_strides,
                      std::vector<std::size_t> const& ds_ms_ns_lengths,
                      std::vector<std::size_t> const& ds_ms_ns_strides,
                      std::vector<std::size_t> const& e_ms_ns_lengths,
                      std::vector<std::size_t> const& e_ms_ns_strides,
                      void*                           workspacePtr) override
        {
            using Traits = MetaTraits<DeviceOp>;

            // Clear previous data
            Base::resetArgs();

            // Promote to derived class for necessary functions such as
            // MakeArgumentPointer and MakeInvokerPointer.
            auto* deviceOp = dynamic_cast<DeviceOp*>(mDeviceOp.get());

            // Note: CK ALWAYS uses float for alpha / beta in contraction multipleD
            auto alphaF = 0.0f;
            auto betaF  = 0.0f;

            if(alpha != nullptr)
            {
                alphaF = hiptensor::readVal<float>(alpha, convertToComputeType(HipDataType_v<typename Traits::EDataT>));
            }
            if(beta != nullptr)
            {
                betaF = hiptensor::readVal<float>(beta, convertToComputeType(HipDataType_v<typename Traits::EDataT>));
            }

            // CK has its own format for indices...
            auto toCKVec = [](std::vector<std::size_t> const& v) {
                return std::vector<ck::index_t>(v.begin(), v.end());
            };

            // Initialize the argument pointer
            mArgPtr = std::move(deviceOp->MakeArgumentPointer(
                A,
                B,
                std::array<const void*, 1>{D},
                E,
                toCKVec(a_ms_ks_lengths),
                toCKVec(a_ms_ks_strides),
                toCKVec(b_ns_ks_lengths),
                toCKVec(b_ns_ks_strides),
                std::array<std::vector<ck::index_t>, 1>{toCKVec(ds_ms_ns_lengths)},
                std::array<std::vector<ck::index_t>, 1>{toCKVec(ds_ms_ns_strides)},
                toCKVec(e_ms_ns_lengths),
                toCKVec(e_ms_ns_strides),
                typename Traits::AOp{},
                typename Traits::BOp{},
                typename Traits::CDEOp{alphaF, betaF}));

            // Attach the workspace pointer
            deviceOp->SetWorkSpacePointer(mArgPtr.get(), workspacePtr);

            // Initialize the invoker
            mInvokerPtr = std::move(deviceOp->MakeInvokerPointer());

            // Fill problem metrics
            Base::mM = std::accumulate(e_ms_ns_lengths.begin(),
                                       e_ms_ns_lengths.begin() + Traits::DimsM,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            Base::mN = std::accumulate(e_ms_ns_lengths.begin() + Traits::DimsM,
                                       e_ms_ns_lengths.begin() + Traits::DimsM + Traits::DimsN,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            Base::mK = std::accumulate(a_ms_ks_lengths.begin() + Traits::DimsM,
                                       a_ms_ks_lengths.begin() + Traits::DimsM + Traits::DimsK,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            // Byte count
            Base::mBytes = sizeof(typename Traits::ADataT) * Base::mM * Base::mK
                           + sizeof(typename Traits::BDataT) * Base::mK * Base::mN
                           + sizeof(typename Traits::DDataT) * Base::mM * Base::mN
                           + sizeof(typename Traits::EDataT) * Base::mM * Base::mN;

            // Arg test
            Base::mValid = deviceOp->IsSupportedArgument(mArgPtr.get());

            return Base::mValid;
        }

        float operator()(StreamConfig const& streamConfig /*= StreamConfig{}*/) override
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

        float operator()(void const*                     alpha,
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
                                            StreamConfig const& streamConfig /*= StreamConfig{}*/) override
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

    };

    template <typename DeviceOp, typename DataType>
    class ContractionSolutionImpl<
        DeviceOp,
        DataType,
        std::enable_if_t<!std::is_same<DataType, hipFloatComplex>{} &&
                          std::is_same_v<typename MetaTraits<DeviceOp>::CDEOp,
                                         ck::tensor_operation::element_wise::Scale>>>
        : public ContractionSolution
    {
    protected :
        // Kernel Params
        std::unique_ptr<ContractionSolutionParams>                  mParams;
        std::unique_ptr<ck::tensor_operation::device::BaseArgument> mArgPtr;
        std::unique_ptr<ck::tensor_operation::device::BaseInvoker>  mInvokerPtr;

    public:
        using Base   = ContractionSolution;

        ContractionSolutionImpl(ContractionSolutionImpl&& other)
        : mParams(std::move(other.mParams))
        , mArgPtr(std::move(other.mArgPtr))
        , mInvokerPtr(std::move(other.mInvokerPtr))
        {

        }

        ContractionSolutionImpl& operator=(ContractionSolutionImpl&& other)
        {
            if(this != &other)
            {
                mArgPtr     = std::move(other.mArgPtr);
                mInvokerPtr = std::move(other.mInvokerPtr);
            }
            return *this;
        }

        std::unique_ptr<ContractionSolutionParams> const& params() const override
        {
            return mParams;
        }

        size_t workspaceSize() const override
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

        void resetArgs()
        {
            mArgPtr.reset(nullptr);
            mInvokerPtr.reset(nullptr);

            // Clear out the previous arguments
            Base::resetArgs();
        }

        ContractionSolutionImpl(std::unique_ptr<DeviceOp>&& deviceOp)
            : ContractionSolution(std::move(deviceOp),
                                  std::make_unique<ContractionSolutionParamsImpl<DeviceOp>>())
        {
        }

        bool initArgs(void const*                     alpha,
                      void const*                     A,
                      void const*                     B,
                      void const*                     beta,
                      void const*                     D,
                      void*                           E,
                      std::vector<std::size_t> const& a_ms_ks_lengths,
                      std::vector<std::size_t> const& a_ms_ks_strides,
                      std::vector<std::size_t> const& b_ns_ks_lengths,
                      std::vector<std::size_t> const& b_ns_ks_strides,
                      std::vector<std::size_t> const& ds_ms_ns_lengths,
                      std::vector<std::size_t> const& ds_ms_ns_strides,
                      std::vector<std::size_t> const& e_ms_ns_lengths,
                      std::vector<std::size_t> const& e_ms_ns_strides,
                      void*                           workspacePtr) override
        {
            using Traits = MetaTraits<DeviceOp>;

            // Clear previous data
            Base::resetArgs();

            // Promote to derived class for necessary functions such as
            // MakeArgumentPointer and MakeInvokerPointer.
            auto* deviceOp = dynamic_cast<DeviceOp*>(mDeviceOp.get());

            // Note: CK ALWAYS uses float for alpha / beta in contraction multipleD
            auto alphaF = 0.0f;

            if(alpha != nullptr)
            {
                alphaF = hiptensor::readVal<float>(alpha, convertToComputeType(HipDataType_v<typename Traits::EDataT>));
            }

            // CK has its own format for indices...
            auto toCKVec = [](std::vector<std::size_t> const& v) {
                return std::vector<ck::index_t>(v.begin(), v.end());
            };

            // Initialize the argument pointer
            mArgPtr
                = std::move(deviceOp->MakeArgumentPointer(A,
                                                          B,
                                                          std::array<const void*, 0>{},
                                                          E,
                                                          toCKVec(a_ms_ks_lengths),
                                                          toCKVec(a_ms_ks_strides),
                                                          toCKVec(b_ns_ks_lengths),
                                                          toCKVec(b_ns_ks_strides),
                                                          std::array<std::vector<ck::index_t>, 0>{},
                                                          std::array<std::vector<ck::index_t>, 0>{},
                                                          toCKVec(e_ms_ns_lengths),
                                                          toCKVec(e_ms_ns_strides),
                                                          typename Traits::AOp{},
                                                          typename Traits::BOp{},
                                                          typename Traits::CDEOp{alphaF}));

            // Attach the workspace pointer
            deviceOp->SetWorkSpacePointer(mArgPtr.get(), workspacePtr);

            // Initialize the invoker
            mInvokerPtr = std::move(deviceOp->MakeInvokerPointer());

            // Fill problem metrics
            Base::mM = std::accumulate(e_ms_ns_lengths.begin(),
                                       e_ms_ns_lengths.begin() + Traits::DimsM,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            Base::mN = std::accumulate(e_ms_ns_lengths.begin() + Traits::DimsM,
                                       e_ms_ns_lengths.begin() + Traits::DimsM + Traits::DimsN,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            Base::mK = std::accumulate(a_ms_ks_lengths.begin() + Traits::DimsM,
                                       a_ms_ks_lengths.begin() + Traits::DimsM + Traits::DimsK,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            // Byte count
            Base::mBytes = sizeof(typename Traits::ADataT) * Base::mM * Base::mK
                           + sizeof(typename Traits::BDataT) * Base::mK * Base::mN
                           + sizeof(typename Traits::EDataT) * Base::mM * Base::mN;

            // Arg test
            Base::mValid = deviceOp->IsSupportedArgument(mArgPtr.get());

            return Base::mValid;
        }

        float operator()(StreamConfig const& streamConfig /*= StreamConfig{}*/) override
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

        float operator()(void const*                     alpha,
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
                                            StreamConfig const& streamConfig /*= StreamConfig{}*/) override
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

    };

    //-------------------------------------------------------------------------------------------------//
    // hipFloatComplex

    template <typename DeviceOp, typename DataType>
    class ContractionSolutionImpl<
        DeviceOp,
        DataType,
        std::enable_if_t<std::is_same<DataType, hipFloatComplex>{} &&
                         std::is_same_v<typename MetaTraits<DeviceOp>::CDEOp,
                                        ck::tensor_operation::element_wise::Bilinear>>>
        : public ContractionSolution
    {
    protected :
        // Kernel Params
        std::unique_ptr<ContractionSolutionParams>                  mParams;
        std::vector<std::unique_ptr<ck::tensor_operation::device::BaseArgument>> mArgPtr;
        std::vector<std::unique_ptr<ck::tensor_operation::device::BaseInvoker>>  mInvokerPtr;

        void *A_re, *A_img, *B_re, *B_img, *D_re, *D_img, *E_re, *E_img;

    public:
        using Base   = ContractionSolution;

        ContractionSolutionImpl(ContractionSolutionImpl&& other)
        : mParams(std::move(other.mParams))
        , mArgPtr(std::move(other.mArgPtr))
        , mInvokerPtr(std::move(other.mInvokerPtr))
        {

        }

        ContractionSolutionImpl& operator=(ContractionSolutionImpl&& other)
        {
            if(this != &other)
            {
                mArgPtr     = std::move(other.mArgPtr);
                mInvokerPtr = std::move(other.mInvokerPtr);
            }
            return *this;
        }

        std::unique_ptr<ContractionSolutionParams> const& params() const override
        {
            return mParams;
        }

        size_t workspaceSize() const override
        {
            if(mValid)
            {
                return mDeviceOp->GetWorkSpaceSize(mArgPtr[0].get());
            }
            else
            {
                return 0;
            }
        }

        void resetArgs()
        {
            mArgPtr.reserve(4);
            mArgPtr[0].reset(nullptr);
            mArgPtr[1].reset(nullptr);
            mArgPtr[2].reset(nullptr);
            mArgPtr[3].reset(nullptr);
            mInvokerPtr.reserve(4);
            mInvokerPtr[0].reset(nullptr);
            mInvokerPtr[1].reset(nullptr);
            mInvokerPtr[2].reset(nullptr);
            mInvokerPtr[3].reset(nullptr);

            // Clear out the previous arguments
            Base::resetArgs();
        }

        ContractionSolutionImpl(std::unique_ptr<DeviceOp>&& deviceOp)
            : ContractionSolution(std::move(deviceOp),
                                  std::make_unique<ContractionSolutionParamsImpl<DeviceOp>>())
        {
        }

        bool initArgs(void const*                     alpha,
                      void const*                     A,
                      void const*                     B,
                      void const*                     beta,
                      void const*                     D,
                      void*                           E,
                      std::vector<std::size_t> const& a_ms_ks_lengths,
                      std::vector<std::size_t> const& a_ms_ks_strides,
                      std::vector<std::size_t> const& b_ns_ks_lengths,
                      std::vector<std::size_t> const& b_ns_ks_strides,
                      std::vector<std::size_t> const& ds_ms_ns_lengths,
                      std::vector<std::size_t> const& ds_ms_ns_strides,
                      std::vector<std::size_t> const& e_ms_ns_lengths,
                      std::vector<std::size_t> const& e_ms_ns_strides,
                      void*                           workspacePtr) override
        {
            using Traits = MetaTraits<DeviceOp>;

            // Clear out the previous arguments
            Base::resetArgs();

            // Promote to derived class for necessary functions such as
            // MakeArgumentPointer and MakeInvokerPointer.
            auto* deviceOp = dynamic_cast<DeviceOp*>(mDeviceOp.get());

            // Note: CK ALWAYS uses float for alpha / beta in contraction multipleD
            auto alphaF = 0.0f;
            auto betaF  = 0.0f;

            if(alpha != nullptr)
            {
                alphaF = hiptensor::readVal<float>(alpha, convertToComputeType(HipDataType_v<typename Traits::EDataT>));
            }
            if(beta != nullptr)
            {
                betaF = hiptensor::readVal<float>(beta, convertToComputeType(HipDataType_v<typename Traits::EDataT>));
            }

            // CK has its own format for indices...
            auto toCKVec = [](std::vector<std::size_t> const& v) {
                return std::vector<ck::index_t>(v.begin(), v.end());
            };

            //TODO
            // Extract A_re, A_img from A
            // Extract B_re, B_img from B
            // Extract D_re, D_img from D
            // Extract E_re, E_img from E

            // Initialize the argument pointer
            mArgPtr[0] = std::move(deviceOp->MakeArgumentPointer(
                A_re,
                B_re,
                std::array<const void*, 1>{D_re},
                E_re,
                toCKVec(a_ms_ks_lengths),
                toCKVec(a_ms_ks_strides),
                toCKVec(b_ns_ks_lengths),
                toCKVec(b_ns_ks_strides),
                std::array<std::vector<ck::index_t>, 1>{toCKVec(ds_ms_ns_lengths)},
                std::array<std::vector<ck::index_t>, 1>{toCKVec(ds_ms_ns_strides)},
                toCKVec(e_ms_ns_lengths),
                toCKVec(e_ms_ns_strides),
                typename Traits::AOp{},
                typename Traits::BOp{},
                typename Traits::CDEOp{alphaF, betaF}));

            // Attach the workspace pointer
            deviceOp->SetWorkSpacePointer(mArgPtr[0].get(), workspacePtr);

            // Initialize the invoker
            mInvokerPtr[0] = std::move(deviceOp->MakeInvokerPointer());

            mArgPtr[1] = std::move(deviceOp->MakeArgumentPointer(
                A_img,
                B_img,
                std::array<const void*, 1>{E_re}, // Confirm if it can do in place fma, else create intermediate pointer result
                E_re,
                toCKVec(a_ms_ks_lengths),
                toCKVec(a_ms_ks_strides),
                toCKVec(b_ns_ks_lengths),
                toCKVec(b_ns_ks_strides),
                std::array<std::vector<ck::index_t>, 1>{toCKVec(ds_ms_ns_lengths)},
                std::array<std::vector<ck::index_t>, 1>{toCKVec(ds_ms_ns_strides)},
                toCKVec(e_ms_ns_lengths),
                toCKVec(e_ms_ns_strides),
                typename Traits::AOp{},
                typename Traits::BOp{},
                typename Traits::CDEOp{alphaF * -1.0f, betaF}));

            mInvokerPtr[1] = std::move(deviceOp->MakeInvokerPointer());

            mArgPtr[2] = std::move(deviceOp->MakeArgumentPointer(
                A_re,
                B_img,
                std::array<const void*, 1>{D_img},
                E_img,
                toCKVec(a_ms_ks_lengths),
                toCKVec(a_ms_ks_strides),
                toCKVec(b_ns_ks_lengths),
                toCKVec(b_ns_ks_strides),
                std::array<std::vector<ck::index_t>, 1>{toCKVec(ds_ms_ns_lengths)},
                std::array<std::vector<ck::index_t>, 1>{toCKVec(ds_ms_ns_strides)},
                toCKVec(e_ms_ns_lengths),
                toCKVec(e_ms_ns_strides),
                typename Traits::AOp{},
                typename Traits::BOp{},
                typename Traits::CDEOp{alphaF, betaF}));

            mInvokerPtr[2] = std::move(deviceOp->MakeInvokerPointer());

            mArgPtr[3] = std::move(deviceOp->MakeArgumentPointer(
                A_img,
                B_re,
                std::array<const void*, 1>{E_img},
                E_img,
                toCKVec(a_ms_ks_lengths),
                toCKVec(a_ms_ks_strides),
                toCKVec(b_ns_ks_lengths),
                toCKVec(b_ns_ks_strides),
                std::array<std::vector<ck::index_t>, 1>{toCKVec(ds_ms_ns_lengths)},
                std::array<std::vector<ck::index_t>, 1>{toCKVec(ds_ms_ns_strides)},
                toCKVec(e_ms_ns_lengths),
                toCKVec(e_ms_ns_strides),
                typename Traits::AOp{},
                typename Traits::BOp{},
                typename Traits::CDEOp{alphaF, betaF}));

            mInvokerPtr[3] = std::move(deviceOp->MakeInvokerPointer());

            // Fill problem metrics
            Base::mM = std::accumulate(e_ms_ns_lengths.begin(),
                                       e_ms_ns_lengths.begin() + Traits::DimsM,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            Base::mN = std::accumulate(e_ms_ns_lengths.begin() + Traits::DimsM,
                                       e_ms_ns_lengths.begin() + Traits::DimsM + Traits::DimsN,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            Base::mK = std::accumulate(a_ms_ks_lengths.begin() + Traits::DimsM,
                                       a_ms_ks_lengths.begin() + Traits::DimsM + Traits::DimsK,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            // Byte count
            Base::mBytes = sizeof(typename Traits::ADataT) * Base::mM * Base::mK
                           + sizeof(typename Traits::BDataT) * Base::mK * Base::mN
                           + sizeof(typename Traits::DDataT) * Base::mM * Base::mN
                           + sizeof(typename Traits::EDataT) * Base::mM * Base::mN;

            // Arg test
            Base::mValid = deviceOp->IsSupportedArgument(mArgPtr[0].get()) &&
                           deviceOp->IsSupportedArgument(mArgPtr[1].get()) &&
                           deviceOp->IsSupportedArgument(mArgPtr[2].get()) &&
                           deviceOp->IsSupportedArgument(mArgPtr[3].get());

            return Base::mValid;
        }

        float operator()(StreamConfig const& streamConfig /*= StreamConfig{}*/)  override
        {
        if(!mArgPtr[0] || !mArgPtr[1] || !mArgPtr[2] || !mArgPtr[3] || !mInvokerPtr[0] || !mInvokerPtr[1] || !mInvokerPtr[2] || !mInvokerPtr[3] || !mParams || mParams->opCDE() == ContractionOpId_t::UNKNOWN)
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

            bool isValidRun = mInvokerPtr[0]->Run(mArgPtr[0].get(), streamConfig) &&
                              mInvokerPtr[1]->Run(mArgPtr[1].get(), streamConfig) &&
                              mInvokerPtr[2]->Run(mArgPtr[2].get(), streamConfig) &&
                              mInvokerPtr[3]->Run(mArgPtr[3].get(), streamConfig);

            if( isValidRun ) {}
                // Construct E from E_re and E_img

            return isValidRun;
        }

        float operator()(void const*                     alpha,
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
                         StreamConfig const& streamConfig /*= StreamConfig{}*/)  override
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

            bool isRunValid = mInvokerPtr[0]->Run(mArgPtr[0].get(), streamConfig) &&
                              mInvokerPtr[1]->Run(mArgPtr[1].get(), streamConfig) &&
                              mInvokerPtr[2]->Run(mArgPtr[2].get(), streamConfig) &&
                              mInvokerPtr[3]->Run(mArgPtr[3].get(), streamConfig);

            if( isRunValid ) {}
                //Construct E from E_re and E_img

            return isRunValid;
        }
    };

    template <typename DeviceOp, typename DataType>
    class ContractionSolutionImpl<
        DeviceOp,
        DataType,
        std::enable_if_t<std::is_same<DataType, hipFloatComplex>{} &&
                         std::is_same_v<typename MetaTraits<DeviceOp>::CDEOp,
                                        ck::tensor_operation::element_wise::Scale>>>
        : public ContractionSolution
    {
    protected :
        // Kernel Params
        std::unique_ptr<ContractionSolutionParams>                  mParams;
        std::vector<std::unique_ptr<ck::tensor_operation::device::BaseArgument>> mArgPtr;
        std::vector<std::unique_ptr<ck::tensor_operation::device::BaseInvoker>>  mInvokerPtr;

        void *A_re, *A_img, *B_re, *B_img, *E_re, *E_img;

    public:
        using Base   = ContractionSolution;

        ContractionSolutionImpl(ContractionSolutionImpl&& other)
        : mParams(std::move(other.mParams))
        , mArgPtr(std::move(other.mArgPtr))
        , mInvokerPtr(std::move(other.mInvokerPtr))
        {

        }

        ContractionSolutionImpl& operator=(ContractionSolutionImpl&& other)
        {
            if(this != &other)
            {
                mArgPtr     = std::move(other.mArgPtr);
                mInvokerPtr = std::move(other.mInvokerPtr);
            }
            return *this;
        }

        std::unique_ptr<ContractionSolutionParams> const& params() const override
        {
            return mParams;
        }

        size_t workspaceSize() const override
        {
            if(mValid)
            {
                return mDeviceOp->GetWorkSpaceSize(mArgPtr[0].get());
            }
            else
            {
                return 0;
            }
        }

        void resetArgs()
        {
            mArgPtr.reserve(4);
            mArgPtr[0].reset(nullptr);
            mArgPtr[1].reset(nullptr);
            mArgPtr[2].reset(nullptr);
            mArgPtr[3].reset(nullptr);
            mInvokerPtr.reserve(4);
            mInvokerPtr[0].reset(nullptr);
            mInvokerPtr[1].reset(nullptr);
            mInvokerPtr[2].reset(nullptr);
            mInvokerPtr[3].reset(nullptr);

            // Clear previous data
            Base::resetArgs();
        }

        ContractionSolutionImpl(std::unique_ptr<DeviceOp>&& deviceOp)
            : ContractionSolution(std::move(deviceOp),
                                  std::make_unique<ContractionSolutionParamsImpl<DeviceOp>>())
        {
        }

        bool initArgs(void const*                     alpha,
                      void const*                     A,
                      void const*                     B,
                      void const*                     beta,
                      void const*                     D,
                      void*                           E,
                      std::vector<std::size_t> const& a_ms_ks_lengths,
                      std::vector<std::size_t> const& a_ms_ks_strides,
                      std::vector<std::size_t> const& b_ns_ks_lengths,
                      std::vector<std::size_t> const& b_ns_ks_strides,
                      std::vector<std::size_t> const& ds_ms_ns_lengths,
                      std::vector<std::size_t> const& ds_ms_ns_strides,
                      std::vector<std::size_t> const& e_ms_ns_lengths,
                      std::vector<std::size_t> const& e_ms_ns_strides,
                      void*                           workspacePtr) override
        {
            using Traits = MetaTraits<DeviceOp>;

            // Clear previous data
            Base::resetArgs();

            // Promote to derived class for necessary functions such as
            // MakeArgumentPointer and MakeInvokerPointer.
            auto* deviceOp = dynamic_cast<DeviceOp*>(mDeviceOp.get());

            // Note: CK ALWAYS uses float for alpha / beta in contraction multipleD
            auto alphaF = 0.0f;

            if(alpha != nullptr)
            {
                alphaF = hiptensor::readVal<float>(alpha, convertToComputeType(HipDataType_v<typename Traits::EDataT>));
            }

            // CK has its own format for indices...
            auto toCKVec = [](std::vector<std::size_t> const& v) {
                return std::vector<ck::index_t>(v.begin(), v.end());
            };

            //TODO
            // Extract A_re, A_img from A
            // Extract B_re, B_img from B
            // Extract E_re, E_img from E

            // Initialize the argument pointer
            mArgPtr[0]
                = std::move(deviceOp->MakeArgumentPointer(A_re,
                                                          B_re,
                                                          std::array<const void*, 0>{},
                                                          E_re,
                                                          toCKVec(a_ms_ks_lengths),
                                                          toCKVec(a_ms_ks_strides),
                                                          toCKVec(b_ns_ks_lengths),
                                                          toCKVec(b_ns_ks_strides),
                                                          std::array<std::vector<ck::index_t>, 0>{},
                                                          std::array<std::vector<ck::index_t>, 0>{},
                                                          toCKVec(e_ms_ns_lengths),
                                                          toCKVec(e_ms_ns_strides),
                                                          typename Traits::AOp{},
                                                          typename Traits::BOp{},
                                                          typename Traits::CDEOp{alphaF}));

            // Attach the workspace pointer
            deviceOp->SetWorkSpacePointer(mArgPtr[0].get(), workspacePtr);

            // Initialize the invoker
            mInvokerPtr[0] = std::move(deviceOp->MakeInvokerPointer());

            mArgPtr[1]
                = std::move(deviceOp->MakeArgumentPointer(A_img,
                                                          B_img,
                                                          E_re,
                                                          E_re,
                                                          toCKVec(a_ms_ks_lengths),
                                                          toCKVec(a_ms_ks_strides),
                                                          toCKVec(b_ns_ks_lengths),
                                                          toCKVec(b_ns_ks_strides),
                                                          std::array<std::vector<ck::index_t>, 0>{},
                                                          std::array<std::vector<ck::index_t>, 0>{},
                                                          toCKVec(e_ms_ns_lengths),
                                                          toCKVec(e_ms_ns_strides),
                                                          typename Traits::AOp{},
                                                          typename Traits::BOp{},
                                                          typename Traits::CDEOp{alphaF * -1.0f}));

            mInvokerPtr[1] = std::move(deviceOp->MakeInvokerPointer());

            mArgPtr[2]
                = std::move(deviceOp->MakeArgumentPointer(A_re,
                                                          B_img,
                                                          std::array<const void*, 0>{},
                                                          E_img,
                                                          toCKVec(a_ms_ks_lengths),
                                                          toCKVec(a_ms_ks_strides),
                                                          toCKVec(b_ns_ks_lengths),
                                                          toCKVec(b_ns_ks_strides),
                                                          std::array<std::vector<ck::index_t>, 0>{},
                                                          std::array<std::vector<ck::index_t>, 0>{},
                                                          toCKVec(e_ms_ns_lengths),
                                                          toCKVec(e_ms_ns_strides),
                                                          typename Traits::AOp{},
                                                          typename Traits::BOp{},
                                                          typename Traits::CDEOp{alphaF}));

            mInvokerPtr[2] = std::move(deviceOp->MakeInvokerPointer());

            mArgPtr[3]
                = std::move(deviceOp->MakeArgumentPointer(A_img,
                                                          B_re,
                                                          E_img,
                                                          E_img,
                                                          toCKVec(a_ms_ks_lengths),
                                                          toCKVec(a_ms_ks_strides),
                                                          toCKVec(b_ns_ks_lengths),
                                                          toCKVec(b_ns_ks_strides),
                                                          std::array<std::vector<ck::index_t>, 0>{},
                                                          std::array<std::vector<ck::index_t>, 0>{},
                                                          toCKVec(e_ms_ns_lengths),
                                                          toCKVec(e_ms_ns_strides),
                                                          typename Traits::AOp{},
                                                          typename Traits::BOp{},
                                                          typename Traits::CDEOp{alphaF}));

            mInvokerPtr[3] = std::move(deviceOp->MakeInvokerPointer());

            // Fill problem metrics
            Base::mM = std::accumulate(e_ms_ns_lengths.begin(),
                                       e_ms_ns_lengths.begin() + Traits::DimsM,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            Base::mN = std::accumulate(e_ms_ns_lengths.begin() + Traits::DimsM,
                                       e_ms_ns_lengths.begin() + Traits::DimsM + Traits::DimsN,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            Base::mK = std::accumulate(a_ms_ks_lengths.begin() + Traits::DimsM,
                                       a_ms_ks_lengths.begin() + Traits::DimsM + Traits::DimsK,
                                       ck::index_t{1},
                                       std::multiplies<ck::index_t>{});

            // Byte count
            Base::mBytes = sizeof(typename Traits::ADataT) * Base::mM * Base::mK
                           + sizeof(typename Traits::BDataT) * Base::mK * Base::mN
                           + sizeof(typename Traits::EDataT) * Base::mM * Base::mN;

            // Arg test
            Base::mValid = deviceOp->IsSupportedArgument(mArgPtr[0].get()) &&
                           deviceOp->IsSupportedArgument(mArgPtr[1].get()) &&
                           deviceOp->IsSupportedArgument(mArgPtr[2].get()) &&
                           deviceOp->IsSupportedArgument(mArgPtr[3].get());

            return Base::mValid;
        }


        float operator()(StreamConfig const& streamConfig /*= StreamConfig{}*/) override
        {
        if(!mArgPtr[0] || !mArgPtr[1] || !mArgPtr[2] || !mArgPtr[3] || !mInvokerPtr[0] || !mInvokerPtr[1] || !mInvokerPtr[2] || !mInvokerPtr[3] || !mParams || mParams->opCDE() == ContractionOpId_t::UNKNOWN)
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

            return mInvokerPtr[0]->Run(mArgPtr[0].get(), streamConfig)
                   && mInvokerPtr[1]->Run(mArgPtr[1].get(), streamConfig)
                   && mInvokerPtr[2]->Run(mArgPtr[2].get(), streamConfig)
                   && mInvokerPtr[3]->Run(mArgPtr[3].get(), streamConfig);
        }

        float operator()(void const*                     alpha,
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
                                                            StreamConfig const& streamConfig /*= StreamConfig{}*/) override
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

            return mInvokerPtr[0]->Run(mArgPtr[0].get(), streamConfig)
                   && mInvokerPtr[1]->Run(mArgPtr[1].get(), streamConfig)
                   && mInvokerPtr[2]->Run(mArgPtr[2].get(), streamConfig)
                   && mInvokerPtr[3]->Run(mArgPtr[3].get(), streamConfig);
        }
    };

    //--------------------------------------------------------------------------------------------------

    template <ck::index_t NumDimM,
              ck::index_t NumDimN,
              ck::index_t NumDimK,
              typename ADataType,
              typename BDataType,
              typename DsDataType,
              typename EDataType,
              typename AElementwiseOperation,
              typename BElementwiseOperation,
              typename CDEElementwiseOperation>
    std::vector<std::unique_ptr<hiptensor::ContractionSolution>> enumerateContractionSolutions()
    {
        using ContractionOp
            = ck::tensor_operation::device::DeviceContractionMultipleD<NumDimM,
                                                                       NumDimN,
                                                                       NumDimK,
                                                                       ADataType,
                                                                       BDataType,
                                                                       DsDataType,
                                                                       EDataType,
                                                                       AElementwiseOperation,
                                                                       BElementwiseOperation,
                                                                       CDEElementwiseOperation>;

        using Factory
            = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<ContractionOp>;

        std::vector<std::unique_ptr<ContractionSolution>> result;
        for(auto& opPtr : Factory::GetInstances())
        {
            result.push_back(
                std::make_unique<ContractionSolutionImpl<ContractionOp, ADataType>>(std::move(opPtr)));
        }
        return result;
    }

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_SOLUTION_IMPL_HPP
