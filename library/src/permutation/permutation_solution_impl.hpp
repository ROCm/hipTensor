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

#ifndef HIPTENSOR_PERMUTATION_SOLUTION_IMPL_HPP
#define HIPTENSOR_PERMUTATION_SOLUTION_IMPL_HPP

#include <numeric>
#include <map>

#include "permutation_solution.hpp"
#include "hash.hpp"

namespace std
{
    template <>
    struct hash<hiptensor::PermutationSolution>
    {
        size_t operator()(hiptensor::PermutationSolution const& s) const noexcept
        {
            return hash<hiptensor::PermutationSolutionParams>{}(*s.params());
        }
    };
}

namespace hiptensor
{
    template <typename DeviceOp, typename Enabler = void>
    class PermutationSolutionImpl;

    template <typename DeviceOp>
    class PermutationSolutionImpl<DeviceOp>
        : public PermutationSolution
    {
    public:
        PermutationSolutionImpl(std::unique_ptr<DeviceOp>&& deviceOp)
            : PermutationSolution(std::move(deviceOp),
                                  std::make_unique<PermutationSolutionParamsImpl<DeviceOp>>())
        {
        }

        bool initArgs(void const*                       alpha,
                      void const*                       A,
                      void*                             B,
                      std::vector<std::size_t> const&   a_lengths,
                      std::vector<std::size_t> const&   a_strides,
                      const int32_t                     modeA[],
                      std::vector<std::size_t> const&   b_lengths,
                      std::vector<std::size_t> const&   b_strides,
                      const int32_t                     modeB[],
                      const hipDataType                 typeScalar) override
        {
            using Base   = PermutationSolution;
            using Traits = MetaTraits<DeviceOp>;

            // Clear out the previous arguments
            resetArgs();

            // Promote to derived class for necessary functions such as
            // MakeArgumentPointer and MakeInvokerPointer.
            auto* deviceOp = dynamic_cast<DeviceOp*>(Base::mDeviceOp.get());

            // Note: CK ALWAYS uses float for alpha in permutation
            ScalarData alphaF;

            if(alpha != nullptr)
            {
                alphaF = hiptensor::readVal<ScalarData>(alpha, convertToComputeType(typeScalar));
            }

            // CK has its own format for indices...
            auto toCKArr = [](std::vector<std::size_t> const& v, std::array<ck::index_t, Traits::NDim>& a) {
                std::copy_n(v.begin(), Traits::NDim, a.begin());
            };
 
            // Re-construct strides from lengths, assuming packed.
            std::array<ck::index_t, Traits::NDim> aStrides, bStrides, bStridesCk, abLengths;

            std::map<char, ck::index_t> toModeA;
            for(int i = 0; i < Traits::NDim; i++)
            {
                toModeA[modeB[i]] = i;
            }

            toCKArr(hiptensor::stridesFromLengths(a_lengths, HIPTENSOR_DATA_LAYOUT_COL_MAJOR), aStrides);
            toCKArr(hiptensor::stridesFromLengths(b_lengths, HIPTENSOR_DATA_LAYOUT_COL_MAJOR), bStrides);
            for(int i = 0; i < Traits::NDim; i++)
                bStridesCk[toModeA[modeA[i]]] = bStrides[i];

            toCKArr(a_lengths, abLengths);

            // Initialize the argument pointer
            Base::mArgPtr = std::move(deviceOp->MakeArgumentPointer(
                abLengths,
                {aStrides},
                {bStridesCk},
                {A},
                {B},
                typename Traits::ElementOp{},
                typename Traits::UnaryOp{},
                typename Traits::ScaleOp{alphaF}));

            // Initialize the invoker
            Base::mInvokerPtr = std::move(deviceOp->MakeInvokerPointer());

            // Fill problem metrics
            Base::mDim = Traits::NDim;

            // Byte count
            Base::mBytes = sizeof(typename Traits::InDataT) * Base::mDim
                           + sizeof(typename Traits::OutDataT) * Base::mDim;

            // Arg test
            Base::mValid = deviceOp->IsSupportedArgument(Base::mArgPtr.get());

            return mValid;
        }
    };

    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename ElementwiseOperation,
              typename UnaryOperation,
              typename Scale,
              ck::index_t NumDim>
    std::vector<std::unique_ptr<hiptensor::PermutationSolution>> enumeratePermutationSolutions()
    {
        using PermutationOp
             = ck::tensor_operation::device::DeviceElementwise<InDataTypeTuple,
                                                               OutDataTypeTuple,
                                                               ElementwiseOperation,
                                                               UnaryOperation,
                                                               Scale,
                                                               NumDim>;

        using Factory
            = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<PermutationOp>;

        std::vector<std::unique_ptr<PermutationSolution>> result;
        for(auto& opPtr : Factory::GetInstances())
        {
            result.push_back(
                std::make_unique<PermutationSolutionImpl<PermutationOp>>(std::move(opPtr)));
        }
        return result;
    }

} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_SOLUTION_IMPL_HPP
