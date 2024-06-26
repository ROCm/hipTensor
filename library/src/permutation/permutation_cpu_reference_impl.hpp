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

#ifndef HIPTENSOR_PERMUTATION_CPU_REFERENCE_IMPL_HPP
#define HIPTENSOR_PERMUTATION_CPU_REFERENCE_IMPL_HPP

// Std includes
#include <array>
#include <list>
#include <numeric>
#include <vector>

// CK includes
#include <combined_element_wise_operation.hpp>
#include <device_elementwise_dynamic_vector_dims_impl.hpp>
#include <host_tensor.hpp>

#include "permutation_meta_traits.hpp"
#include "permutation_solution.hpp"

namespace hiptensor
{
    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename ElementOp,
              ck::index_t NumDim>
    struct ReferencePermutation
        : public ck::tensor_operation::device::DeviceElementwiseImpl<InDataTypeTuple,
                                                                     OutDataTypeTuple,
                                                                     ElementOp,
                                                                     NumDim,
                                                                     32,
                                                                     16,
                                                                     32,
                                                                     4,
                                                                     4,
                                                                     ck::Sequence<1, 0>,
                                                                     ck::Sequence<1>,
                                                                     ck::Sequence<1>>
    {
        using BaseArgument = ck::tensor_operation::device::BaseArgument;
        using BaseInvoker  = ck::tensor_operation::device::BaseInvoker;
        using index_t      = ck::index_t;

        using mInDataType  = ck::tuple_element_t<0, InDataTypeTuple>;
        using mOutDataType = ck::tuple_element_t<0, OutDataTypeTuple>;

        static constexpr int NumInput  = InDataTypeTuple::Size();
        static constexpr int NumOutput = OutDataTypeTuple::Size();

        // Argument
        struct Argument : public BaseArgument
        {
            Argument(const std::array<index_t, NumDim>                        lengths,
                     const std::array<std::array<index_t, NumDim>, NumInput>  inStridesArray,
                     const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                     const void*                                              in_dev_buffers,
                     void*                                                    out_dev_buffers,
                     ElementOp                                                elementwise_op)
                : BaseArgument()
                , mLengths(lengths)
                , mInStrides(inStridesArray)
                , mOutStrides(outStridesArray)
                , mElementOp(elementwise_op)
            {
                mInput  = (mInDataType*)in_dev_buffers;
                mOutput = (mOutDataType*)out_dev_buffers;
            }

            Argument(Argument const&)            = default;
            Argument& operator=(Argument const&) = default;
            ~Argument()                          = default;

            const mInDataType* mInput;
            mOutDataType*      mOutput;

            std::array<index_t, NumDim>                        mLengths;
            std::array<std::array<index_t, NumDim>, NumInput>  mInStrides;
            std::array<std::array<index_t, NumDim>, NumOutput> mOutStrides;

            ElementOp mElementOp;
        };

        // Invoker
        struct Invoker : public BaseInvoker
        {
            using Argument = ReferencePermutation::Argument;

            float Run(const Argument& arg)
            {
                int  modeSize     = arg.mLengths.size();
                auto elementCount = hiptensor::elementsFromLengths(
                    std::vector<index_t>(std::begin(arg.mLengths), std::end(arg.mLengths)));

#if HIPTENSOR_DATA_LAYOUT_COL_MAJOR
                // Sort the output strides to calculate output tensor lengths
                std::vector<int> outStrides(std::begin(arg.mOutStrides[0]),
                                            std::end(arg.mOutStrides[0]));
                std::sort(outStrides.begin(), outStrides.end());
                assert(outStrides.front() == 1);

                // Map the lengths in output to its indices( using list to cover redundant lengths )
                std::unordered_map<int32_t, std::list<int32_t>> bLengthToIndex;
                int                                             prevLength = 1, i;
                for(i = 1; i < modeSize; i++)
                {
                    bLengthToIndex[outStrides[i] / prevLength].push_back(i - 1);
                    prevLength = outStrides[i];
                }
                bLengthToIndex[elementCount / prevLength].push_back(i - 1);
#else
                // Sort the output strides to calculate output tensor lengths
                std::vector<int> outStrides(std::begin(arg.mOutStrides[0]),
                                            std::end(arg.mOutStrides[0]));
                std::sort(outStrides.rbegin(), outStrides.rend());
                assert(outStrides.back() == 1);

                // Map the lengths in output to its indices( using list to cover redundant lengths )
                std::unordered_map<int32_t, std::list<int32_t>> bLengthToIndex;
                int                                             prevLength = 1, i;
                for(i = modeSize - 2; i >= 0; i--)
                {
                    bLengthToIndex[outStrides[i] / prevLength].push_back(i + 1);
                    prevLength = outStrides[i];
                }
                bLengthToIndex[elementCount / prevLength].push_back(i + 1);
#endif

                // From computed output lengths and argument's input lengths,
                // create a mode map between input and output
                std::map<int, int> modeATomodeBmap;
                for(int i = 0; i < modeSize; i++)
                {
                    modeATomodeBmap[i] = bLengthToIndex[arg.mLengths[i]].front();
                    bLengthToIndex[arg.mLengths[i]].pop_front();
                }

                // Find the write offset and index in output for every input element
                auto bIndices = std::vector<int32_t>(modeSize, 0);
                for(int elementIndex = 0; elementIndex < elementCount; elementIndex++)
                {
                    auto index = elementIndex;
#if HIPTENSOR_DATA_LAYOUT_COL_MAJOR
                    for(int modeIndex = 0; modeIndex < modeSize; modeIndex++)
                    {
                        bIndices[modeATomodeBmap[modeIndex]] = index % arg.mLengths[modeIndex];
                        index /= arg.mLengths[modeIndex];
                    }
                    auto bOffset = std::inner_product(
                        bIndices.begin(), bIndices.end(), std::begin(outStrides), 0);
#else // HIPTENSOR_DATA_LAYOUT_COL_MAJOR
                    for(int modeIndex = modeSize - 1; modeIndex >= 0; modeIndex--)
                    {
                        bIndices[modeATomodeBmap[modeIndex]] = index % arg.mLengths[modeIndex];
                        index /= arg.mLengths[modeIndex];
                    }
                    auto bOffset = std::inner_product(
                        bIndices.rbegin(), bIndices.rend(), std::rbegin(outStrides), 0);
#endif // HIPTENSOR_DATA_LAYOUT_COL_MAJOR

                    // Perforn sequence of unary, scale operations on input
                    arg.mElementOp(arg.mOutput[bOffset], arg.mInput[elementIndex]);
                }
                return 0;
            }

            float Run(const BaseArgument* p_arg,
                      const StreamConfig& /* stream_config */ = StreamConfig{}) override
            {
                return Run(*dynamic_cast<const Argument*>(p_arg));
            }
        };

        static constexpr bool IsValidCompilationParameter()
        {
            // TODO: properly implement this check
            return true;
        }

        bool IsSupportedArgument(const BaseArgument*) override
        {
            return true;
        }

        static auto
            MakeArgument(const std::array<index_t, NumDim>                        lengths,
                         const std::array<std::array<index_t, NumDim>, NumInput>  inStridesArray,
                         const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                         const void*                                              in_dev_buffers,
                         void*                                                    out_dev_buffers,
                         ElementOp                                                elementwise_op)
        {
            return Argument{lengths,
                            inStridesArray,
                            outStridesArray,
                            in_dev_buffers,
                            out_dev_buffers,
                            elementwise_op};
        }

        std::unique_ptr<BaseArgument> MakeArgumentPointer(
            const std::array<index_t, NumDim>                        lengths,
            const std::array<std::array<index_t, NumDim>, NumInput>  inStridesArray,
            const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
            const void*                                              in_dev_buffers,
            void*                                                    out_dev_buffers,
            ElementOp                                                elementwise_op)
        {
            return std::make_unique<Argument>(Argument{lengths,
                                                       inStridesArray,
                                                       outStridesArray,
                                                       in_dev_buffers,
                                                       out_dev_buffers,
                                                       elementwise_op});
        }

        static auto MakeInvoker()
        {
            return Invoker{};
        }

        std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
        {
            return std::make_unique<Invoker>(Invoker{});
        }

        std::string GetTypeString() const override
        {
            auto str = std::stringstream();

            // clang-format off
            str << "ReferencePermutation<";
            str << NumDim << ", ";
            str << 1 << ">";
            // clang-format on

            return str.str();
        }
    };

    // Partial specialize for reference permutation
    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename Aop,
              typename Bop,
              typename Scale,
              ck::index_t NumDim>
    struct MetaTraits<
        ReferencePermutation<InDataTypeTuple,
                             OutDataTypeTuple,
                             ck::tensor_operation::element_wise::UnaryCombinedOp<Aop, Scale, Bop>,
                             NumDim>>
        : public MetaTraits<ck::tensor_operation::device::DeviceElementwise<
              InDataTypeTuple,
              OutDataTypeTuple,
              ck::tensor_operation::element_wise::UnaryCombinedOp<Aop, Scale, Bop>,
              NumDim>>
    {
    };

    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename Aop,
              typename Bop,
              typename Scale,
              ck::index_t NumDim>
    auto enumerateReferenceSolutions()
    {
        using ReferenceOp = ReferencePermutation<
            InDataTypeTuple,
            OutDataTypeTuple,
            ck::tensor_operation::element_wise::UnaryCombinedOp<Aop, Scale, Bop>,
            NumDim>;

        auto solution = std::make_unique<PermutationSolutionImpl<ReferenceOp>>(
            std::make_unique<ReferenceOp>());

        auto result = std::vector<std::unique_ptr<PermutationSolution>>();
        result.push_back(std::move(solution));

        return result;
    }

} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_CPU_REFERENCE_IMPL_HPP
