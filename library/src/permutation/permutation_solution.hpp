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

#ifndef HIPTENSOR_PERMUTATION_SOLUTION_HPP
#define HIPTENSOR_PERMUTATION_SOLUTION_HPP

#include <functional>
#include <memory>
#include <tuple>
#include <vector>

#include "permutation_meta_traits.hpp"
#include "permutation_solution_params.hpp"
#include "performance.hpp"
#include "util.hpp"

namespace hiptensor
{
    class PermutationSolution
    {
    public:
        // Due to unique_ptr ownership of members,
        // PermutationSolutions should also be considered unique.
        // This means disabling default and copy ctor
        PermutationSolution()                                      = delete;
        PermutationSolution(PermutationSolution const&)            = delete;
        virtual ~PermutationSolution()                             = default;
        PermutationSolution& operator=(PermutationSolution const&) = delete;

        // This class is intended to receive DeviceOp kernel pointers from
        // the CK generator and take ownership.
        PermutationSolution(std::unique_ptr<ck::tensor_operation::device::BaseOperator>&& deviceOp,
                            std::unique_ptr<PermutationSolutionParams>&&                  params);
        PermutationSolution(PermutationSolution&& other);
        PermutationSolution& operator=(PermutationSolution&& other);

        // Must specialize incoming arg handling
        virtual bool initArgs(void const*                 alpha,
                              void const*                 A,
                              void*                       B,
                              std::vector<std::size_t> const& a_lengths,
                              std::vector<std::size_t> const& a_strides,
                              const int32_t                   modeA[],
                              std::vector<std::size_t> const& b_lengths,
                              std::vector<std::size_t> const& b_strides,
                              const int32_t                   modeB[],
                              const hipDataType               typeScalar)
            = 0;

        float operator()(StreamConfig const& streamConfig = StreamConfig{});

        float operator()(void const*                 alpha,
                         void const*                 A,
                         void*                       B,
                         std::vector<std::size_t> const& a_lengths,
                         std::vector<std::size_t> const& a_strides,
                         const int32_t                   modeA[],
                         std::vector<std::size_t> const& b_lengths,
                         std::vector<std::size_t> const& b_strides,
                         const int32_t                   modeB[],
                         const hipDataType               typeScalar,
                         StreamConfig const&             streamConfig = StreamConfig{});

        /// Accessors

        // Problem can be solved with this kernel
        bool isValid() const;

        // Run-time solution parameters
        std::unique_ptr<PermutationSolutionParams> const& params() const;

        // Unique ID for the kernel
        size_t uid() const;

        // Problem dimension
        ck::index_t problemDim() const;

        // Byte count
        ck::index_t problemBytes() const;

        // Kernel's name encoding
        std::string kernelName() const;

        // Kernel's required workspace size
        size_t workspaceSize() const;

        // Reset all arguments
        void resetArgs();

    protected:
        // Derived runtime arguments
        ck::index_t mDim;
        ck::index_t mBytes;
        bool        mValid;

        // Kernel Params
        std::unique_ptr<PermutationSolutionParams>                  mParams;
        std::unique_ptr<ck::tensor_operation::device::BaseOperator> mDeviceOp;
        std::unique_ptr<ck::tensor_operation::device::BaseArgument> mArgPtr;
        std::unique_ptr<ck::tensor_operation::device::BaseInvoker>  mInvokerPtr;
    };

    template <typename InDataTypeTuple,
              typename OutDataTypeTuple,
              typename ElementwiseOperation,
              typename UnaryOperation,
              typename Scale,
              ck::index_t NumDim>
    std::vector<std::unique_ptr<hiptensor::PermutationSolution>> enumeratePermutationSolutions();

} // namespace hiptensor

#include "permutation_solution_impl.hpp"

#endif // HIPTENSOR_PERMUTATION_SOLUTION_HPP
