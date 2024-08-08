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

#ifndef HIPTENSOR_REDUCTION_SOLUTION_IMPL_HPP
#define HIPTENSOR_REDUCTION_SOLUTION_IMPL_HPP

#include <map>
#include <numeric>

#include "hash.hpp"

#include "ck/ck.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_reduce.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_blockwise.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_multiblock.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/utility/reduction_enums.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/utility/reduction_enums.hpp"

namespace std
{
    template <>
    struct hash<hiptensor::ReductionSolution>
    {
        size_t operator()(hiptensor::ReductionSolution const& s) const noexcept
        {
            return hash<hiptensor::ReductionSolutionParams>{}(*s.params());
        }
    };
}

namespace hiptensor
{
    template <typename DeviceOp, typename Enabler = void>
    class ReductionSolutionImpl;

    template <typename DeviceOp>
    class ReductionSolutionImpl<DeviceOp> : public ReductionSolution
    {
    public:
        ReductionSolutionImpl(std::unique_ptr<DeviceOp>&& deviceOp)
            : ReductionSolution(std::move(deviceOp),
                                std::make_unique<ReductionSolutionParamsImpl<DeviceOp>>())
        {
        }

        bool initArgs(std::vector<std::size_t> const& a_lengths,
                      std::vector<std::size_t> const& a_strides,
                      std::vector<int32_t> const&     a_modes,
                      std::vector<std::size_t> const& c_lengths,
                      std::vector<std::size_t> const& c_strides,
                      std::vector<int32_t> const&     c_modes,
                      double                          alpha,
                      double                          beta,
                      void const*                     A,
                      void*                           C,
                      hiptensorOperator_t             opReduce) override
        {
            using Base   = ReductionSolution;
            using Traits = MetaTraits<DeviceOp>;

            // Clear out the previous arguments
            resetArgs();

            // Promote to derived class for necessary functions such as
            // MakeArgumentPointer and MakeInvokerPointer.
            auto* deviceOp = dynamic_cast<DeviceOp*>(Base::mDeviceOp.get());
            if(deviceOp == nullptr)
            {
                return 0;
            }

            static_assert(Traits::TensorRank >= Traits::TensorNumReduceDim,
                          "TensorRank must be greater than or equal to TensorNumReduceDim");
            constexpr ck::index_t OutputDim
                = (Traits::TensorRank - Traits::TensorNumReduceDim)
                      ? (Traits::TensorRank - Traits::TensorNumReduceDim)
                      : 1;
            std::array<ck::index_t, Traits::TensorRank>         arrInLengths;
            std::array<ck::index_t, Traits::TensorRank>         arrInStrides;
            std::array<ck::index_t, OutputDim>                  arrOutLengths;
            std::array<ck::index_t, OutputDim>                  arrOutStrides;
            std::array<ck::index_t, Traits::TensorNumReduceDim> reduceDims;
            auto                                                toCKArr
                = [](auto const& v, auto& a) { std::copy(v.cbegin(), v.cend(), a.begin()); };
            auto findReduceModes
                = [](const std::vector<int32_t>& modeA, const std::vector<int32_t> modeD) {
                      std::vector<size_t> reduceModes;
                      for(int i = 0; i < modeA.size(); i++)
                      {
                          if(auto it = std::find(modeD.cbegin(), modeD.cend(), modeA[i]);
                             it == modeD.cend())
                          {
                              reduceModes.push_back(i);
                          }
                      }
                      return reduceModes;
                  };
            toCKArr(a_lengths, arrInLengths);
            toCKArr(a_strides.empty() ? hiptensor::stridesFromLengths(a_lengths) : a_strides,
                    arrInStrides);
            // @todo
            // toCKArr( a_strides.empty() ? hiptensor::stridesFromLengths(a_lengths, HIPTENSOR_DATA_LAYOUT_COL_MAJOR) : a_strides, arrInStrides);
            toCKArr(c_lengths, arrOutLengths);
            toCKArr(c_strides.empty() ? hiptensor::stridesFromLengths(c_lengths) : c_strides,
                    arrOutStrides);
            // toCKArr( c_strides.empty() ? hiptensor::stridesFromLengths(c_lengths, HIPTENSOR_DATA_LAYOUT_COL_MAJOR) : c_strides, arrOutStrides);
            toCKArr(findReduceModes(a_modes, c_modes), reduceDims);

            auto [in_elementwise_op, acc_elementwise_op]
                = reductionUnaryOperators(opReduce,
                                          hiptensor::elementsFromLengths(a_lengths)
                                              / hiptensor::elementsFromLengths(c_lengths));

            Base::mInvokerArgPtr = std::move(deviceOp->MakeArgumentPointer(arrInLengths,
                                                                           arrInStrides,
                                                                           arrOutLengths,
                                                                           arrOutStrides,
                                                                           reduceDims,
                                                                           alpha,
                                                                           beta,
                                                                           A,
                                                                           nullptr,
                                                                           C,
                                                                           nullptr,
                                                                           in_elementwise_op,
                                                                           acc_elementwise_op));

            // Initialize the invoker
            Base::mInvokerPtr = std::move(deviceOp->MakeInvokerPointer());

            // @todo
            // Fill problem metrics
            // Base::mDim = Traits::NDim;

            // Byte count
            // Base::mBytes = sizeof(typename Traits::InDataT) * Base::mDim
            // + sizeof(typename Traits::OutDataT) * Base::mDim;

            // Arg test
            Base::mValid = deviceOp->IsSupportedArgument(Base::mInvokerArgPtr.get());

            // Base::mThreadDim = findThreadDim(deviceOp->GetTypeString());

            return mValid;
        }
    };

    template <typename InDataType,
              typename AccDataType,
              typename OutDataType,
              int                 Rank,
              int                 NumReduceDim,
              hiptensorOperator_t opReduce,
              bool                PropagateNan,
              bool                OutputIndex>
    auto enumerateReductionSolutions()
    {
        constexpr auto ReduceOpId = convertHiptensorReduceOperatorToCk<opReduce>();

        using ReduceOperation = typename ck::reduce_binary_operator<ReduceOpId>::opType;
        using InElementwiseOperation =
            typename ck::reduce_unary_operator<ReduceOpId, true, true>::InElementwiseOperation;
        using AccElementwiseOperation =
            typename ck::reduce_unary_operator<ReduceOpId, true, true>::AccElementwiseOperation;

        using DeviceOp    = ck::tensor_operation::device::DeviceReduce<InDataType,
                                                                    AccDataType,
                                                                    OutDataType,
                                                                    Rank,
                                                                    NumReduceDim,
                                                                    ReduceOperation,
                                                                    InElementwiseOperation,
                                                                    AccElementwiseOperation,
                                                                    PropagateNan,
                                                                    OutputIndex>;
        using DeviceOpPtr = ck::tensor_operation::device::DeviceReducePtr<InDataType,
                                                                          AccDataType,
                                                                          OutDataType,
                                                                          Rank,
                                                                          NumReduceDim,
                                                                          ReduceOperation,
                                                                          InElementwiseOperation,
                                                                          AccElementwiseOperation,
                                                                          PropagateNan,
                                                                          OutputIndex>;

        using ReduceOpInstance = ck::tensor_operation::device::DeviceReduceMultiBlock<
            InDataType,
            AccDataType,
            OutDataType,
            Rank,
            NumReduceDim,
            ReduceOperation,
            InElementwiseOperation,
            AccElementwiseOperation,
            ck::InMemoryDataOperationEnum::Set,
            PropagateNan,
            OutputIndex,
            false, // HaveIndexInputIfOutputIndex
            256,
            4,
            64,
            1,
            1,
            0,
            1,
            1>;

        std::vector<std::unique_ptr<ReductionSolution>> result;
        result.push_back(std::make_unique<ReductionSolutionImpl<DeviceOp>>(
            std::make_unique<ReduceOpInstance>(ReduceOpInstance{})));
        return result;
    }

} // namespace hiptensor

#endif // HIPTENSOR_REDUCTION_SOLUTION_IMPL_HPP
