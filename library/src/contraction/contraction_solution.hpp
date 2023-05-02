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

#ifndef HIPTENSOR_CONTRACTION_SOLUTION_HPP
#define HIPTENSOR_CONTRACTION_SOLUTION_HPP

#include <functional>
#include <memory>
#include <vector>

// CK includes
#include <contraction_bilinear.hpp>
#include <contraction_scale.hpp>
#include <device_contraction_multiple_d.hpp>
#include <element_wise_operation.hpp>

#include "contraction_meta_traits.hpp"

namespace hiptensor
{

    /**
     * \brief This enum decides the over the operation based on the inputs.
     * \details This enum decides the operation based on the in puts passed in the
     * hipTensorContractionGetWorkspaceSize
     */
    enum struct ContractionOpId_t : int32_t
    {
        SCALE    = 0, ///< \f${C=\alpha\mathcal{A}\mathcal{B}}\f$
        BILINEAR = 1, ///< \f${D=\alpha\mathcal{A}\mathcal{B}+\beta\mathcal{C}}\f$
        UNKNOWN,
    };

    struct ContractionSolution
    {
        using InitArgsFuncT = std::function<void(ContractionSolution&,
                                                 void const*,
                                                 void const*,
                                                 void const*,
                                                 void const*,
                                                 void const*,
                                                 void*,
                                                 std::vector<ck::index_t> const&,
                                                 std::vector<ck::index_t> const&,
                                                 std::vector<ck::index_t> const&,
                                                 std::vector<ck::index_t> const&,
                                                 std::vector<std::vector<ck::index_t>> const&,
                                                 std::vector<std::vector<ck::index_t>> const&,
                                                 std::vector<ck::index_t> const&,
                                                 std::vector<ck::index_t> const&)>;

        // Due to unique_ptr ownership of members,
        // ContractionSolutions should also be considered unique.
        // This means disabling default and copy ctor
        ContractionSolution()                                      = delete;
        ContractionSolution(ContractionSolution const&)            = delete;
        ~ContractionSolution()                                     = default;
        ContractionSolution& operator=(ContractionSolution const&) = delete;

        // Move ctor / assignement will inherit the other launcher's
        // members.
        ContractionSolution(ContractionSolution&& other);
        ContractionSolution& operator=(ContractionSolution&& other);

        /// This class is intended to receive DeviceOp kernel pointers from
        /// the CK generator. Wrap ownership of the CK kernel with generated
        /// arg and invoke pointers, such that invokation of these kernels
        /// is handled entirely in the operator() overloads.

        template <
            typename DeviceOp,
            typename std::enable_if_t<std::is_same_v<typename MetaTraits<DeviceOp>::CDEOp,
                                                     ck::tensor_operation::element_wise::Bilinear>,
                                      void*>
            = nullptr>
        ContractionSolution(std::unique_ptr<DeviceOp>&& deviceOp);

        template <
            typename DeviceOp,
            typename std::enable_if_t<std::is_same_v<typename MetaTraits<DeviceOp>::CDEOp,
                                                     ck::tensor_operation::element_wise::Scale>,
                                      void*>
            = nullptr>
        ContractionSolution(std::unique_ptr<DeviceOp>&& deviceOp);

        bool initArgs(void const*                                  alpha,
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
                      std::vector<ck::index_t> const&              e_ms_ns_strides);

        float operator()(StreamConfig const& streamConfig = StreamConfig{});

        float operator()(void const*                                  alpha,
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
                         StreamConfig const& streamConfig = StreamConfig{});

        bool isValid() const;

        ck::index_t       mM, mN, mK;
        ck::index_t       mBytes;
        bool              mValid;
        std::string       mKernelName;
        ContractionOpId_t mOpId;

        InitArgsFuncT                                               mInitArgs;
        std::unique_ptr<ck::tensor_operation::device::BaseOperator> mDeviceOp;
        std::unique_ptr<ck::tensor_operation::device::BaseArgument> mArgPtr;
        std::unique_ptr<ck::tensor_operation::device::BaseInvoker>  mInvokerPtr;
    };

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
    std::vector<hiptensor::ContractionSolution> enumerateContractionSolutions();

} // namespace hiptensor

#include "contraction_solution_impl.hpp"

#endif // HIPTENSOR_CONTRACTION_SOLUTION_HPP
