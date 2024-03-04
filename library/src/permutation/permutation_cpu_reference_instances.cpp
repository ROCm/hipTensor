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

#include "permutation_cpu_reference_instances.hpp"
#include "permutation_cpu_reference_impl.hpp"

namespace hiptensor
{
    PermutationCpuReferenceInstances::PermutationCpuReferenceInstances()
    {
        // Register all the solutions exactly once
        // 1d Permutation
        registerSolutions(
            enumerateReferenceSolutions<ck::Tuple<float>,
                                        ck::Tuple<float>,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::UnarySquare,
                                        ck::tensor_operation::element_wise::Scale,
                                        1>());

        registerSolutions(
            enumerateReferenceSolutions<ck::Tuple<ck::half_t>,
                                        ck::Tuple<ck::half_t>,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::UnarySquare,
                                        ck::tensor_operation::element_wise::Scale,
                                        1>());

        // 2d Permutation
        registerSolutions(
            enumerateReferenceSolutions<ck::Tuple<float>,
                                        ck::Tuple<float>,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::UnarySquare,
                                        ck::tensor_operation::element_wise::Scale,
                                        2>());

        registerSolutions(
            enumerateReferenceSolutions<ck::Tuple<ck::half_t>,
                                          ck::Tuple<ck::half_t>,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::UnarySquare,
                                          ck::tensor_operation::element_wise::Scale,
                                          2>());
        // 3d Permutation
        registerSolutions(
            enumerateReferenceSolutions<ck::Tuple<float>,
                                        ck::Tuple<float>,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::UnarySquare,
                                        ck::tensor_operation::element_wise::Scale,
                                        3>());

        registerSolutions(
            enumerateReferenceSolutions<ck::Tuple<ck::half_t>,
                                        ck::Tuple<ck::half_t>,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::UnarySquare,
                                        ck::tensor_operation::element_wise::Scale,
                                        3>());

        // 4d Permutation
        registerSolutions(
            enumerateReferenceSolutions<ck::Tuple<float>,
                                        ck::Tuple<float>,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::UnarySquare,
                                        ck::tensor_operation::element_wise::Scale,
                                        4>());

        registerSolutions(
            enumerateReferenceSolutions<ck::Tuple<ck::half_t>,
                                        ck::Tuple<ck::half_t>,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::UnarySquare,
                                        ck::tensor_operation::element_wise::Scale,
                                        4>());

        // 5d Permutation
        registerSolutions(
            enumerateReferenceSolutions<ck::Tuple<float>,
                                        ck::Tuple<float>,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::UnarySquare,
                                        ck::tensor_operation::element_wise::Scale,
                                        5>());

        registerSolutions(
            enumerateReferenceSolutions<ck::Tuple<ck::half_t>,
                                        ck::Tuple<ck::half_t>,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::UnarySquare,
                                        ck::tensor_operation::element_wise::Scale,
                                        5>());

        // 6d Permutation
        registerSolutions(
            enumerateReferenceSolutions<ck::Tuple<float>,
                                        ck::Tuple<float>,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::UnarySquare,
                                        ck::tensor_operation::element_wise::Scale,
                                        6>());

        registerSolutions(
            enumerateReferenceSolutions<ck::Tuple<ck::half_t>,
                                        ck::Tuple<ck::half_t>,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::UnarySquare,
                                        ck::tensor_operation::element_wise::Scale,
                                        6>());

    }
} // namespace hiptensor
