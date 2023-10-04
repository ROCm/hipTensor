/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "contraction_solution_instances.hpp"
#include "contraction_solution.hpp"

namespace hiptensor
{
    ContractionSolutionInstances::ContractionSolutionInstances()
    {
        // Register all the solutions exactly once
        // Bilinear f32
        registerSolutions(
            enumerateContractionSolutions<2,
                                          2,
                                          2,
                                          float,
                                          float,
                                          ck::Tuple<float>,
                                          float,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Bilinear>());

        // Bilinear f64
        registerSolutions(
            enumerateContractionSolutions<2,
                                          2,
                                          2,
                                          double,
                                          double,
                                          ck::Tuple<double>,
                                          double,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Bilinear>());

        // Scale f32
        registerSolutions(
            enumerateContractionSolutions<2,
                                          2,
                                          2,
                                          float,
                                          float,
                                          ck::Tuple<>,
                                          float,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Scale>());

        // Scale f64
        registerSolutions(
            enumerateContractionSolutions<2,
                                          2,
                                          2,
                                          double,
                                          double,
                                          ck::Tuple<>,
                                          double,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Scale>());
    }
} // namespace hiptensor
