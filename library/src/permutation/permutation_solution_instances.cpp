/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "permutation_solution_instances.hpp"
#include "permutation_solution.hpp"

// Ensure access to
#include "device/hiptensor_permutation_scale_instances.hpp"

namespace hiptensor
{
    PermutationSolutionInstances::PermutationSolutionInstances()
    {
        PermutationSolution2DFloatNoopInstances();
        PermutationSolution2DFloatSquareSquareInstances();
        PermutationSolution2DFloatSquareThroughInstances();
        PermutationSolution2DFloatThroughSquareInstances();
        PermutationSolution2DFloatThroughThroughInstances();
        PermutationSolution2DHalfNoopInstances();
        PermutationSolution2DHalfSquareSquareInstances();
        PermutationSolution2DHalfSquareThroughInstances();
        PermutationSolution2DHalfThroughSquareInstances();
        PermutationSolution2DHalfThroughThroughInstances();
        PermutationSolution3DFloatNoopInstances();
        PermutationSolution3DFloatSquareSquareInstances();
        PermutationSolution3DFloatSquareThroughInstances();
        PermutationSolution3DFloatThroughSquareInstances();
        PermutationSolution3DFloatThroughThroughInstances();
        PermutationSolution3DHalfNoopInstances();
        PermutationSolution3DHalfSquareSquareInstances();
        PermutationSolution3DHalfSquareThroughInstances();
        PermutationSolution3DHalfThroughSquareInstances();
        PermutationSolution3DHalfThroughThroughInstances();
        PermutationSolution4DFloatNoopInstances();
        PermutationSolution4DFloatSquareSquareInstances();
        PermutationSolution4DFloatSquareThroughInstances();
        PermutationSolution4DFloatThroughSquareInstances();
        PermutationSolution4DFloatThroughThroughInstances();
        PermutationSolution4DHalfNoopInstances();
        PermutationSolution4DHalfSquareSquareInstances();
        PermutationSolution4DHalfSquareThroughInstances();
        PermutationSolution4DHalfThroughSquareInstances();
        PermutationSolution4DHalfThroughThroughInstances();
        PermutationSolution5DFloatNoopInstances();
        PermutationSolution5DFloatSquareSquareInstances();
        PermutationSolution5DFloatSquareThroughInstances();
        PermutationSolution5DFloatThroughSquareInstances();
        PermutationSolution5DFloatThroughThroughInstances();
        PermutationSolution5DHalfNoopInstances();
        PermutationSolution5DHalfSquareSquareInstances();
        PermutationSolution5DHalfSquareThroughInstances();
        PermutationSolution5DHalfThroughSquareInstances();
        PermutationSolution5DHalfThroughThroughInstances();
        PermutationSolution6DFloatNoopInstances();
        PermutationSolution6DFloatSquareSquareInstances();
        PermutationSolution6DFloatSquareThroughInstances();
        PermutationSolution6DFloatThroughSquareInstances();
        PermutationSolution6DFloatThroughThroughInstances();
        PermutationSolution6DHalfNoopInstances();
        PermutationSolution6DHalfSquareSquareInstances();
        PermutationSolution6DHalfSquareThroughInstances();
        PermutationSolution6DHalfThroughSquareInstances();
        PermutationSolution6DHalfThroughThroughInstances();
    }
} // namespace hiptensor
