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

#ifndef HIPTENSOR_PERMUTATION_SOLUTION_INSTANCES_HPP
#define HIPTENSOR_PERMUTATION_SOLUTION_INSTANCES_HPP

#include <memory>

#include "permutation_solution_registry.hpp"
#include "singleton.hpp"

namespace hiptensor
{
    class PermutationSolutionInstances : public PermutationSolutionRegistry,
                                         public LazySingleton<PermutationSolutionInstances>
    {
    public:
        // For static initialization
        friend std::unique_ptr<PermutationSolutionInstances>
            std::make_unique<PermutationSolutionInstances>();

        ~PermutationSolutionInstances() = default;

    private:
        void PermutationSolution2DFloatSquareSquareInstances();
        void PermutationSolution2DFloatSquareThroughInstances();
        void PermutationSolution2DFloatThroughSquareInstances();
        void PermutationSolution2DFloatThroughThroughInstances();
        void PermutationSolution2DHalfSquareSquareInstances();
        void PermutationSolution2DHalfSquareThroughInstances();
        void PermutationSolution2DHalfThroughSquareInstances();
        void PermutationSolution2DHalfThroughThroughInstances();
        void PermutationSolution3DFloatSquareSquareInstances();
        void PermutationSolution3DFloatSquareThroughInstances();
        void PermutationSolution3DFloatThroughSquareInstances();
        void PermutationSolution3DFloatThroughThroughInstances();
        void PermutationSolution3DHalfSquareSquareInstances();
        void PermutationSolution3DHalfSquareThroughInstances();
        void PermutationSolution3DHalfThroughSquareInstances();
        void PermutationSolution3DHalfThroughThroughInstances();
        void PermutationSolution4DFloatSquareSquareInstances();
        void PermutationSolution4DFloatSquareThroughInstances();
        void PermutationSolution4DFloatThroughSquareInstances();
        void PermutationSolution4DFloatThroughThroughInstances();
        void PermutationSolution4DHalfSquareSquareInstances();
        void PermutationSolution4DHalfSquareThroughInstances();
        void PermutationSolution4DHalfThroughSquareInstances();
        void PermutationSolution4DHalfThroughThroughInstances();
        void PermutationSolution5DFloatSquareSquareInstances();
        void PermutationSolution5DFloatSquareThroughInstances();
        void PermutationSolution5DFloatThroughSquareInstances();
        void PermutationSolution5DFloatThroughThroughInstances();
        void PermutationSolution5DHalfSquareSquareInstances();
        void PermutationSolution5DHalfSquareThroughInstances();
        void PermutationSolution5DHalfThroughSquareInstances();
        void PermutationSolution5DHalfThroughThroughInstances();
        void PermutationSolution6DFloatSquareSquareInstances();
        void PermutationSolution6DFloatSquareThroughInstances();
        void PermutationSolution6DFloatThroughSquareInstances();
        void PermutationSolution6DFloatThroughThroughInstances();
        void PermutationSolution6DHalfSquareSquareInstances();
        void PermutationSolution6DHalfSquareThroughInstances();
        void PermutationSolution6DHalfThroughSquareInstances();
        void PermutationSolution6DHalfThroughThroughInstances();
        // Singleton: only one instance
        PermutationSolutionInstances();
        PermutationSolutionInstances(PermutationSolutionInstances const&)            = delete;
        PermutationSolutionInstances(PermutationSolutionInstances&&)                 = delete;
        PermutationSolutionInstances& operator=(PermutationSolutionInstances const&) = delete;
        PermutationSolutionInstances& operator=(PermutationSolutionInstances&&)      = delete;
    };

} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_SOLUTION_INSTANCES_HPP
