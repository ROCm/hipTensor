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

#ifndef HIPTENSOR_REDUCTION_SOLUTION_INSTANCES_HPP
#define HIPTENSOR_REDUCTION_SOLUTION_INSTANCES_HPP

#include <memory>

#include "reduction_solution_registry.hpp"
#include "singleton.hpp"

namespace hiptensor
{
    class ReductionSolutionInstances : public ReductionSolutionRegistry,
                                       public LazySingleton<ReductionSolutionInstances>
    {
    public:
        // For static initialization
        friend std::unique_ptr<ReductionSolutionInstances>
            std::make_unique<ReductionSolutionInstances>();

        ~ReductionSolutionInstances() = default;

    private:
        // Singleton: only one instance
        ReductionSolutionInstances();
        ReductionSolutionInstances(ReductionSolutionInstances const&)            = delete;
        ReductionSolutionInstances(ReductionSolutionInstances&&)                 = delete;
        ReductionSolutionInstances& operator=(ReductionSolutionInstances const&) = delete;
        ReductionSolutionInstances& operator=(ReductionSolutionInstances&&)      = delete;
    };

} // namespace hiptensor

#endif // HIPTENSOR_REDUCTION_SOLUTION_INSTANCES_HPP
