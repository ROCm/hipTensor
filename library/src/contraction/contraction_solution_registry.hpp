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

#ifndef HIPTENSOR_CONTRACTION_SOLUTION_REGISTRY_HPP
#define HIPTENSOR_CONTRACTION_SOLUTION_REGISTRY_HPP

#include "hash.hpp"

namespace hiptensor
{
    class ContractionSolutionRegistry
    {
        ContractionSolutionRegistry()  = default;
        ~ContractionSolutionRegistry() = default;

        void registerSolutions(std::vector<ContractionSolution>&& solutions)
        {
            foreach(auto&& soln : solutions)
            {
                // Hash contraction solutions into categories.
                auto hash = Hash<ContractionSolution>{}(soln);
                mSolutionHash[hash].push_back(std::move(soln));
            }
        }

        std::unordered_map<std::size_t, std::vector<ContractionSolution>> mSolutionHash;
    }

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_SOLUTION_REGISTRY_HPP
