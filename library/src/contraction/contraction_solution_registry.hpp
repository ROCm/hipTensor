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

#include <memory>
#include <unordered_map>
#include <vector>

#include "contraction_solution.hpp"
#include "types.hpp"

namespace hiptensor
{
    class ContractionSolutionRegistry
    {
    public:
        ContractionSolutionRegistry()  = default;
        ~ContractionSolutionRegistry() = default;

        ContractionSolutionRegistry(ContractionSolutionRegistry const&)            = delete;
        ContractionSolutionRegistry(ContractionSolutionRegistry&&)                 = delete;
        ContractionSolutionRegistry* operator=(ContractionSolutionRegistry const&) = delete;
        ContractionSolutionRegistry* operator=(ContractionSolutionRegistry&&)      = delete;

        // Import contraction solutions for the registry to manage
        void registerSolutions(std::vector<std::unique_ptr<ContractionSolution>>&& solutions);

        std::vector<ContractionSolution*> querySolutions(int32_t             dimsM,
                                                         int32_t             dimsN,
                                                         int32_t             dimsK,
                                                         hipDataType         typeA,
                                                         hipDataType         typeB,
                                                         hipDataType         typeC,
                                                         hipDataType         typeD,
                                                         hiptensorOperator_t opA,
                                                         hiptensorOperator_t opB,
                                                         ContractionOpId_t   opCDE);

        ContractionSolution* querySolution(size_t solutionUid);

        std::vector<ContractionSolution*> allSolutions();

    private:
        std::unordered_map<std::size_t, std::unique_ptr<ContractionSolution>> mAllSolutions;
        std::unordered_map<std::size_t, std::vector<ContractionSolution*>>    mSolutionHash;
    };

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_SOLUTION_REGISTRY_HPP
