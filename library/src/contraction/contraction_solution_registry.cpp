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

#include "contraction_solution_registry.hpp"

namespace hiptensor
{

    void ContractionSolutionRegistry::registerSolutions(
        std::vector<std::unique_ptr<ContractionSolution>>&& solutions)
    {
        for(auto&& soln : solutions)
        {
            // Acquire unique ID and category ID per kernel
            auto solutionUid  = soln->uuid();
            auto hashCategory = std::hash<ContractionSolution>{}(*soln);

            if(auto solution = mAllSolutions.find(solutionUid); solution == mAllSolutions.end())
            {
                // Hash contraction solutions into categories and then register
                // into master list.
                mSolutionHash[hashCategory].push_back(soln.get());
                mAllSolutions[solutionUid] = std::move(soln);
            }
            else
            {
#if !NDEBUG
                std::cout << "Unique solution: " << solutionUid << " already exists!" << std::endl;
#endif // !NDEBUG
            }
        }
    }

    std::vector<ContractionSolution*>
        ContractionSolutionRegistry::querySolutions(int32_t             dimsM,
                                                    int32_t             dimsN,
                                                    int32_t             dimsK,
                                                    hipDataType         typeA,
                                                    hipDataType         typeB,
                                                    hipDataType         typeC,
                                                    hipDataType         typeD,
                                                    hiptensorOperator_t opA,
                                                    hiptensorOperator_t opB,
                                                    ContractionOpId_t   opCDE)
    {
#if !NDEBUG
        std::cout << "Requesting: " << std::endl;
        std::cout << "M:" << dimsM << " N: " << dimsN << " K: " << dimsK << " A: " << typeA
                  << " B: " << typeB << " C: " << typeC << " D: " << typeD << " OpA: " << opA
                  << " OpB: " << opB << " opCDE: " << (int)opCDE << std::endl;
#endif // !NDEBUG

        auto key = Hash{}(dimsM, dimsN, dimsK, typeA, typeB, typeC, typeD, opA, opB, opCDE);

#if !NDEBUG
        std::cout << "Hash: " << key << std::endl;
#endif // !NDEBUG

        if(auto solutions = mSolutionHash.find(key); solutions != mSolutionHash.end())
        {
            return solutions->second;
        }

        return std::vector<ContractionSolution*>{};
    }

    ContractionSolution* ContractionSolutionRegistry::querySolution(size_t solutionUid)
    {
        if(auto solution = mAllSolutions.find(solutionUid); solution != mAllSolutions.end())
        {
            return solution->second.get();
        }

        return nullptr;
    }

    std::vector<ContractionSolution*> ContractionSolutionRegistry::allSolutions()
    {
        std::vector<ContractionSolution*> result;
        for(auto it = mAllSolutions.begin(); it != mAllSolutions.end(); it++)
        {
            result.push_back(it->second.get());
        }

        return result;
    }

} // namespace hiptensor
