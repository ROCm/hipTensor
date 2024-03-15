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

#include "permutation_solution_registry.hpp"
#include "permutation_solution.hpp"

namespace hiptensor
{
    ////////////////////////////////////////////////
    /// Class PermutationSolutionRegistry::Query ///
    ////////////////////////////////////////////////

    // @cond
    PermutationSolutionRegistry::Query::Query(Query const& other)
        : mAllSolutions(other.mAllSolutions)
        , mSolutionHash(other.mSolutionHash)
    {
    }

    PermutationSolutionRegistry::Query&
        PermutationSolutionRegistry::Query::operator=(Query const& other)
    {
        if(&other != this)
        {
            mAllSolutions = other.mAllSolutions;
            mSolutionHash = other.mSolutionHash;
        }

        return *this;
    }

    PermutationSolutionRegistry::Query
        PermutationSolutionRegistry::Query::query(int32_t                dim,
                                                  hipDataType            typeIn,
                                                  hipDataType            typeOut,
                                                  hiptensorOperator_t    opElement,
                                                  hiptensorOperator_t    opUnary,
                                                  PermutationOpId_t      opScale,
                                                  uint32_t                threadDim) const
    {
        auto solutionHash = hashSolution(
            dim, typeIn, typeOut, opElement, opUnary, opScale, threadDim);

        if(auto solutions = mSolutionHash.find(solutionHash); solutions != mSolutionHash.end())
        {
            return Query(mSolutionHash.at(solutionHash));
        }

        return Query();
    }

    PermutationSolutionRegistry::Query
        PermutationSolutionRegistry::Query::query(int32_t dim) const
    {
        return query(hashDim(dim));
    }

    PermutationSolutionRegistry::Query
        PermutationSolutionRegistry::Query::query(hipDataType            typeIn,
                                                  hipDataType            typeOut) const
    {
        return query(hashTypesInOut(typeIn, typeOut));
    }

    PermutationSolutionRegistry::Query
        PermutationSolutionRegistry::Query::query(hiptensorOperator_t opElement,
                                                  hiptensorOperator_t opUnary) const
    {
        return query(hashElementOps(opElement, opUnary));
    }

    PermutationSolutionRegistry::Query
        PermutationSolutionRegistry::Query::query(PermutationOpId_t   opScale) const
    {
        return query(hashScaleOp(opScale));
    }

    PermutationSolutionRegistry::Query
        PermutationSolutionRegistry::Query::query(uint32_t threadDim) const
    {
        return query(hashThreadDim(threadDim));
    }

    PermutationSolutionRegistry::Query
        PermutationSolutionRegistry::Query::operator||(Query const& other) const
    {
        auto newQuery = *this;
        newQuery.addSolutions(other.mAllSolutions);
        return newQuery;
    }

    PermutationSolutionRegistry::Query
        PermutationSolutionRegistry::Query::operator&&(Query const& other) const
    {
        auto newQuery = Query();

        // Add only if both queries have the solution
        for(auto& solutionPair : other.mAllSolutions)
        {
            if(auto solution = mAllSolutions.find(solutionPair.first);
               solution != mAllSolutions.end())
            {
                newQuery.addSolution(solutionPair.second);
            }
        }

        return newQuery;
    }

    std::unordered_map<PermutationSolutionRegistry::Query::Uid, PermutationSolution*> const&
        PermutationSolutionRegistry::Query::solutions() const
    {
        return mAllSolutions;
    }

    uint32_t PermutationSolutionRegistry::Query::solutionCount() const
    {
        return mAllSolutions.size();
    }

    ///////////////
    /// Private ///
    ///////////////

    PermutationSolutionRegistry::Query::Query(std::vector<PermutationSolution*> const& solutions)
    {
        addSolutions(solutions);
    }

    PermutationSolutionRegistry::Query
        PermutationSolutionRegistry::Query::query(HashId queryHash) const
    {
        if(auto solutions = mSolutionHash.find(queryHash); solutions != mSolutionHash.end())
        {
            return Query(mSolutionHash.at(queryHash));
        }

        return Query();
    }

    /* static */
    PermutationSolutionRegistry::Query::HashId
        PermutationSolutionRegistry::Query::hashSolution(int32_t                dim,
                                                         hipDataType            typeIn,
                                                         hipDataType            typeOut,
                                                         hiptensorOperator_t    opElement,
                                                         hiptensorOperator_t    opUnary,
                                                         PermutationOpId_t      opScale,
                                                         uint32_t                threadDim)
    {
        return Hash{}(dim, typeIn, typeOut, opElement, opUnary, opScale, threadDim);
    }

    /* static */
    PermutationSolutionRegistry::Query::HashId
        PermutationSolutionRegistry::Query::hashDim(int32_t dim)
    {
        return Hash{}(dim);
    }

    /* static */
    PermutationSolutionRegistry::Query::HashId
        PermutationSolutionRegistry::Query::hashTypesInOut(hipDataType            typeIn,
                                                           hipDataType            typeOut)
    {
        return Hash{}(typeIn, typeOut);
    }

    /* static */
    PermutationSolutionRegistry::Query::HashId
        PermutationSolutionRegistry::Query::hashElementOps(hiptensorOperator_t opElement,
                                                           hiptensorOperator_t opUnary)
    {
        return Hash{}(opElement, opUnary);
    }

    /* static */
    PermutationSolutionRegistry::Query::HashId
        PermutationSolutionRegistry::Query::hashScaleOp(PermutationOpId_t      opScale)
    {
        return Hash{}(opScale);
    }

    /* static */
    PermutationSolutionRegistry::Query::HashId
        PermutationSolutionRegistry::Query::hashThreadDim(uint32_t threadDim)
    {
        return Hash{}(threadDim);
    }

    void PermutationSolutionRegistry::Query::addSolution(PermutationSolution* solution)
    {
        // Acquire unique ID and category ID per solution
        auto  solutionUid = solution->uid();
        auto& params      = solution->params();

        if(auto const& result = mAllSolutions.emplace(std::make_pair(solutionUid, solution));
           result.second == true)
        {
            auto solutionHash = hashSolution(params->dim(),
                                             params->typeIn(),
                                             params->typeOut(),
                                             params->opElement(),
                                             params->opUnary(),
                                             params->opScale(),
                                             solution->threadDim());

            auto dimHash = hashDim(params->dim());

            auto typesInOutHash = hashTypesInOut(params->typeIn(),
                                                 params->typeOut());

            auto elementOpsHash = hashElementOps(params->opElement(),
                                                 params->opUnary());

            auto scaleOpHash = hashScaleOp(params->opScale());

            auto threadDimHash = hashThreadDim(solution->threadDim());

            // Hash permutation solutions into categories and then register
            // into master list.
            mAllSolutions[solutionUid] = solution;
            mSolutionHash[solutionHash].push_back(solution);
            mSolutionHash[dimHash].push_back(solution);
            mSolutionHash[typesInOutHash].push_back(solution);
            mSolutionHash[elementOpsHash].push_back(solution);
            mSolutionHash[scaleOpHash].push_back(solution);
            mSolutionHash[threadDimHash].push_back(solution);
        }
        else
        {
#if !NDEBUG
            std::cout << "Unique solution: " << solutionUid << " already exists!" << std::endl;
#endif // !NDEBUG
        }
    }

    void PermutationSolutionRegistry::Query::addSolutions(
        std::vector<PermutationSolution*> const& solutions)
    {
        for(auto* soln : solutions)
        {
            addSolution(soln);
        }
    }

    void PermutationSolutionRegistry::Query::addSolutions(
        std::unordered_map<Uid, PermutationSolution*> const& solutions)
    {
        for(auto& solutionPair : solutions)
        {
            addSolution(solutionPair.second);
        }
    }

    /////////////////////////////////////////
    /// Class PermutationSolutionRegistry ///
    /////////////////////////////////////////

    void PermutationSolutionRegistry::registerSolutions(
        std::vector<std::unique_ptr<PermutationSolution>>&& solutions)
    {
        for(auto&& solution : solutions)
        {
            // Register with the query then take ownership
            mSolutionQuery.addSolution(solution.get());
            mSolutionStorage.push_back(std::move(solution));
        }
    }

    PermutationSolutionRegistry::Query const& PermutationSolutionRegistry::allSolutions() const
    {
        return mSolutionQuery;
    }

    uint32_t PermutationSolutionRegistry::solutionCount() const
    {
        return mSolutionStorage.size();
    }
    // @endcond

} // namespace hiptensor
