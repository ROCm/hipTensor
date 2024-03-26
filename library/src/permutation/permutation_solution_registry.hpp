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

#ifndef HIPTENSOR_PERMUTATION_SOLUTION_REGISTRY_HPP
#define HIPTENSOR_PERMUTATION_SOLUTION_REGISTRY_HPP

#include <memory>
#include <unordered_map>
#include <vector>

#include "permutation_types.hpp"
#include "data_types.hpp"
#include "singleton.hpp"

namespace hiptensor
{
    // @cond
    class PermutationSolution;

    class PermutationSolutionRegistry
    {
    public:
        class Query
        {
        public:
            friend class PermutationSolutionRegistry;
            using Uid    = std::size_t;
            using HashId = std::size_t;

            Query()  = default;
            ~Query() = default;
            Query(Query const& other);
            Query& operator=(Query const& other);

            /// Subsequent queries that may be performed on the current query object.
            /// E.g. in this context, query further parameters.

            // By solution type
            Query query(int32_t                dim,
                        hipDataType            typeIn,
                        hipDataType            typeOut,
                        hiptensorOperator_t    opA,
                        hiptensorOperator_t    opB,
                        PermutationOpId_t      opScale) const;

            // By full solution type
            Query query(int32_t                dim,
                        hipDataType            typeIn,
                        hipDataType            typeOut,
                        hiptensorOperator_t    opA,
                        hiptensorOperator_t    opB,
                        PermutationOpId_t      opScale,
                        uint32_t               threadDim) const;

            // By dimension
            Query query(int32_t dim) const;

            // By data types
            Query query(hipDataType            typeIn,
                        hipDataType            typeOut) const;

            // By element-wise operations
            Query query(hiptensorOperator_t opA,
                        hiptensorOperator_t opB) const;

            // By permutation operation
            Query query(PermutationOpId_t  opScale) const;

            // By thread dimension
            Query query(uint32_t threadDim) const;

            // union
            Query operator||(Query const& other) const;

            // intersection
            Query operator&&(Query const& other) const;

            // Full map of Uid to PermutationSolution*
            std::unordered_map<Uid, PermutationSolution*> const& solutions() const;

            uint32_t solutionCount() const;

            // Internal ctor
            Query(std::vector<PermutationSolution*> const& solutions);

        private:
            // Query by explicit hash
            Query query(HashId queryHash) const;

            // Hashing helpers
            static HashId hashSolution(int32_t                dim,
                                       hipDataType            typeIn,
                                       hipDataType            typeOut,
                                       hiptensorOperator_t    opA,
                                       hiptensorOperator_t    opB,
                                       PermutationOpId_t      opScale);

            static HashId hashSolution(int32_t                dim,
                                       hipDataType            typeIn,
                                       hipDataType            typeOut,
                                       hiptensorOperator_t    opA,
                                       hiptensorOperator_t    opB,
                                       PermutationOpId_t      opScale,
                                       uint32_t               threadDim);

            static HashId hashDim(int32_t dim);

            static HashId hashTypesInOut(hipDataType            typeIn,
                                         hipDataType            typeOut);

            static HashId hashElementOps(hiptensorOperator_t    opA,
                                         hiptensorOperator_t    opB);

            static HashId hashScaleOp(PermutationOpId_t  opScale);

            static HashId hashThreadDim(uint32_t threadDim);

            // Adding solutions to the query
            void addSolution(PermutationSolution* solution);
            void addSolutions(std::vector<PermutationSolution*> const& solutions);
            void addSolutions(std::unordered_map<Uid, PermutationSolution*> const& solutions);

        private: // members
            // This is the has of all solutions, by unique Uid
            // [Key = KernelUid, element = PermutationSolution*]
            std::unordered_map<Uid, PermutationSolution*> mAllSolutions;

            // This is the contextual query hash
            // [Key = Query hash, Element = PermutationSolution*]
            std::unordered_map<HashId, std::vector<PermutationSolution*>> mSolutionHash;
        };

    protected:
        // Move only
        PermutationSolutionRegistry()                                              = default;
        PermutationSolutionRegistry(PermutationSolutionRegistry&&)                 = default;
        PermutationSolutionRegistry& operator=(PermutationSolutionRegistry&&)      = default;
        PermutationSolutionRegistry(PermutationSolutionRegistry const&)            = delete;
        PermutationSolutionRegistry& operator=(PermutationSolutionRegistry const&) = delete;

        // Import permutation solutions for the registry to manage
        void registerSolutions(std::vector<std::unique_ptr<PermutationSolution>>&& solutions);

    public:
        virtual ~PermutationSolutionRegistry() = default;

        template <typename... Ts>
        Query querySolutions(Ts... ts)
        {
            return mSolutionQuery.query(ts...);
        }

        Query const& allSolutions() const;

        uint32_t solutionCount() const;

    private:
        std::vector<std::unique_ptr<PermutationSolution>> mSolutionStorage;
        Query                                             mSolutionQuery;
    };
    // @endcond

} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_SOLUTION_REGISTRY_HPP
