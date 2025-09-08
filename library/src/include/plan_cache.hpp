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

#ifndef HIPTENSOR_PLAN_CACHE_HPP
#define HIPTENSOR_PLAN_CACHE_HPP

#include <vector>
#include <unordered_map>
#include <chrono>
#include <queue>
#include <mutex>
#include <fstream>
#include <iostream>

#include <hiptensor/hiptensor_types.hpp>

namespace hiptensor
{
    template<typename T>
    auto findSolutionByUid(const std::vector<T*>& candidates, std::size_t Uid)
    {
        T* solPtr=nullptr;
        for(auto sol:candidates) {
            if(sol->uid()==Uid) {
                solPtr = sol;
                break;
            }
        }
        return solPtr;
    }

    class PlanCache
    {
    public:
        using HashId = std::size_t;
        using Uid = std::size_t;
        using time_t = std::chrono::system_clock::time_point;
        using Uid_Pair = std::pair<time_t,HashId>;

        struct CompareUidPairs {
            bool operator()(const Uid_Pair &p1, const Uid_Pair &p2) const {
                if(p1.first != p2.first) return p1.first > p2.first;
                return p1.second < p2.second;
            }
        };

        PlanCache();
        ~PlanCache() = default;

        //Get hash ID from peration descriptor pointer
        //Calculate hash key by: hash({Tag, DatatypeA, DatatypeB, DatatypeC, DatatypeD, DatatypeCompute, mOperationType, mContractionOpId, lengths, strides, modes})
        HashId getHashID(hiptensorOperationDescriptor_t desc);

        //Query solution Uid through operation descriptor pointer
        Uid querySolutionUid(hiptensorOperationDescriptor_t desc);

        //Add a cache record to the solution table
        //If the table size exceeds the maximum size, then remove one LRU(least-recently-used) record
        void addCacheLine(HashId hash_id, Uid sol_id);

        //Get solution id through hash key
        Uid getSolutionID(HashId hash_id);

        //Function for serialization to disk
        void serialization(std::ofstream& fstream);

        //Function for deserialization from disk
        void deserialization(std::ifstream& fstream);

        uint32_t getCachelinesNum() {return mPlanCacheLines.size();}

       //Resize Plan Cache
       void Resize(uint32_t numEntries);

    private:
        //Max size of the hash table
        uint32_t max_cachelines;

        //Hash table to store solution uid for lookup by hash key
        std::unordered_map<HashId,Uid> mPlanCacheLines;

        //Heap to sort the solution uid by used time
        std::priority_queue<Uid_Pair, std::vector<Uid_Pair>,CompareUidPairs> pq_UidUsedTimes;

        //Mutex to implement the plan cache in a thread-safe manner
        mutable std::mutex mMutex;

    };

} // namespace hiptensor

#endif // HIPTENSOR_PLAN_CACHE_HPP
