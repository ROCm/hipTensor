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

#include "plan_cache.hpp"
#include "data_types.hpp"
#include "util.hpp"
#include "hash.hpp"

namespace hiptensor
{
    /////// Class PlanCache ////////////////////////

    PlanCache::PlanCache()
        : max_cachelines(128)
    {

    }

    PlanCache::HashId PlanCache::getHashID(hiptensorOperationDescriptor_t desc)
    {
        auto tag         = desc->mTag;
        auto ADataType   = desc->mDescA ? desc->mDescA->mType : hiptensor::NONE_TYPE;
        auto BDataType   = desc->mDescB ? desc->mDescB->mType : hiptensor::NONE_TYPE;
        auto CDataType   = desc->mDescC ? desc->mDescC->mType : hiptensor::NONE_TYPE;
        auto DDataType   = desc->mDescD ? desc->mDescD->mType : hiptensor::NONE_TYPE;
        auto computeType = desc->mDescCompute;
        auto mOperationType = desc->mOperationType;
        auto mContractionOpId = desc->mContractionOpId;

        //Get lengths
        std::vector<std::size_t> lengthsA = hiptensor::getTensorLengths(desc->mDescA);
        std::vector<std::size_t> lengthsB = hiptensor::getTensorLengths(desc->mDescB);
        std::vector<std::size_t> lengthsC = hiptensor::getTensorLengths(desc->mDescC);
        std::vector<std::size_t> lengthsD = hiptensor::getTensorLengths(desc->mDescD);

        std::vector<std::size_t> arr_lsm;
        arr_lsm.insert(arr_lsm.end(),lengthsA.begin(),lengthsA.end());
        arr_lsm.insert(arr_lsm.end(),lengthsB.begin(),lengthsB.end());
        arr_lsm.insert(arr_lsm.end(),lengthsC.begin(),lengthsC.end());
        arr_lsm.insert(arr_lsm.end(),lengthsD.begin(),lengthsD.end());

        //Get strides
        std::vector<std::size_t> stridesA = hiptensor::getTensorStrides(desc->mDescA);
        std::vector<std::size_t> stridesB = hiptensor::getTensorStrides(desc->mDescB);
        std::vector<std::size_t> stridesC = hiptensor::getTensorStrides(desc->mDescC);
        std::vector<std::size_t> stridesD = hiptensor::getTensorStrides(desc->mDescD);

        arr_lsm.insert(arr_lsm.end(),stridesA.begin(),stridesA.end());
        arr_lsm.insert(arr_lsm.end(),stridesB.begin(),stridesB.end());
        arr_lsm.insert(arr_lsm.end(),stridesC.begin(),stridesC.end());
        arr_lsm.insert(arr_lsm.end(),stridesD.begin(),stridesD.end());

        //Get modes
        arr_lsm.insert(arr_lsm.end(),desc->mModeA.begin(),desc->mModeA.end());
        arr_lsm.insert(arr_lsm.end(),desc->mModeB.begin(),desc->mModeB.end());
        arr_lsm.insert(arr_lsm.end(),desc->mModeC.begin(),desc->mModeC.end());
        arr_lsm.insert(arr_lsm.end(),desc->mModeD.begin(),desc->mModeD.end());

        //generate hash ID by {Tag, DatatypeA, DatatypeB, DatatypeC, DatatypeD, DatatypeCompute, mOperationType, mContractionOpId, lengths, strides, modes}
        PlanCache::HashId hashID = Hash{}(tag, ADataType, BDataType, CDataType, DDataType, computeType, mOperationType, mContractionOpId, arr_lsm);

        return hashID;
    }

    PlanCache::Uid PlanCache::querySolutionUid(hiptensorOperationDescriptor_t desc)
    {
        PlanCache::HashId hashID = getHashID(desc);

        PlanCache::Uid solution_uid = getSolutionID(hashID);
        return solution_uid;
    }

    void PlanCache::addCacheLine(HashId hash_id, Uid sol_id)
    {
        std::scoped_lock lock(mMutex);

        assert(max_cachelines > 0);

        //If the table size equal to the maximum size, then remove extra LRU(least-recently-used) record
        if(mPlanCacheLines.size() == max_cachelines && mPlanCacheLines.find(hash_id) != mPlanCacheLines.end()) {
            Uid_Pair lru_item = pq_UidUsedTimes.top();
            pq_UidUsedTimes.pop();

            mPlanCacheLines.erase(lru_item.second);
        }

        //Add the cache line
        mPlanCacheLines[hash_id] = sol_id;

        //Also add the cache line generation time to a min heap
        Uid_Pair item = std::make_pair(std::chrono::system_clock::now(),hash_id);
        //If the cache line already in the heap, remove that element first
        std::priority_queue<PlanCache::Uid_Pair, std::vector<Uid_Pair>,PlanCache::CompareUidPairs> tmp_heap;
        while(!pq_UidUsedTimes.empty())
        {
            if(pq_UidUsedTimes.top().second != hash_id) tmp_heap.push(pq_UidUsedTimes.top());
            pq_UidUsedTimes.pop();
        }
        pq_UidUsedTimes = std::move(tmp_heap);
        //After make sure same item is removed, add the item
        pq_UidUsedTimes.push(item);

        assert(pq_UidUsedTimes.size() == mPlanCacheLines.size());
    }

    void PlanCache::Resize(uint32_t numEntries)
    {
        if(numEntries < 1u) return;

        max_cachelines = numEntries;

        //If the table size exceeds the maximum size, then remove extra LRU(least-recently-used) records
        while(pq_UidUsedTimes.size() > max_cachelines) {
            Uid_Pair lru_item = pq_UidUsedTimes.top();
            pq_UidUsedTimes.pop();

            mPlanCacheLines.erase(lru_item.second);
        }
    }

    PlanCache::Uid PlanCache::getSolutionID(HashId hash_id)
    {
        if(mPlanCacheLines.find(hash_id)!=mPlanCacheLines.end()) return mPlanCacheLines[hash_id];
        return 0ull;
    }

    void PlanCache::serialization(std::ofstream& fstream)
    {
        fstream.write(reinterpret_cast<char*>(&max_cachelines), sizeof(max_cachelines));
        std::size_t size = mPlanCacheLines.size();
        fstream.write(reinterpret_cast<char*>(&size), sizeof(size));
        PlanCache::HashId hashId;
        PlanCache::Uid uId;
        for(auto item:mPlanCacheLines) {
            hashId = item.first;
            uId = item.second;
            fstream.write(reinterpret_cast<char*>(&hashId), sizeof(hashId));
            fstream.write(reinterpret_cast<char*>(&uId), sizeof(uId));
        }

        std::priority_queue<PlanCache::Uid_Pair, std::vector<Uid_Pair>,PlanCache::CompareUidPairs> tmp_heap = pq_UidUsedTimes;
        size = tmp_heap.size();
        fstream.write(reinterpret_cast<char*>(&size), sizeof(size));
        while(!tmp_heap.empty()) {
            PlanCache::Uid_Pair mPair = tmp_heap.top();
            tmp_heap.pop();
            fstream.write(reinterpret_cast<char*>(&mPair.first), sizeof(mPair.first));
            fstream.write(reinterpret_cast<char*>(&mPair.second), sizeof(mPair.second));
        }
    }

    void PlanCache::deserialization(std::ifstream& fstream)
    {
        fstream.read(reinterpret_cast<char*>(&max_cachelines), sizeof(max_cachelines));

        std::size_t size=0;

        mPlanCacheLines.clear();
        fstream.read(reinterpret_cast<char*>(&size), sizeof(size));
        PlanCache::HashId hashId;
        PlanCache::Uid uId;
        for(int num=0; num<size; num++)
        {
            fstream.read(reinterpret_cast<char*>(&hashId), sizeof(hashId));
            fstream.read(reinterpret_cast<char*>(&uId), sizeof(uId));
            mPlanCacheLines.insert({hashId, uId});
        }

        fstream.read(reinterpret_cast<char*>(&size), sizeof(size));
        PlanCache::Uid_Pair mPair;
        while(!pq_UidUsedTimes.empty()) pq_UidUsedTimes.pop();
        for(int num=0; num<size; num++)
        {
            fstream.read(reinterpret_cast<char*>(&mPair.first), sizeof(mPair.first));
            fstream.read(reinterpret_cast<char*>(&mPair.second), sizeof(mPair.second));
            pq_UidUsedTimes.push(mPair);
        }
    }

} // namespace hiptensor




