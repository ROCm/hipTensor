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

#ifndef HIPTENSOR_PLAN_CACHE_AUTOTUNE_HPP
#define HIPTENSOR_PLAN_CACHE_AUTOTUNE_HPP

#include <vector>
#include <unordered_map>

#include "plan_cache.hpp"
#include "singleton.hpp"

namespace hiptensor
{
    class PlancacheAutotuneMgr : public LazySingleton<PlancacheAutotuneMgr>
    {
    private:

        std::unordered_map<std::string,bool> mpAutotuneStarted; //If autotune process has been started
        std::unordered_map<std::string,uint32_t> mpCallCount;  //Number of calling a function
        std::unordered_map<std::string,std::pair<float,void*>> bestSolution; //The best solution pair (run time <--> solution)

        //Singleton: only one instance
        PlancacheAutotuneMgr() {}
        PlancacheAutotuneMgr(PlancacheAutotuneMgr const&) = delete;
        PlancacheAutotuneMgr(PlancacheAutotuneMgr&&) = delete;
        PlancacheAutotuneMgr& operator=(PlancacheAutotuneMgr const&) = delete;
        PlancacheAutotuneMgr& operator=(PlancacheAutotuneMgr&&) = delete;

    public:

        friend std::unique_ptr<PlancacheAutotuneMgr> std::make_unique<PlancacheAutotuneMgr>();

        //Start the autotune by setting initial status
        void startAutotune(const std::string strFuncName)
        {
            if(mpAutotuneStarted.find(strFuncName) == mpAutotuneStarted.end() || mpAutotuneStarted[strFuncName] == false)
            {
                mpAutotuneStarted[strFuncName] = true;
                mpCallCount[strFuncName] = 0;
                bestSolution[strFuncName] = std::make_pair(1e100, nullptr);
            }
        }

        //Set current solution to try in autotune
        template <typename T>
        void setAutotune(const std::string strFuncName, const hiptensorHandle_t handle, const hiptensorPlan_t plan)
        {
            if(handle -> planCache == nullptr) return;
            if(plan->mPref == nullptr) return;
            if(plan->mPref->mCacheMode != HIPTENSOR_CACHE_MODE_PEDANTIC) return;

            // Look for solution from memory cache (Plan Cache) or do autotuning for Plan Cache
            auto Uid = handle->planCache->querySolutionUid(plan->mOpDesc);
            if(Uid > 0ull)
            {
                //If there is a solution already in the Plan Cache, directly set solution from Plan Cache
                auto candidates = std::vector<T*>(plan->mPref->mCandidates.size());
                std::transform(plan->mPref->mCandidates.begin(), plan->mPref->mCandidates.end(), candidates.begin(),
                                                                    [](auto* p) { return (T*)p; });

                plan->mPref->mSolution = findSolutionByUid(candidates,Uid);
            }
            else if(plan->mPref->mAutotuneMode == HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL)
            {
                //If there is no solution in the Plan Cache, we may need to do autotuning based on the number of calling this function
                uint32_t callCount = mpCallCount[strFuncName];
                if (callCount < plan->mPref->mIncrementalCount)
                {
                    //If the number of calling this function is less than the number of different candidates to explore,
                    //we are doing the autotuning to find the best solution
                    if(callCount < plan->mPref->mCandidates.size()) plan->mPref->mSolution = plan->mPref->mCandidates[callCount];
                    // TODO: need a ranked solution array to explore from fastest to slowest
                }
            }
        }

        //Check and save autotune results to Plan Cache
        template <typename T>
        void saveAutotune(const std::string strFuncName, float time, const hiptensorHandle_t handle, const hiptensorPlan_t plan)
        {
            if(handle -> planCache == nullptr) return;
            if(plan->mPref == nullptr) return;
            if(plan->mPref->mCacheMode != HIPTENSOR_CACHE_MODE_PEDANTIC) return;

            if(plan->mPref->mAutotuneMode == HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL)
            {
                //For Plan Cache autotuning, remember the best performace solution
                float minTime = bestSolution[strFuncName].first;
                if (time < minTime) {
                    bestSolution[strFuncName].first = time;
                    bestSolution[strFuncName].second = plan->mPref->mSolution;
                }

                //Count the number of calling this function for Plan Cache autotuning
                mpCallCount[strFuncName]++;
            }

            //When the number of calling this function is equal to the number of different candidates to explore
            //or there is no autotuning, add the best solution to the Plan Cache.
            if(mpCallCount[strFuncName] >= plan->mPref->mIncrementalCount || plan->mPref->mAutotuneMode == HIPTENSOR_AUTOTUNE_MODE_NONE)
            {
                hiptensor::PlanCache::HashId hashID = handle->planCache->getHashID(plan->mOpDesc);
                T* solution = (T*)bestSolution[strFuncName].second;
                if (solution == nullptr) solution = (T*)(plan->mPref->mSolution);
                handle->planCache->addCacheLine(hashID, solution->uid());

                //reset callCount for next round of autotuning
                mpAutotuneStarted[strFuncName] = false;
                mpCallCount[strFuncName] = 0;
                bestSolution[strFuncName] = std::make_pair(1e100, nullptr);
            }
        }
    };

} // namespace hiptensor

#endif // HIPTENSOR_PLAN_CACHE_AUTOTUNE_HPP
