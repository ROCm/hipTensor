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
#include <functional>

#include <hiptensor/hiptensor_types.hpp>

namespace hiptensor
{
    template <typename T>
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

    //Updatable priority queue
    template <typename T, typename PriorityType, typename Compare = std::less<PriorityType>>
    class Updatable_Priority_Queue
    {
    public:
        //An element in the queue
        struct Element {
            T key;
            PriorityType priority;
            std::size_t heap_index;
        };

        Updatable_Priority_Queue() = default;

        Updatable_Priority_Queue(const Updatable_Priority_Queue& other)
           : heap_data(other.heap_data), mp_key_to_element(other.mp_key_to_element) {}

        Updatable_Priority_Queue& operator=(const Updatable_Priority_Queue& other)
        {
            if(this != &other)
            {
                heap_data = other.heap_data;
                mp_key_to_element = other.mp_key_to_element;
            }
            return (*this);
        }

        //Insert a new element
        void push(const T& key, const PriorityType& priority)
        {
            Element *item = new Element{key, priority, heap_data.size()};
            heap_data.push_back(item);
            mp_key_to_element[key] = heap_data.back();
            sift_up(heap_data.size() - 1);
        }

        //Update the priority of an existing element
        bool update_item(const T& key, const PriorityType& new_priority)
        {
            bool retVal = false;
            auto it = mp_key_to_element.find(key);
            if(it != mp_key_to_element.end())
            {
                Element *elem = it->second;
                PriorityType old_priority = elem->priority;
                elem->priority = new_priority;

                if(compare(new_priority, old_priority))
                {
                    sift_up(elem->heap_index);
                }
                else
                {
                    sift_down(elem->heap_index);
                }

                retVal = true;
            }

            return retVal;
        }

        const T& top_key() const
        {
            return heap_data[0]->key;
        }

        const std::pair<T, PriorityType> top() const
        {
            return {heap_data[0]->key, heap_data[0]->priority};
        }

        //Remove the top element
        void pop()
        {
            if(empty()) return;
            mp_key_to_element.erase(heap_data[0]->key);
            std::swap(heap_data[0], heap_data.back());
            heap_data[0]->heap_index = 0;
            delete heap_data.back();
            heap_data.pop_back();
            if(!empty())
            {
                sift_down(0);
            }
        }

        bool empty() const {
            return heap_data.empty();
        }

        std::size_t size() const {
            return heap_data.size();
        }

        ~Updatable_Priority_Queue()
        {
            for(auto elem:heap_data) delete elem;
            heap_data.clear();
        }

    private:
        std::vector<Element*> heap_data;
        std::unordered_map<T,Element*> mp_key_to_element;
        Compare compare;

        void sift_up(std::size_t index)
        {
            if(index < 1) return;

            std::size_t parent_index = (index - 1) / 2;
            if(compare(heap_data[index]->priority, heap_data[parent_index]->priority))
            {
                std::swap(heap_data[index], heap_data[parent_index]);
                heap_data[index]->heap_index = index;
                heap_data[parent_index]->heap_index = parent_index;
                sift_up(parent_index);
            }
        }

        void sift_down(std::size_t index)
        {
            std::size_t left_child = 2*index + 1;
            std::size_t right_child = 2*index + 2;
            std::size_t largest = index;

            if(left_child < heap_data.size() && compare(heap_data[left_child]->priority, heap_data[largest]->priority)) largest = left_child;
            if(right_child < heap_data.size() && compare(heap_data[right_child]->priority, heap_data[largest]->priority)) largest = right_child;

            if(largest != index) {
                std::swap(heap_data[largest], heap_data[index]);
                heap_data[largest]->heap_index = largest;
                heap_data[index]->heap_index = index;
                sift_down(largest);
            }
        }
    };

    class PlanCache
    {
    public:
        using HashId = std::size_t;
        using Uid = std::size_t;
        using time_point = std::chrono::system_clock::time_point;
        using Uid_Pair = std::pair<time_point,HashId>;

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

        //Function for serialization to disk
        hiptensorStatus_t writeFile(const char filename[]);

        //Function for deserialization from disk
        hiptensorStatus_t readFile(const char filename[]);

        uint32_t getCachelinesNum() {return mPlanCacheLines.size();}

       //Resize Plan Cache
       void Resize(uint32_t numEntries);

    private:
        //Max size of the hash table
        uint32_t max_cachelines;

        //Hash table to store solution uid for lookup by hash key
        std::unordered_map<HashId,Uid> mPlanCacheLines;

        //Heap to sort the solution uid by used time
        Updatable_Priority_Queue<HashId, time_point> pq_UidUsedTimes;

        //Mutex to implement the plan cache in a thread-safe manner
        mutable std::mutex mMutex;

        //Get solution id through hash key
        Uid getSolutionID(HashId hash_id);

        //Update cache line used time
        void updateCachelineTime(HashId hash_id);

    };

} // namespace hiptensor

#endif // HIPTENSOR_PLAN_CACHE_HPP
