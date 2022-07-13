#pragma once

#include <hip/hip_runtime.h>

inline void hip_check_error(hipError_t x)
{
    if(x != hipSuccess)
    {
        std::ostringstream ss;
        ss << "HIP runtime error: " << hipGetErrorString(x) << ". " << __FILE__ << ": " << __LINE__
           << "in function: " << __func__;
        throw std::runtime_error(ss.str());
    }
}

template<typename T>
void hiptensorPrintTensor ( T *vec, size_t size)
{
     int index = 0;
     while (index != size)
     {
        if (index == size-1)
           std::cout << vec[index];
        else
           std::cout << vec[index] << "," ;

        index++;
     }
}
