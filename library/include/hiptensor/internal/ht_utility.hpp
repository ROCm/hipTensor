#pragma once

#include <hip/hip_runtime.h>
#include <fstream>

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
void hipTensorPrintArrayElements ( T *vec, size_t size)
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

template <typename S>
void hipTensorPrintVectorElements(const std::vector<S>& vec,
                    	std::string sep = " ")
{
    for (auto elem : vec) 
    {
        std::cout << elem << sep;
    }
 
}

template <typename F>
void hipTensorPrintElementsToFile(std::ofstream& fs, F *output, size_t size, char delim)
{
	if(!fs.is_open())
    {
        std::cout << "File not found!\n";
        return;
    }

    for(int i = 0;i < size; i++)
    {
        if (i == size-1)
          fs << static_cast<F>(output[i]);
        else
           fs << static_cast<F>(output[i]) << delim;
    }
    return;
}

