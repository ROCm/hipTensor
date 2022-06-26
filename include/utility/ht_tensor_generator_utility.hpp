#pragma once
#include <fstream>
#if 0
template <typename T>
struct GeneratorTensor_cuTensor
{
    template <typename... Is>
    T operator()(Is...)
    {
        float tmp = ((float(std::rand()))/float(RAND_MAX))*2;
        return static_cast<T>(tmp);
    }
};


template <typename T, typename Range>
void LogRangeToFile(std::ofstream& fs, Range&& range, std::string delim)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            fs << delim;
        fs << static_cast<T>(v);
    }
    return;
}
#endif
