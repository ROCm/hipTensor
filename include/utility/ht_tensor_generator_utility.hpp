#pragma once

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

template <>
struct GeneratorTensor_cuTensor<ck::bhalf_t>
{
    template <typename... Is>
    ck::bhalf_t operator()(Is...)
    {
        float tmp = ((float(std::rand()))/float(RAND_MAX))*2;
        return ck::type_convert<ck::bhalf_t>(tmp);
    }
};

