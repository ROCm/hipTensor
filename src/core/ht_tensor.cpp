#include <cassert>

#include "ht_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "ht_tensor_generator_utility.hpp"

hiptensorStatus_t hiptensorInit(hiptensorHandle_t* handle)
{
    if (!handle)
	    return HIPTENSOR_STATUS_NOT_INITIALIZED;

    else if (hipInit(0) == hipErrorInvalidDevice)
        return HIPTENSOR_STATUS_ROCM_ERROR;

    else if(hipInit(0) == hipErrorInvalidValue)
        return HIPTENSOR_STATUS_INVALID_VALUE;

    return HIPTENSOR_STATUS_SUCCESS;
}


hiptensorStatus_t hiptensorInitTensorDescriptor(const hiptensorHandle_t* handle,
			  hiptensorTensorDescriptor_t* desc, const uint32_t numModes,
              		  const int64_t lens[], const int64_t strides[], 
              		  hiptensorDataType_t dataType, hiptensorOperator_t unaryOp)
{   
    if (!handle || !desc)
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    
    if (((!lens) && (!strides)) || dataType != HIPTENSOR_R_32F || unaryOp != HIPTENSOR_OP_IDENTITY)
        return  HIPTENSOR_STATUS_INVALID_VALUE;

    using descType = float;
    int ht_index = 0;

    std::vector<std::int64_t> ht_lens;
    std::vector<std::int64_t> ht_strides;

    for (int index=0; index < numModes; index++)
    {
	    ht_lens.push_back(lens[index]);
	    if (strides)
	        ht_strides.push_back(strides[index]);
    }
	
    if (!strides) 
   	    *desc = hiptensorTensorDescriptor_t(std::vector<std::size_t>(ht_lens.begin(), ht_lens.end()));
    else
      	    *desc = hiptensorTensorDescriptor_t(std::vector<std::size_t>(ht_lens.begin(), ht_lens.end()),
		       					std::vector<std::size_t>(ht_strides.begin(), ht_strides.end()));
    
    desc->ht_type = dataType;

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorGetAlignmentRequirement(const hiptensorHandle_t* handle,
                          const void *ptr, const hiptensorTensorDescriptor_t* desc, uint32_t* alignmentRequirement)
{
    if (!handle || !desc)
        return HIPTENSOR_STATUS_NOT_INITIALIZED;

    if (desc->ht_type != HIPTENSOR_R_32F)
		return  HIPTENSOR_STATUS_INVALID_VALUE;
        
    using descType = float;
    
    *alignmentRequirement = sizeof (descType) * desc->hiptensorGetElementSpace();
    
    return HIPTENSOR_STATUS_SUCCESS;
}

void hiptensorContractionDescriptor_t:: hiptensorContractionAttrUpdate(const hiptensorTensorDescriptor_t* desc[],
                                                            const uint32_t tensor_size[], const int tensor_desc_num)
{
    for(int index = 0; index < tensor_desc_num; index++)
    {
        ht_contract_attr_desc.push_back({desc[index]->hiptensorGetLengths(), desc[index]->hiptensorGetStrides(), tensor_size[index]});
    }	
    return;
}

void hiptensorTensorDescriptor_t::hiptensorCalculateStrides()
{
    mStrides.clear();
    mStrides.resize(mLens.size(), 0);
    if(mStrides.empty())
        return;

    mStrides.back() = 1;
    std::partial_sum(
        mLens.rbegin(), mLens.rend() - 1, mStrides.rbegin() + 1, std::multiplies<std::size_t>());
}

std::size_t hiptensorTensorDescriptor_t::hiptensorGetNumOfDimension() const { return mLens.size(); }

std::size_t hiptensorTensorDescriptor_t::hiptensorGetElementSize() const
{
    assert(mLens.size() == mStrides.size());
    return std::accumulate(
        mLens.begin(), mLens.end(), std::size_t{1}, std::multiplies<std::size_t>());
}

std::size_t hiptensorTensorDescriptor_t::hiptensorGetElementSpace() const
{
    std::size_t space = 1;
    for(std::size_t i = 0; i < mLens.size(); ++i)
    {
        space += (mLens[i] - 1) * mStrides[i];
    }
    return space;
}

const std::vector<std::size_t>&  hiptensorTensorDescriptor_t::hiptensorGetLengths() const { return mLens; }

const std::vector<std::size_t>&  hiptensorTensorDescriptor_t::hiptensorGetStrides() const { return mStrides; }

std::ostream& operator<<(std::ostream& os, const hiptensorTensorDescriptor_t& desc)
{
    os << "dim " << desc.hiptensorGetNumOfDimension() << ", ";

    os << "lengths {";
    LogRange(os, desc.hiptensorGetLengths(), ", ");
    os << "}, ";

    os << "strides {";
    LogRange(os, desc.hiptensorGetStrides(), ", ");
    os << "}";

    return os;
}
