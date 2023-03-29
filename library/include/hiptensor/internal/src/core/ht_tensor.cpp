#include "ht_tensor.hpp"
#include "ht_tensor_generator_utility.hpp"

hipTensorStatus_t hipTensorInit(hipTensorHandle_t* handle)
{
    if (!handle)
	    return hipTensor_STATUS_NOT_INITIALIZED;

    else if (hipInit(0) == hipErrorInvalidDevice)
        return hipTensor_STATUS_ROCM_ERROR;

    else if(hipInit(0) == hipErrorInvalidValue)
        return hipTensor_STATUS_INVALID_VALUE;

    return hipTensor_STATUS_SUCCESS;
}


hipTensorStatus_t hipTensorInitTensorDescriptor(const hipTensorHandle_t* handle,
			  hipTensorTensorDescriptor_t* desc, const uint32_t numModes,
              		  const int64_t lens[], const int64_t strides[], 
              		  hipTensorDataType_t dataType, hipTensorOperator_t unaryOp)
{   
    if (!handle || !desc)
        return hipTensor_STATUS_NOT_INITIALIZED;
    
    if (((!lens) && (!strides)) || dataType != hipTensor_R_32F || unaryOp != hipTensor_OP_IDENTITY)
        return  hipTensor_STATUS_INVALID_VALUE;

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
   	    *desc = hipTensorTensorDescriptor_t(std::vector<std::size_t>(ht_lens.begin(), ht_lens.end()));
    else
      	*desc = hipTensorTensorDescriptor_t(std::vector<std::size_t>(ht_lens.begin(), ht_lens.end()),
		       					std::vector<std::size_t>(ht_strides.begin(), ht_strides.end()));
    desc->ht_type = dataType;

    return hipTensor_STATUS_SUCCESS;
}

hipTensorStatus_t hipTensorGetAlignmentRequirement(const hipTensorHandle_t* handle,
                          const void *ptr, const hipTensorTensorDescriptor_t* desc, uint32_t* alignmentRequirement)
{
    if (!handle || !desc)
        return hipTensor_STATUS_NOT_INITIALIZED;

    if (desc->ht_type != hipTensor_R_32F)
		return  hipTensor_STATUS_INVALID_VALUE;
        
    using descType = float;
    
    *alignmentRequirement = sizeof (descType) * desc->hipTensorGetElementSpace();
    
    return hipTensor_STATUS_SUCCESS;
}

void hipTensorContractionDescriptor_t:: hipTensorContractionAttrUpdate(const hipTensorTensorDescriptor_t* desc[],
                                                            const uint32_t tensor_size[], const int tensor_desc_num)
{
    for(int index = 0; index < tensor_desc_num; index++)
    {
        ht_contract_attr_desc.push_back({desc[index]->hipTensorGetLengths(), desc[index]->hipTensorGetStrides(), tensor_size[index]});
    }	
    return;
}

void hipTensorTensorDescriptor_t::hipTensorCalculateStrides()
{
    mStrides.clear();
    mStrides.resize(mLens.size(), 0);
    if(mStrides.empty())
        return;

    mStrides.back() = 1;
    std::partial_sum(
        mLens.rbegin(), mLens.rend() - 1, mStrides.rbegin() + 1, std::multiplies<std::size_t>());
}

std::size_t hipTensorTensorDescriptor_t::hipTensorGetNumOfDimension() const { return mLens.size(); }

std::size_t hipTensorTensorDescriptor_t::hipTensorGetElementSize() const
{
    assert(mLens.size() == mStrides.size());
    return std::accumulate(
        mLens.begin(), mLens.end(), std::size_t{1}, std::multiplies<std::size_t>());
}

std::size_t hipTensorTensorDescriptor_t::hipTensorGetElementSpace() const
{
    std::size_t space = 1;
    for(std::size_t i = 0; i < mLens.size(); ++i)
    {
        space += (mLens[i] - 1) * mStrides[i];
    }
    return space;
}

const std::vector<std::size_t>&  hipTensorTensorDescriptor_t::hipTensorGetLengths() const { return mLens; }

const std::vector<std::size_t>&  hipTensorTensorDescriptor_t::hipTensorGetStrides() const { return mStrides; }

std::ostream& operator<<(std::ostream& os, const hipTensorTensorDescriptor_t& desc)
{
    os << "dim " << desc.hipTensorGetNumOfDimension() << ", ";

    os << "lengths {";
    hipTensorPrintVectorElements(desc.hipTensorGetLengths(), ", ");
    os << "}, ";

    os << "strides {";
    hipTensorPrintVectorElements(desc.hipTensorGetStrides(), ", ");
    os << "}";

    return os;
}
