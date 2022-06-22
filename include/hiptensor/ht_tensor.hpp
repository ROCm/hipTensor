#ifndef HT_TENSOR_HPP
#define HT_TENSOR_HPP

#include "ht_types.hpp"

hiptensorStatus_t hiptensorInit(hiptensorHandle_t* handle);

hiptensorStatus_t hiptensorInitTensorDescriptor(const hiptensorHandle_t* handle,
                          hiptensorTensorDescriptor_t* desc, const uint32_t numModes,
                          const int64_t lens[], const int64_t strides[],
                          hiptensorDataType_t dataType, hiptensorOperator_t unaryOp);

hiptensorStatus_t hiptensorGetAlignmentRequirement(const hiptensorHandle_t* handle,
                          const void *ptr, const hiptensorTensorDescriptor_t* desc, 
                          uint32_t* alignmentRequirement);
#endif
