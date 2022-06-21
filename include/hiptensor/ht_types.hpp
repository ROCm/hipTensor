#pragma once
#include "host_tensor.hpp"
#include "device.hpp"

typedef enum
{
    /** The operation completed successfully.*/
    HIPTENSOR_STATUS_SUCCESS                = 0,
    /** The opaque data structure was not initialized.*/
    HIPTENSOR_STATUS_NOT_INITIALIZED        = 1,
    /** Resource allocation failed inside the cuTENSOR library.*/
    HIPTENSOR_STATUS_ALLOC_FAILED           = 3,
    /** An unsupported value or parameter was passed to the function (indicates an user error).*/
    HIPTENSOR_STATUS_INVALID_VALUE          = 7,
    /** Indicates that the device is either not ready, or the target architecture is not supported.*/
    HIPTENSOR_STATUS_ARCH_MISMATCH          = 8,
    /** An access to GPU memory space failed, which is usually caused by a failure to bind a texture.*/
    HIPTENSOR_STATUS_MAPPING_ERROR          = 11,
    /** The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.*/
    HIPTENSOR_STATUS_EXECUTION_FAILED       = 13,
    /** An internal cuTENSOR error has occurred.*/
    HIPTENSOR_STATUS_INTERNAL_ERROR         = 14,
    /** The requested operation is not supported.*/
    HIPTENSOR_STATUS_NOT_SUPPORTED          = 15,
    /** The functionality requested requires some license and an error was detected when trying to check the current licensing.*/
    HIPTENSOR_STATUS_LICENSE_ERROR          = 16,
    /** A call to CUBLAS did not succeed.*/
    HIPTENSOR_STATUS_CK_ERROR           = 17,
    /** Some unknown HIPTENSOR error has occurred.*/
    HIPTENSOR_STATUS_ROCM_ERROR             = 18,
    /** The provided workspace was insufficient.*/
    HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19,
    /** Indicates that the driver version is insufficient.*/
    HIPTENSOR_STATUS_INSUFFICIENT_DRIVER    = 20,
    /** Indicates an error related to file I/O.*/
    HIPTENSOR_STATUS_IO_ERROR               = 21,
} hiptensorStatus_t;

typedef enum 
{
	HIPTENSOR_R_16F  =  2, /* real as a half */
	HIPTENSOR_C_16F  =  6, /* complex as a pair of half numbers */
	HIPTENSOR_R_16BF = 14, /* real as a nv_bfloat16 */
	HIPTENSOR_C_16BF = 15, /* complex as a pair of nv_bfloat16 numbers */
	HIPTENSOR_R_32F  =  0, /* real as a float */
	HIPTENSOR_C_32F  =  4, /* complex as a pair of float numbers */
	HIPTENSOR_R_64F  =  1, /* real as a double */
	HIPTENSOR_C_64F  =  5, /* complex as a pair of double numbers */
	HIPTENSOR_R_4I   = 16, /* real as a signed 4-bit int */
	HIPTENSOR_C_4I   = 17, /* complex as a pair of signed 4-bit int numbers */
	HIPTENSOR_R_4U   = 18, /* real as a unsigned 4-bit int */
	HIPTENSOR_C_4U   = 19, /* complex as a pair of unsigned 4-bit int numbers */
	HIPTENSOR_R_8I   =  3, /* real as a signed 8-bit int */
	HIPTENSOR_C_8I   =  7, /* complex as a pair of signed 8-bit int numbers */
	HIPTENSOR_R_8U   =  8, /* real as a unsigned 8-bit int */
	HIPTENSOR_C_8U   =  9, /* complex as a pair of unsigned 8-bit int numbers */
	HIPTENSOR_R_16I  = 20, /* real as a signed 16-bit int */
	HIPTENSOR_C_16I  = 21, /* complex as a pair of signed 16-bit int numbers */
	HIPTENSOR_R_16U  = 22, /* real as a unsigned 16-bit int */
	HIPTENSOR_C_16U  = 23, /* complex as a pair of unsigned 16-bit int numbers */
	HIPTENSOR_R_32I  = 10, /* real as a signed 32-bit int */
	HIPTENSOR_C_32I  = 11, /* complex as a pair of signed 32-bit int numbers */
	HIPTENSOR_R_32U  = 12, /* real as a unsigned 32-bit int */
	HIPTENSOR_C_32U  = 13, /* complex as a pair of unsigned 32-bit int numbers */
	HIPTENSOR_R_64I  = 24, /* real as a signed 64-bit int */
	HIPTENSOR_C_64I  = 25, /* complex as a pair of signed 64-bit int numbers */
	HIPTENSOR_R_64U  = 26, /* real as a unsigned 64-bit int */
	HIPTENSOR_C_64U  = 27  /* complex as a pair of unsigned 64-bit int numbers */
} hiptensorDataType_t;

typedef enum
{
    HIPTENSOR_COMPUTE_16F  = (1U<< 0U),  ///< floating-point: 5-bit exponent and 10-bit mantissa (aka half)
    HIPTENSOR_COMPUTE_16BF = (1U<< 10U),  ///< floating-point: 8-bit exponent and 7-bit mantissa (aka bfloat)
    HIPTENSOR_COMPUTE_TF32 = (1U<< 12U),  ///< floating-point: 8-bit exponent and 10-bit mantissa (aka tensor-float-32)
    HIPTENSOR_COMPUTE_32F  = (1U<< 2U),  ///< floating-point: 8-bit exponent and 23-bit mantissa (aka float)
    HIPTENSOR_COMPUTE_64F  = (1U<< 4U),  ///< floating-point: 11-bit exponent and 52-bit mantissa (aka double)
    HIPTENSOR_COMPUTE_8U   = (1U<< 6U),  ///< 8-bit unsigned integer
    HIPTENSOR_COMPUTE_8I   = (1U<< 8U),  ///< 8-bit signed integer
    HIPTENSOR_COMPUTE_32U  = (1U<< 7U),  ///< 32-bit unsigned integer
    HIPTENSOR_COMPUTE_32I  = (1U<< 9U),  ///< 32-bit signed integer
} hiptensorComputeType_t;

typedef enum
{
    /* Unary */
    HIPTENSOR_OP_IDENTITY = 1,          ///< Identity operator (i.e., elements are not changed)
    HIPTENSOR_OP_SQRT = 2,              ///< Square root
    HIPTENSOR_OP_RELU = 8,              ///< Rectified linear unit
    HIPTENSOR_OP_CONJ = 9,              ///< Complex conjugate
    HIPTENSOR_OP_RCP = 10,              ///< Reciprocal
    HIPTENSOR_OP_SIGMOID = 11,          ///< y=1/(1+exp(-x))
    HIPTENSOR_OP_TANH = 12,             ///< y=tanh(x)
    HIPTENSOR_OP_EXP = 22,              ///< Exponentiation.
    HIPTENSOR_OP_LOG = 23,              ///< Log (base e).
    HIPTENSOR_OP_ABS = 24,              ///< Absolute value.
    HIPTENSOR_OP_NEG = 25,              ///< Negation.
    HIPTENSOR_OP_SIN = 26,              ///< Sine.
    HIPTENSOR_OP_COS = 27,              ///< Cosine.
    HIPTENSOR_OP_TAN = 28,              ///< Tangent.
    HIPTENSOR_OP_SINH = 29,             ///< Hyperbolic sine.
    HIPTENSOR_OP_COSH = 30,             ///< Hyperbolic cosine.
    HIPTENSOR_OP_ASIN = 31,             ///< Inverse sine.
    HIPTENSOR_OP_ACOS = 32,             ///< Inverse cosine.
    HIPTENSOR_OP_ATAN = 33,             ///< Inverse tangent.
    HIPTENSOR_OP_ASINH = 34,            ///< Inverse hyperbolic sine.
    HIPTENSOR_OP_ACOSH = 35,            ///< Inverse hyperbolic cosine.
    HIPTENSOR_OP_ATANH = 36,            ///< Inverse hyperbolic tangent.
    HIPTENSOR_OP_CEIL = 37,             ///< Ceiling.
    HIPTENSOR_OP_FLOOR = 38,            ///< Floor.
    /* Binary */
    HIPTENSOR_OP_ADD = 3,               ///< Addition of two elements
    HIPTENSOR_OP_MUL = 5,               ///< Multiplication of two elements
    HIPTENSOR_OP_MAX = 6,               ///< Maximum of two elements
    HIPTENSOR_OP_MIN = 7,               ///< Minimum of two elements

    HIPTENSOR_OP_UNKNOWN = 126, ///< reserved for internal use only

} hiptensorOperator_t;




typedef struct { /*TODO: Discuss the struct members */ }hiptensorHandle_t;


struct hiptensorTensorDescriptor_t{
    HostTensorDescriptor ht_desc;
    hiptensorDataType_t ht_type;
    hiptensorTensorDescriptor_t() = default;
    std::size_t hiptensorGetElementSpace() const;
    //void* GetHipTensorBuffer();
};

