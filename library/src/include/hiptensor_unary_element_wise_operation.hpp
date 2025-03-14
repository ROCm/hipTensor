/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef HIPTENSOR_UNARY_ELEMENT_WISE_OPERATION
#define HIPTENSOR_UNARY_ELEMENT_WISE_OPERATION
#include <cassert>

#include <ck/utility/data_type.hpp>
#include <ck/utility/math.hpp>
#include <ck/utility/math_v2.hpp>
#include <ck/utility/type_convert.hpp>

#include <hiptensor/hiptensor_types.hpp>

namespace ck
{
    namespace math
    {
        template <typename T>
        inline __device__ T cos(T x)
        {
            return ck::type_convert<T>(::cosf(ck::type_convert<float>(x)));
        };
        template <>
        inline __device__ float cos<float>(float x)
        {
            return ::cosf(x);
        };
        template <>
        inline __device__ double cos<double>(double x)
        {
            return ::cos(x);
        };
        template <>
        inline __device__ half_t cos<half_t>(half_t x)
        {
            return hcos(static_cast<__half>(x));
        };
    }
    namespace tensor_operation
    {
        namespace element_wise
        {
            using FloatFunctionPtr  = void (*)(float& y, float const& x);
            using DoubleFunctionPtr = void (*)(double& y, double const& x);

            __host__ __device__ static void hiptensor_float_identity(float& y, float const& x)
            {
                y = x;
            };
            __host__ __device__ static void hiptensor_float_sqrt(float& y, float const& x)
            {
                y = ck::math::sqrt(x);
            };
            __host__ __device__ static void hiptensor_float_relu(float& y, float const& x)
            {
                y = x > 0 ? x : 0;
            };
            __host__ __device__ static void hiptensor_float_conj(float& y, float const& x)
            {
                y = x;
            };
            __host__ __device__ static void hiptensor_float_rcp(float& y, float const& x)
            {
                y = 1 / x;
            }
            __host__ __device__ static void hiptensor_float_sigmoid(float& y, float const& x)
            {
                y = 1 / (1 + ck::math::exp(-x));
            }
            __host__ __device__ static void hiptensor_float_tanh(float& y, float const& x)
            {
                y = ck::math::tanh(x);
            }
            __host__ __device__ static void hiptensor_float_exp(float& y, float const& x)
            {
                y = ck::math::exp(x);
            }
            __host__ __device__ static void hiptensor_float_log(float& y, float const& x)
            {
                y = ck::math::log(x);
            }
            __host__ __device__ static void hiptensor_float_abs(float& y, float const& x)
            {
                y = ck::math::abs(x);
            }
            __host__ __device__ static void hiptensor_float_neg(float& y, float const& x)
            {
                y = -x;
            }
            __host__ __device__ static void hiptensor_float_sin(float& y, float const& x)
            {
                y = ck::math::sin(x);
            }
            __host__ __device__ static void hiptensor_float_cos(float& y, float const& x)
            {
                y = ck::math::cos(x);
            }
            __host__ __device__ static void hiptensor_float_tan(float& y, float const& x)
            {
                y = ck::math::tan(x);
            }
            __host__ __device__ static void hiptensor_float_sinh(float& y, float const& x)
            {
                y = ck::math::sinh(x);
            }
            __host__ __device__ static void hiptensor_float_cosh(float& y, float const& x)
            {
                y = ck::math::cosh(x);
            }
            __host__ __device__ static void hiptensor_float_asin(float& y, float const& x)
            {
                y = ck::math::asin(x);
            }
            __host__ __device__ static void hiptensor_float_acos(float& y, float const& x)
            {
                y = ck::math::acos(x);
            }
            __host__ __device__ static void hiptensor_float_atan(float& y, float const& x)
            {
                y = ck::math::atan(x);
            }
            __host__ __device__ static void hiptensor_float_asinh(float& y, float const& x)
            {
                y = ck::math::asinh(x);
            }
            __host__ __device__ static void hiptensor_float_acosh(float& y, float const& x)
            {
                y = ck::math::acosh(x);
            }
            __host__ __device__ static void hiptensor_float_atanh(float& y, float const& x)
            {
                y = ck::math::atanh(x);
            }
            __host__ __device__ static void hiptensor_float_ceil(float& y, float const& x)
            {
                y = ck::math::ceil(x);
            }
            __host__ __device__ static void hiptensor_float_floor(float& y, float const& x)
            {
                y = ck::math::floor(x);
            }

            __host__ __device__ static void hiptensor_double_identity(double& y, double const& x)
            {
                y = x;
            };
            __host__ __device__ static void hiptensor_double_sqrt(double& y, double const& x)
            {
                y = ck::math::sqrt(x);
            };
            __host__ __device__ static void hiptensor_double_relu(double& y, double const& x)
            {
                y = x > 0 ? x : 0;
            };
            __host__ __device__ static void hiptensor_double_conj(double& y, double const& x)
            {
                y = x;
            };
            __host__ __device__ static void hiptensor_double_rcp(double& y, double const& x)
            {
                y = 1 / x;
            }
            __host__ __device__ static void hiptensor_double_sigmoid(double& y, double const& x)
            {
                y = 1 / (1 + ck::math::exp(-x));
            }
            __host__ __device__ static void hiptensor_double_tanh(double& y, double const& x)
            {
                y = ck::math::tanh(x);
            }
            __host__ __device__ static void hiptensor_double_exp(double& y, double const& x)
            {
                y = ck::math::exp(x);
            }
            __host__ __device__ static void hiptensor_double_log(double& y, double const& x)
            {
                y = ck::math::log(x);
            }
            __host__ __device__ static void hiptensor_double_abs(double& y, double const& x)
            {
                y = ck::math::abs(x);
            }
            __host__ __device__ static void hiptensor_double_neg(double& y, double const& x)
            {
                y = -x;
            }
            __host__ __device__ static void hiptensor_double_sin(double& y, double const& x)
            {
                y = ck::math::sin(x);
            }
            __host__ __device__ static void hiptensor_double_cos(double& y, double const& x)
            {
                y = ck::math::cos(x);
            }
            __host__ __device__ static void hiptensor_double_tan(double& y, double const& x)
            {
                y = ck::math::tan(x);
            }
            __host__ __device__ static void hiptensor_double_sinh(double& y, double const& x)
            {
                y = ck::math::sinh(x);
            }
            __host__ __device__ static void hiptensor_double_cosh(double& y, double const& x)
            {
                y = ck::math::cosh(x);
            }
            __host__ __device__ static void hiptensor_double_asin(double& y, double const& x)
            {
                y = ck::math::asin(x);
            }
            __host__ __device__ static void hiptensor_double_acos(double& y, double const& x)
            {
                y = ck::math::acos(x);
            }
            __host__ __device__ static void hiptensor_double_atan(double& y, double const& x)
            {
                y = ck::math::atan(x);
            }
            __host__ __device__ static void hiptensor_double_asinh(double& y, double const& x)
            {
                y = ck::math::asinh(x);
            }
            __host__ __device__ static void hiptensor_double_acosh(double& y, double const& x)
            {
                y = ck::math::acosh(x);
            }
            __host__ __device__ static void hiptensor_double_atanh(double& y, double const& x)
            {
                y = ck::math::atanh(x);
            }
            __host__ __device__ static void hiptensor_double_ceil(double& y, double const& x)
            {
                y = ck::math::ceil(x);
            }
            __host__ __device__ static void hiptensor_double_floor(double& y, double const& x)
            {
                y = ck::math::floor(x);
            }

            struct HiptensorUnaryOp
            {
                __host__ __device__ HiptensorUnaryOp(hiptensorOperator_t operator_type)
                    : op_type(operator_type)
                {
                }
                __host__ __device__ HiptensorUnaryOp(const HiptensorUnaryOp& dynamic_op) = default;
                __host__            __device__ ~HiptensorUnaryOp()                       = default;
                __host__ __device__ HiptensorUnaryOp& operator=(const HiptensorUnaryOp& other)
                    = default;

                __host__ __device__ void operator()(double& y, const double& x) const
                {
                    double_ops[op_type](y, x);
                }

                __host__ __device__ void operator()(float& y, const float& x) const
                {
                    float_ops[op_type](y, x);
                }

                __host__ __device__ void operator()(half_t& y, const half_t& x) const
                {
                    float tempX = static_cast<float>(x);
                    float tempY;
                    float_ops[op_type](tempY, tempX);
                    y = static_cast<float>(tempY);
                }

            public:
                hiptensorOperator_t               op_type     = HIPTENSOR_OP_IDENTITY;
                static constexpr FloatFunctionPtr float_ops[] = {
                    hiptensor_float_identity, // placeholder 0
                    hiptensor_float_identity, //HIPTENSOR_OP_IDENTITY = 1, ///< Identity operator (i.e., elements are not changed)
                    hiptensor_float_sqrt, //HIPTENSOR_OP_SQRT     = 2, ///< Square root
                    hiptensor_float_identity, // placeholder 3
                    hiptensor_float_identity, // placeholder 4
                    hiptensor_float_identity, // placeholder 5
                    hiptensor_float_identity, // placeholder 6
                    hiptensor_float_identity, // placeholder 7
                    hiptensor_float_relu, //HIPTENSOR_OP_RELU     = 8, ///< Rectified linear unit
                    hiptensor_float_conj, //HIPTENSOR_OP_CONJ     = 9, ///< Complex conjugate
                    hiptensor_float_rcp, //HIPTENSOR_OP_RCP      = 10, ///< Reciprocal
                    hiptensor_float_sigmoid, //HIPTENSOR_OP_SIGMOID  = 11, ///< y=1/(1+exp(-x))
                    hiptensor_float_tanh, //HIPTENSOR_OP_TANH     = 12, ///< y=tanh(x)
                    hiptensor_float_identity, // placeholder 13
                    hiptensor_float_identity, // placeholder 14
                    hiptensor_float_identity, // placeholder 15
                    hiptensor_float_identity, // placeholder 16
                    hiptensor_float_identity, // placeholder 17
                    hiptensor_float_identity, // placeholder 18
                    hiptensor_float_identity, // placeholder 19
                    hiptensor_float_identity, // placeholder 20
                    hiptensor_float_identity, // placeholder 21
                    hiptensor_float_exp, //HIPTENSOR_OP_EXP      = 22, ///< Exponentiation.
                    hiptensor_float_log, //HIPTENSOR_OP_LOG      = 23, ///< Log (base e).
                    hiptensor_float_abs, //HIPTENSOR_OP_ABS      = 24, ///< Absolute value.
                    hiptensor_float_neg, //HIPTENSOR_OP_NEG      = 25, ///< Negation.
                    hiptensor_float_sin, //HIPTENSOR_OP_SIN      = 26, ///< Sine.
                    hiptensor_float_cos, //HIPTENSOR_OP_COS      = 27, ///< Cosine.
                    hiptensor_float_tan, //HIPTENSOR_OP_TAN      = 28, ///< Tangent.
                    hiptensor_float_sinh, //HIPTENSOR_OP_SINH     = 29, ///< Hyperbolic sine.
                    hiptensor_float_cosh, //HIPTENSOR_OP_COSH     = 30, ///< Hyperbolic cosine.
                    hiptensor_float_asin, //HIPTENSOR_OP_ASIN     = 31, ///< Inverse sine.
                    hiptensor_float_acos, //HIPTENSOR_OP_ACOS     = 32, ///< Inverse cosine.
                    hiptensor_float_atan, //HIPTENSOR_OP_ATAN     = 33, ///< Inverse tangent.
                    hiptensor_float_asinh, //HIPTENSOR_OP_ASINH    = 34, ///< Inverse hyperbolic sine.
                    hiptensor_float_acosh, //HIPTENSOR_OP_ACOSH    = 35, ///< Inverse hyperbolic cosine.
                    hiptensor_float_atanh, //HIPTENSOR_OP_ATANH    = 36, ///< Inverse hyperbolic tangent.
                    hiptensor_float_ceil, //HIPTENSOR_OP_CEIL     = 37, ///< Ceiling.
                    hiptensor_float_floor, //HIPTENSOR_OP_FLOOR    = 38, ///< Floor.
                };

                static constexpr DoubleFunctionPtr double_ops[] = {
                    hiptensor_double_identity, // placeholder 0
                    hiptensor_double_identity, //HIPTENSOR_OP_IDENTITY = 1, ///< Identity operator (i.e., elements are not changed)
                    hiptensor_double_sqrt, //HIPTENSOR_OP_SQRT     = 2, ///< Square root
                    hiptensor_double_identity, // placeholder 3
                    hiptensor_double_identity, // placeholder 4
                    hiptensor_double_identity, // placeholder 5
                    hiptensor_double_identity, // placeholder 6
                    hiptensor_double_identity, // placeholder 7
                    hiptensor_double_relu, //HIPTENSOR_OP_RELU     = 8, ///< Rectified linear unit
                    hiptensor_double_conj, //HIPTENSOR_OP_CONJ     = 9, ///< Complex conjugate
                    hiptensor_double_rcp, //HIPTENSOR_OP_RCP      = 10, ///< Reciprocal
                    hiptensor_double_sigmoid, //HIPTENSOR_OP_SIGMOID  = 11, ///< y=1/(1+exp(-x))
                    hiptensor_double_tanh, //HIPTENSOR_OP_TANH     = 12, ///< y=tanh(x)
                    hiptensor_double_identity, // placeholder 13
                    hiptensor_double_identity, // placeholder 14
                    hiptensor_double_identity, // placeholder 15
                    hiptensor_double_identity, // placeholder 16
                    hiptensor_double_identity, // placeholder 17
                    hiptensor_double_identity, // placeholder 18
                    hiptensor_double_identity, // placeholder 19
                    hiptensor_double_identity, // placeholder 20
                    hiptensor_double_identity, // placeholder 21
                    hiptensor_double_exp, //HIPTENSOR_OP_EXP      = 22, ///< Exponentiation.
                    hiptensor_double_log, //HIPTENSOR_OP_LOG      = 23, ///< Log (base e).
                    hiptensor_double_abs, //HIPTENSOR_OP_ABS      = 24, ///< Absolute value.
                    hiptensor_double_neg, //HIPTENSOR_OP_NEG      = 25, ///< Negation.
                    hiptensor_double_sin, //HIPTENSOR_OP_SIN      = 26, ///< Sine.
                    hiptensor_double_cos, //HIPTENSOR_OP_COS      = 27, ///< Cosine.
                    hiptensor_double_tan, //HIPTENSOR_OP_TAN      = 28, ///< Tangent.
                    hiptensor_double_sinh, //HIPTENSOR_OP_SINH     = 29, ///< Hyperbolic sine.
                    hiptensor_double_cosh, //HIPTENSOR_OP_COSH     = 30, ///< Hyperbolic cosine.
                    hiptensor_double_asin, //HIPTENSOR_OP_ASIN     = 31, ///< Inverse sine.
                    hiptensor_double_acos, //HIPTENSOR_OP_ACOS     = 32, ///< Inverse cosine.
                    hiptensor_double_atan, //HIPTENSOR_OP_ATAN     = 33, ///< Inverse tangent.
                    hiptensor_double_asinh, //HIPTENSOR_OP_ASINH    = 34, ///< Inverse hyperbolic sine.
                    hiptensor_double_acosh, //HIPTENSOR_OP_ACOSH    = 35, ///< Inverse hyperbolic cosine.
                    hiptensor_double_atanh, //HIPTENSOR_OP_ATANH    = 36, ///< Inverse hyperbolic tangent.
                    hiptensor_double_ceil, //HIPTENSOR_OP_CEIL     = 37, ///< Ceiling.
                    hiptensor_double_floor, //HIPTENSOR_OP_FLOOR    = 38, ///< Floor.
                };
            };

            struct HiptensorBinaryOp
            {
                __host__ __device__ HiptensorBinaryOp(hiptensorOperator_t operator_type)
                    : op_type(operator_type)
                {
                }
                __host__ __device__ HiptensorBinaryOp(const HiptensorBinaryOp& dynamic_op)
                    = default;
                __host__            __device__ ~HiptensorBinaryOp() = default;
                __host__ __device__ HiptensorBinaryOp& operator=(const HiptensorBinaryOp& other)
                    = default;

                __host__ __device__ void
                    operator()(double& y, const double& x1, const double& x2) const
                {
                    switch(op_type)
                    {
                    case HIPTENSOR_OP_ADD:
                        y = x1 + x2;
                        break;
                    case HIPTENSOR_OP_MUL:
                        y = x1 * x2;
                        break;
                    case HIPTENSOR_OP_MAX:
                        y = x1 > x2 ? x1 : x2;
                        break;
                    case HIPTENSOR_OP_MIN:
                        y = x1 < x2 ? x1 : x2;
                        break;
                    default:
                        y = y;
                        break;
                    }
                }

                __host__ __device__ void
                    operator()(float& y, const float& x1, const float& x2) const
                {
                    switch(op_type)
                    {
                    case HIPTENSOR_OP_ADD:
                        y = x1 + x2;
                        break;
                    case HIPTENSOR_OP_MUL:
                        y = x1 * x2;
                        break;
                    case HIPTENSOR_OP_MAX:
                        y = x1 > x2 ? x1 : x2;
                        break;
                    case HIPTENSOR_OP_MIN:
                        y = x1 < x2 ? x1 : x2;
                        break;
                    default:
                        y = y;
                        break;
                    }
                }

                __host__ __device__ void
                    operator()(half_t& y, const half_t& x1, const half_t& x2) const
                {
                    float tempX1 = static_cast<float>(x1);
                    float tempX2 = static_cast<float>(x2);
                    float tempY;
                    this->operator()(tempY, tempX1, tempX2);
                    y = static_cast<float>(tempY);
                }

            public:
                hiptensorOperator_t op_type = HIPTENSOR_OP_IDENTITY;
            };
        } // namespace element_wise
    } // namespace tensor_operation
} // namespace ck
#endif // HIPTENSOR_UNARY_ELEMENT_WISE_OPERATION
