/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef HIPTENSOR_CONFIG_HPP
#define HIPTENSOR_CONFIG_HPP

namespace hiptensor
{

///
/// Architecture support
/// Guaranteed symbols:
/// HIPTENSOR_ARCH_GFX908
/// HIPTENSOR_ARCH_GFX90a
/// HIPTENSOR_ARCH_GFX940
/// HIPTENSOR_ARCH_GFX941
/// HIPTENSOR_ARCH_GFX942
/// HIPTENSOR_ARCH_GFX1100
/// HIPTENSOR_ARCH_GFX1101
/// HIPTENSOR_ARCH_GFX1102
#if defined(__gfx908__)
#define HIPTENSOR_ARCH_GFX908 __gfx908__
#elif defined(__gfx90a__)
#define HIPTENSOR_ARCH_GFX90A __gfx90a__
#elif defined(__gfx940__)
#define HIPTENSOR_ARCH_GFX940 __gfx940__
#elif defined(__gfx941__)
#define HIPTENSOR_ARCH_GFX941 __gfx941__
#elif defined(__gfx942__)
#define HIPTENSOR_ARCH_GFX942 __gfx942__
#elif defined(__gfx1100__)
#define HIPTENSOR_ARCH_GFX1100 __gfx1100__
#elif defined(__gfx1101__)
#define HIPTENSOR_ARCH_GFX1101 __gfx1101__
#elif defined(__gfx1102__)
#define HIPTENSOR_ARCH_GFX1102 __gfx1102__
#else
#define HIPTENSOR_ARCH_HOST 1
#endif

#if !defined(HIPTENSOR_ARCH_GFX908)
#define HIPTENSOR_ARCH_GFX908 0
#endif
#if !defined(HIPTENSOR_ARCH_GFX90A)
#define HIPTENSOR_ARCH_GFX90A 0
#endif
#if !defined(HIPTENSOR_ARCH_GFX940)
#define HIPTENSOR_ARCH_GFX940 0
#endif
#if !defined(HIPTENSOR_ARCH_GFX941)
#define HIPTENSOR_ARCH_GFX941 0
#endif
#if !defined(HIPTENSOR_ARCH_GFX942)
#define HIPTENSOR_ARCH_GFX942 0
#endif
#if !defined(HIPTENSOR_ARCH_GFX1100)
#define HIPTENSOR_ARCH_GFX1100 0
#endif
#if !defined(HIPTENSOR_ARCH_GFX1101)
#define HIPTENSOR_ARCH_GFX1101 0
#endif
#if !defined(HIPTENSOR_ARCH_GFX1102)
#define HIPTENSOR_ARCH_GFX1102 0
#endif
#if !defined(HIPTENSOR_ARCH_HOST)
#define HIPTENSOR_ARCH_HOST 0
#endif

///
/// Architecture configuration
/// Guaranteed symbols:
/// HIPTENSOR_ARCH_GFX9
/// HIPTENSOR_ARCH_GFX11
/// HIPTENSOR_WAVE64_MODE
/// HIPTENSOR_WAVE32_MODE
/// HIPTENSOR_BLOCK_DIM_16_SUPPORTED
/// HIPTENSOR_BLOCK_DIM_32_SUPPORTED
///
#if HIPTENSOR_ARCH_GFX908 || HIPTENSOR_ARCH_GFX90A || HIPTENSOR_ARCH_GFX940 \
    || HIPTENSOR_ARCH_GFX941 || HIPTENSOR_ARCH_GFX942
#define HIPTENSOR_ARCH_GFX9 1
#define HIPTENSOR_WAVE64_MODE 1
#define HIPTENSOR_BLOCK_DIM_16_SUPPORTED 1
#define HIPTENSOR_BLOCK_DIM_32_SUPPORTED 1
#endif

#if HIPTENSOR_ARCH_GFX1100 || HIPTENSOR_ARCH_GFX1101 || HIPTENSOR_ARCH_GFX1102
#define HIPTENSOR_ARCH_GFX11 1
#define HIPTENSOR_WAVE32_MODE 1
#define HIPTENSOR_BLOCK_DIM_16_SUPPORTED 1
#endif

#if !defined(HIPTENSOR_ARCH_GFX9)
#define HIPTENSOR_ARCH_GFX9 0
#endif
#if !defined(HIPTENSOR_ARCH_GFX11)
#define HIPTENSOR_ARCH_GFX11 0
#endif
#if !defined(HIPTENSOR_WAVE64_MODE)
#define HIPTENSOR_WAVE64_MODE 0
#endif
#if !defined(HIPTENSOR_WAVE32_MODE)
#define HIPTENSOR_WAVE32_MODE 0
#endif
#if !defined(HIPTENSOR_BLOCK_DIM_16_SUPPORTED)
#define HIPTENSOR_BLOCK_DIM_16_SUPPORTED 0
#endif
#if !defined(HIPTENSOR_BLOCK_DIM_32_SUPPORTED)
#define HIPTENSOR_BLOCK_DIM_32_SUPPORTED 0
#endif

#if defined(NDEBUG)
#define HIPTENSOR_UNSUPPORTED_IMPL(MSG)
#else
#define HIPTENSOR_UNSUPPORTED_IMPL(MSG) __attribute__((deprecated(MSG)))
#endif

#if defined(HIP_NO_HALF)
#define HIPTENSOR_NO_HALF 1
#else
#define HIPTENSOR_NO_HALF 0
#endif // HIP_NO_HALF

#if HIPTENSOR_NO_HALF || (!HIPTENSOR_NO_HALF && defined(__HIP_NO_HALF_CONVERSIONS__))
#define HIPTENSOR_TESTS_NO_HALF 1
#else
#define HIPTENSOR_TESTS_NO_HALF 0
#endif // !HIPTENSOR_NO_HALF && defined(__HIP_NO_HALF_CONVERSIONS__)

///
/// Sanity checks
///
#if HIPTENSOR_ARCH_GFX11
    static_assert((bool)(HIPTENSOR_WAVE32_MODE) && !(bool)(HIPTENSOR_WAVE64_MODE),
                  "hipTensor supports only wave32 for gfx11 arch");
    static_assert((bool)(HIPTENSOR_BLOCK_DIM_16_SUPPORTED)
                      && !(bool)(HIPTENSOR_BLOCK_DIM_32_SUPPORTED),
                  "hipTensor supports only block size of 16 for gfx11 arch");
#endif

#if HIPTENSOR_ARCH_GFX9
    static_assert(!(bool)(HIPTENSOR_WAVE32_MODE) && (bool)(HIPTENSOR_WAVE64_MODE),
                  "hipTensor supports only wave64 for gfx9 arch");
    static_assert((bool)(HIPTENSOR_BLOCK_DIM_16_SUPPORTED)
                      && (bool)(HIPTENSOR_BLOCK_DIM_32_SUPPORTED),
                  "hipTensor requires block size of 16 and 32 for gfx9 arch");
#endif

///
/// Host and Device symbols
///
#define HIPTENSOR_DEVICE __device__

#define HIPTENSOR_HOST __host__

#define HIPTENSOR_HOST_DEVICE HIPTENSOR_HOST HIPTENSOR_DEVICE

#define HIPTENSOR_KERNEL __global__

} // namespace hiptensor

#endif // HIPTENSOR_CONFIG_HPP
