###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 # THE SOFTWARE.
 #
 ###############################################################################

 set(CK_PERMUTATION_INSTANCE_SOURCES
     ${CMAKE_CURRENT_SOURCE_DIR}/hiptensor_permutation_scale_instances.hpp
     ${CMAKE_CURRENT_SOURCE_DIR}/permutation_scale_2d_instances.hpp
     ${CMAKE_CURRENT_SOURCE_DIR}/permutation_scale_3d_instances.hpp
     ${CMAKE_CURRENT_SOURCE_DIR}/permutation_scale_4d_instances.hpp
     ${CMAKE_CURRENT_SOURCE_DIR}/permutation_scale_5d_instances.hpp
     ${CMAKE_CURRENT_SOURCE_DIR}/permutation_scale_6d_instances.hpp
     )

add_hiptensor_component(hiptensor_permutation_instances ${CK_PERMUTATION_INSTANCE_SOURCES})
target_include_directories(hiptensor_permutation_instances PRIVATE ${composable_kernel_INCLUDES})
