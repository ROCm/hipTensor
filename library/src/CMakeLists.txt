###############################################################################
 #
 # MIT License
 #
 # Copyright 2023 Advanced Micro Devices, Inc.
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
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
 #
 ###############################################################################

find_package( composable_kernel 1.0.0 REQUIRED PATHS /opt/rocm /opt/rocm/ck $ENV{CK_DIR}/lib/cmake COMPONENTS device_operations )
rocm_package_add_dependencies("composable_kernel >= 1.0.0" COMPONENT tests)

include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/library/include/hiptensor
    ${PROJECT_SOURCE_DIR}/library/src/include
)

# Generates hiptensor_contraction and hiptensor_contraction_instances
add_subdirectory(contraction)

# Core API code
set(HIPTENSOR_CORE_SOURCES
   ${CMAKE_CURRENT_SOURCE_DIR}/hiptensor.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/logger.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/types.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/hip_device.cpp
)

add_library(hiptensor_core OBJECT ${HIPTENSOR_CORE_SOURCES})
target_compile_features(hiptensor_core PUBLIC)
set_target_properties(hiptensor_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Generate shared lib
add_library(hiptensor SHARED
	    $<TARGET_OBJECTS:hiptensor_core>
	    $<TARGET_OBJECTS:hiptensor_contraction>
        $<TARGET_OBJECTS:hiptensor_contraction_instances>
        )

add_library(hiptensor::hiptensor ALIAS hiptensor)

target_compile_options(hiptensor PRIVATE ${CMAKE_CXX_FLAGS})

# Users of hiptensor will need HIP libs
target_link_libraries(hiptensor INTERFACE hip::device hip::host)
set_target_properties(hiptensor PROPERTIES POSITION_INDEPENDENT_CODE ON)

rocm_install_targets(
    TARGETS hiptensor
    EXPORT hiptensorTargets
    INCLUDE ${PROJECT_SOURCE_DIR}/library/include
)

rocm_install (EXPORT hiptensorTargets
    FILE hiptensorTargets.cmake
    NAMESPACE hiptensor::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hiptensor
)
