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

find_package( composable_kernel 1.0.0 REQUIRED PATHS /opt/rocm /opt/rocm/ck $ENV{CK_DIR}/lib/cmake COMPONENTS device_contraction_operations device_other_operations)
rocm_package_add_dependencies("composable_kernel >= 1.0.0" COMPONENT tests)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Create test executables and deploy
function(add_hiptensor_component COMPONENT_NAME COMPONENT_SOURCES)

    message( STATUS "adding hiptensor component: ${COMPONENT_NAME}")

    # Ensure that all the sources are captured
    list(APPEND COMPONENT_SOURCES ${ARGN})

    # Build components as object files.
    # Make sure they have -fPIC and that they inherit the hip::device environment
    # so that they build for all archs in AMDGPU_TARGETS
    add_library(${COMPONENT_NAME} OBJECT ${COMPONENT_SOURCES})
    set_target_properties(${COMPONENT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${COMPONENT_NAME} PRIVATE hip::device)
    if(HIPTENSOR_BUILD_COMPRESSED_DBG)
        target_compile_options(${COMPONENT_NAME} PRIVATE
            "$<$<CONFIG:Debug>:-gz>"
            "$<$<CONFIG:RelWithDebInfo>:-gz>"
            )
    endif()

endfunction()

include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/library/include
    ${PROJECT_SOURCE_DIR}/library/src/include
)

# Generates hiptensor_contraction and hiptensor_contraction_instances
add_subdirectory(contraction)
# Generates hiptensor_permutation and hiptensor_permutation_instances
add_subdirectory(permutation)

# Core API code
set(HIPTENSOR_CORE_SOURCES
   ${CMAKE_CURRENT_SOURCE_DIR}/hiptensor.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/logger.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/performance.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/data_types.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/hip_device.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/handle.cpp
)

add_hiptensor_component(hiptensor_core ${HIPTENSOR_CORE_SOURCES})

# Generate shared lib
add_library(hiptensor SHARED
    $<TARGET_OBJECTS:hiptensor_core>
    $<TARGET_OBJECTS:hiptensor_contraction>
    $<TARGET_OBJECTS:hiptensor_contraction_instances>
    $<TARGET_OBJECTS:hiptensor_permutation>
    )

add_library(hiptensor::hiptensor ALIAS hiptensor)

target_compile_options(hiptensor PRIVATE ${CMAKE_CXX_FLAGS})
if(HIPTENSOR_BUILD_COMPRESSED_DBG)
    target_compile_options(hiptensor PRIVATE
        "$<$<CONFIG:Debug>:-gz>"
        "$<$<CONFIG:RelWithDebInfo>:-gz>"
        )
endif()

set ( hiptensor_SOVERSION 0.1 )
# nvcc can not recognize shared libraray file name with suffix other than *.so when linking.
if (NOT BUILD_WITH_COMPILER STREQUAL "HIP-NVCC")
  rocm_set_soversion(hiptensor ${hiptensor_SOVERSION})
endif()

# Users of hiptensor will need HIP libs
target_link_libraries(hiptensor INTERFACE hip::device hip::host)
target_link_libraries(hiptensor PRIVATE Threads::Threads)
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
