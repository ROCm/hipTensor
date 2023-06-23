###############################################################################
 #
 # MIT License
 #
 # Copyright (c) 2023 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.14)

if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc" CACHE STRING
    "Choose the type of compiler to build: Default point to hipcc"
    FORCE)
endif()

# NOTE: This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." )
endif()
message( VERBOSE "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE )

if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif()
if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE PATH "Install path prefix for packages.")
  set(CPACK_SET_DESTDIR OFF)
endif()

# This helps cmake properly find hip-config.cmake
list( APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} ${ROCM_PATH} /opt/rocm )
# Append our library helper cmake path and the cmake path for hip (for convenience).
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

# Build shared libs
set(BUILD_SHARED_LIBS ON)

project(hiptensor)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Project options
if( CMAKE_PROJECT_NAME STREQUAL "hiptensor" )
  option( HIPTENSOR_BUILD_TESTS "Build hiptensor tests" ON )
  option( HIPTENSOR_BUILD_SAMPLES "Build hiptensor samples" ON )
endif()

# Setup output paths
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)

# ROCm setup
find_package(ROCM 0.8 CONFIG REQUIRED)
include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMCheckTargetIds)

include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(CheckCXXCompilerFlag)

# gpu arch configuration
set( AMDGPU_TARGETS "all" CACHE STRING "Compile for which gpu architectures?")
set_property( CACHE AMDGPU_TARGETS PROPERTY STRINGS all gfx908:xnack- gfx90a:xnack+ gfx90a:xnack- )

# Detect if target ID syntax if supported for default all AMDGPU_TARGETS list
# Sets the AMDGPU_TARGETS with backward compatibility
if (AMDGPU_TARGETS STREQUAL "all")
  rocm_check_target_ids(target_list
      TARGETS "gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-"
  )
else()
  rocm_check_target_ids(target_list TARGETS "${AMDGPU_TARGETS}")
endif()

# must FORCE set this AMDGPU_TARGETS before any find_package( hip ...),
# to override CACHE var and set --offload-arch flags via hip-config.cmake hip::device dependency
set( AMDGPU_TARGETS "${target_list}" CACHE STRING "AMD GPU targets to compile for" FORCE )

# Setup HIP
find_package(hip REQUIRED )
message(STATUS "HIP version: ${hip_VERSION}")

# Configure library build
add_subdirectory(library/src)

# Configure testing setup
if(HIPTENSOR_BUILD_TESTS OR HIPTENSOR_BUILD_SAMPLES)
  enable_testing()
  rocm_package_setup_component(clients)
endif()

# Configure tests build
if(HIPTENSOR_BUILD_TESTS)
  rocm_package_setup_component(tests PARENT clients)
  add_subdirectory(test)
endif()

# Configure clients build
if(HIPTENSOR_BUILD_SAMPLES)
  rocm_package_setup_component(samples PARENT clients)
  add_subdirectory(samples)
endif()

# Versioning via rocm-cmake
set ( VERSION_STRING "1.0.0" )
rocm_setup_version( VERSION ${VERSION_STRING} )

# configure a header file to pass the CMake version settings to the source
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/library/include/hiptensor/internal/hiptensor-version.hpp.in"
               "${CMAKE_CURRENT_SOURCE_DIR}/library/include/hiptensor/hiptensor-version.hpp" )

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hiptensorConfigVersion.cmake"
    VERSION "${VERSION_STRING}"
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
     "${CMAKE_CURRENT_BINARY_DIR}/hiptensorConfig.cmake"
     INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hiptensor
     NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

rocm_install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/hiptensorConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/hiptensorConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hiptensor
)

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

rocm_create_package(
    NAME hiptensor
    DESCRIPTION "Adaptation library of tensor contraction with composable_kernel backend"
    MAINTAINER "hiptensor Maintainer <hiptensor-maintainer@amd.com>"
    LDCONFIG
)
