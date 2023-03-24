cmake_minimum_required(VERSION 3.14)

# NOTE: This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." )
endif()
message( VERBOSE "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE )

project(hiptensor)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Project options
if( CMAKE_PROJECT_NAME STREQUAL "hiptensor" )
  option( HIPTENSOR_BUILD_TESTS "Build hipTensor tests" ON )
  option( HIPTENSOR_BUILD_SAMPLES "Build hipTensor samples" ON )
  option( HIPTENSOR_BUILD_DEV "Developer build mode" OFF )
  option( HIPTENSOR_BUILD_DOCS "Build hipTensor documentation" OFF )
endif()

if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif()
if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE PATH "Install path prefix for packages.")
  set(CPACK_SET_DESTDIR OFF)
endif()
set(BUILD_SHARED_LIBS ON)

# This helps cmake properly find hip-config.cmake
list( APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} ${ROCM_PATH} /opt/rocm )
# Append our library helper cmake path and the cmake path for hip (for convenience).
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

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

rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
  TARGETS "gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+" )

# Variable AMDGPU_TARGET must be a cached variable and must be specified before calling find_package(hip)
# This is because hip-config.cmake sets --offload-arch via AMDGPU_TARGET cached variable __after__ setting
# default cached variable AMDGPU_TARGET to "gfx900;gfx906;gfx908;gfx1100;gfx1101;gfx1102", where not all archs are compatible with MFMA instructions
#
# By rule, once cached variable is set, it cannot be overridden unless we use the FORCE option
if(AMDGPU_TARGETS)
  set(AMDGPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
else()
  set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
endif()
message( VERBOSE "AMDGPU_TARGETS=${AMDGPU_TARGETS}")

rocm_setup_version(VERSION 0.2.0)
#list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}
#    ${CMAKE_INSTALL_PREFIX}/llvm ${CMAKE_INSTALL_PREFIX}/hip /opt/rocm
#    /opt/rocm/llvm /opt/rocm/hip ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc" CACHE STRING
    "Choose the type of compiler to build: Default point to the hipcc"
    FORCE)
endif()


if(NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE STRING
    "Choose the path to be pointed: Default point to /opt/rocm"
    FORCE)
endif()


include( FetchContent )

FetchContent_Declare(ck
	GIT_REPOSITORY   https://github.com/ROCmSoftwarePlatform/composable_kernel.git
	GIT_TAG          develop
	GIT_PROGRESS     true
)

FetchContent_GetProperties(ck)
if(NOT ck_POPULATED)
  # Fetch the content using default details
  FetchContent_Populate(ck)

  # Populate specific ck vars
  set(GPU_TARGETS ${AMDGPU_TARGETS})
  set(BUILD_DEV ${HIPTENSOR_BUILD_DEV})

  # Don't add CK to the 'all' target
  add_subdirectory(${ck_SOURCE_DIR} ${ck_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

#add_library(device_operations SHARED IMPORTED)
#set_target_properties(device_operations PROPERTIES IMPORTED_LOCATION ${CMAKE_CK_CLONE_LIB}/libdevice_operations.a)

add_subdirectory(src)
add_subdirectory(test)

if(HIPTENSOR_BUILD_DOCS MATCHES "ON")
  # Add the cmake folder so the FindSphinx module is found
  set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
  add_subdirectory (docs)
endif()

#Create an interface target for the include only files and call it "hipTENSOR"
include(CMakePackageConfigHelpers)

set(version 1.0.0)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hipTENSORConfigVersion.cmake"
    VERSION "${version}"
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
     "${CMAKE_CURRENT_BINARY_DIR}/hipTENSORConfig.cmake"
     INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipTENSOR
     NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

rocm_install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/hipTENSORConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/hipTENSORConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipTENSOR
)

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

rocm_create_package(
    NAME hipTENSOR
    DESCRIPTION "Adaptation library of tensor contraction with composable_kernel backend"
    MAINTAINER "hipTensor Maintainer <hiptensor-maintainer@amd.com>"
    LDCONFIG
    HEADER_ONLY
)

