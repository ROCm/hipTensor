cmake_minimum_required(VERSION 3.14)
project(hipTENSOR)

include(ExternalProject)

set(ROCM_SYMLINK_LIBS OFF)
find_package(ROCM 0.8 REQUIRED PATHS /opt/rocm)

include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMSetupVersion)
include(ROCMInstallSymlinks)
include(ROCMCreatePackage)
include(CheckCXXCompilerFlag)

rocm_setup_version(VERSION 0.2.0)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX}
    ${CMAKE_INSTALL_PREFIX}/llvm ${CMAKE_INSTALL_PREFIX}/hip /opt/rocm
    /opt/rocm/llvm /opt/rocm/hip ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")


if(NOT BUILD_DEV)
  set(BUILD_DEV "OFF" CACHE STRING
    "Choose the type of build, options are: ON, OFF."
    FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif()

if(NOT CMAKE_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS "--offload-arch=gfx908 --offload-arch=gfx90a -O3" CACHE STRING
    "Choose the type of chipset need to build: Default building for MI200 board"
    FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc" CACHE STRING
    "Choose the type of compiler to build: Default point to the hipcc"
    FORCE)
endif()


if(NOT CMAKE_PREFIX_PATH)
  set(CMAKE_PREFIX_PATH "/opt/rocm" CACHE STRING
    "Choose the path to be pointed: Default point to /opt/rocm"
    FORCE)
endif()

if(NOT HT_DEBUG_MODE)
  set(HT_DEBUG_MODE "OFF" CACHE STRING
    "Choose the type of build, options are: ON, OFF."
    FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_CK_CLONE_DIRECTORY  ${CMAKE_BINARY_DIR}/composable_kernel)
set(CMAKE_CK_CLONE_LIB  ${CMAKE_CK_CLONE_DIRECTORY}/build/lib)

ExternalProject_Add(ck
	SOURCE_DIR       ${CMAKE_CK_CLONE_DIRECTORY}
	GIT_REPOSITORY   https://github.com/ROCmSoftwarePlatform/composable_kernel.git
	GIT_TAG          origin/contraction_hipTENSOR
	GIT_PROGRESS     true
	LIST_SEPARATOR   "|"
	PATCH_COMMAND    mkdir -p build
	BINARY_DIR       ${CMAKE_CK_CLONE_DIRECTORY}/build
	CMAKE_ARGS       -D BUILD_DEV=${BUILD_DEV}
                     -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                     -D CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                     -D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                     -D CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}

	BUILD_COMMAND    cmake ${CMAKE_ARGS} .. COMMAND make -j all
    INSTALL_COMMAND  COMMAND make install
	BUILD_ALWAYS     FALSE

)

add_library(ck_dep_ht SHARED IMPORTED)
set_target_properties(ck_dep_ht PROPERTIES IMPORTED_LOCATION ${CMAKE_CK_CLONE_LIB}/libhost_tensor.a)
add_library(ck_dep_do SHARED IMPORTED)
set_target_properties(ck_dep_do PROPERTIES IMPORTED_LOCATION ${CMAKE_CK_CLONE_LIB}/libdevice_operations.a)

add_subdirectory(src)
add_subdirectory(test)
if(CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
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
    MAINTAINER "Chaitanya Inumella <Chaitanya.Inumella@amd.com>"
    LDCONFIG
    HEADER_ONLY
)

