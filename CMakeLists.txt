cmake_minimum_required(VERSION 3.14)
project(hipTENSOR)

include(ExternalProject)

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

	BUILD_COMMAND    cmake ${CMAKE_ARGS} .. COMMAND make -j examples
	INSTALL_COMMAND  ""
	BUILD_ALWAYS     FALSE

)

add_library(ck_dep SHARED IMPORTED)
set_target_properties(ck_dep PROPERTIES IMPORTED_LOCATION ${CMAKE_CK_CLONE_LIB}/libhost_tensor.a)

add_subdirectory(src)
