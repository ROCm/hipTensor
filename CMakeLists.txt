cmake_minimum_required(VERSION 3.14)
project(hipTENSOR)

include(cmake/GitUtils.cmake)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif()

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/bin)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)

set(CK_DEP_DIR "external")

if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${CK_DEP_DIR})
  message(STATUS "Creating the external directory for the composable kernels.......")
  file(MAKE_DIRECTORY ${CK_DEP_DIR})
else()
  message(STATUS "external dependencies directory already present")
endif()

git_clone(
	PROJECT_NAME    composable_kernels
        GIT_URL         "https://github.com/ROCmSoftwarePlatform/composable_kernel.git"
        GIT_BRANCH      contraction
	DIRECTORY       ${PROJECT_SOURCE_DIR}/${CK_DEP_DIR}
)

#include(${ucm_SOURCE_DIR}/cmake/ucm.cmake)

#ucm_print_flags()

#add_executable(example example.cpp)
