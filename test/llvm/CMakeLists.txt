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

# Find / configure LLVM
find_package(LLVM REQUIRED CONFIG)

# Required:
# LLVMObjectYAML for YAML parsing
# LLVMSupport for CL processing
find_library(LLVMObjectYAML_LIBRARY NAMES LLVMObjectYAML PATHS ${LLVM_LIBRARY_DIR})
find_library(LLVMSupport_LIBRARY NAMES LLVMSupport PATHS ${LLVM_LIBRARY_DIR})

message(STATUS "adding hiptensor component: hiptensor_llvm")
message(STATUS "LLVM_LIBRARY_DIR: ${LLVM_LIBRARY_DIR}")
message(STATUS "LLVMObjectYAML_LIBRARY: ${LLVMObjectYAML_LIBRARY}")
message(STATUS "LLVMSupport_LIBRARY: ${LLVMSupport_LIBRARY}")

if(NOT LLVMObjectYAML_LIBRARY)
    message(ERROR "Could not find library: LLVMObjectYAML" )
endif()
if(NOT LLVMSupport_LIBRARY)
    message(ERROR "Could not find library: LLVMSupport")
endif()

# LLVM Libs we need to link against
set(HIPTENSOR_LLVM_LIBS "-L${LLVM_LIBRARY_DIR}"
                        "-Wl,-rpath=${LLVM_LIBRARY_DIR}"
                        LLVMObjectYAML
                        LLVMSupport
                        hip::device
                        )

# Includes
set(HIPTENSOR_LLVM_INCLUDES ${LLVM_INCLUDE_DIRS}
                            ${PROJECT_SOURCE_DIR}/library/include
                            ${PROJECT_SOURCE_DIR}/test
                            )

# Sources for this static object
set(HIPTENSOR_LLVM_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/yaml_parser_contraction_config.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/hiptensor_options.cpp)

# Create hiptensor_llvm object target
# All LLVM resources are private to this object.
# Interface will fwd includes of this dir to users of the object.
add_library(hiptensor_llvm OBJECT ${HIPTENSOR_LLVM_SOURCES})
set_target_properties(hiptensor_llvm PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hiptensor_llvm PRIVATE ${HIPTENSOR_LLVM_LIBS})
target_include_directories(hiptensor_llvm PRIVATE ${HIPTENSOR_LLVM_INCLUDES})
target_include_directories(hiptensor_llvm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(hiptensor_llvm PRIVATE -fno-rtti)
