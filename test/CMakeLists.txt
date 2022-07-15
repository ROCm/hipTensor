include_directories(BEFORE
	${PROJECT_SOURCE_DIR}/include/hiptensor/ck
	${PROJECT_SOURCE_DIR}/include/hiptensor
	${PROJECT_SOURCE_DIR}/include/utility
    ${CMAKE_CK_CLONE_DIRECTORY}/library/include/ck/library/host_tensor
)

find_package(hipTENSOR 1.0.0 COMPONENTS hiptensor)
#find_package(composable_kernel 1.0.0 COMPONENTS host_tensor deviceoperations)
find_package(hip REQUIRED PATHS /opt/rocm)
message(STATUS "Build with HIP ${hip_VERSION}")

add_custom_target(test)
function(add_executable_no_testing BINARY_NAME FILE_NAME)
    message( STATUS "adding binary ${BINARY_NAME}")
    add_executable(${BINARY_NAME} ${FILE_NAME})
    if("${HT_DEBUG_MODE}" STREQUAL "ON")
        target_compile_options(${BINARY_NAME} PRIVATE -DHT_PRINT_DEBUG)
    endif()
    target_link_libraries(${BINARY_NAME} PRIVATE ck_dep_do ck_dep_ht hipTENSOR::hiptensor)
    add_dependencies(test ${BINARY_NAME})
endfunction(add_executable_no_testing BINARY_NAME)

add_subdirectory(01_contraction)
