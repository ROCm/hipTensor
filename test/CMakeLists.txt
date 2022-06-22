include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include/hiptensor
    ${PROJECT_SOURCE_DIR}/include/utility
    ${CMAKE_CK_CLONE_DIRECTORY}/external/include/half
    ${CMAKE_CK_CLONE_DIRECTORY}/include
    ${CMAKE_CK_CLONE_DIRECTORY}/include/ck
    ${CMAKE_CK_CLONE_DIRECTORY}/include/ck/utility
    ${CMAKE_CK_CLONE_DIRECTORY}/include/ck/host_utility
    ${CMAKE_CK_CLONE_DIRECTORY}/include/ck/tensor_description
    ${CMAKE_CK_CLONE_DIRECTORY}/include/ck/tensor
    ${CMAKE_CK_CLONE_DIRECTORY}/include/ck/problem_transform
    ${CMAKE_CK_CLONE_DIRECTORY}/include/ck/tensor_operation/gpu/block
    ${CMAKE_CK_CLONE_DIRECTORY}/include/ck/tensor_operation/gpu/warp
    ${CMAKE_CK_CLONE_DIRECTORY}/include/ck/tensor_operation/gpu/device
    ${CMAKE_CK_CLONE_DIRECTORY}/include/ck/tensor_operation/gpu/grid
    ${CMAKE_CK_CLONE_DIRECTORY}/include/ck/tensor_operation/gpu/thread
    ${CMAKE_CK_CLONE_DIRECTORY}/include/ck/tensor_operation/gpu/element
    ${CMAKE_CK_CLONE_DIRECTORY}/library/include/ck/library/host_tensor
    ${CMAKE_CK_CLONE_DIRECTORY}/library/include/ck/library/reference_tensor_operation/cpu
    ${CMAKE_CK_CLONE_DIRECTORY}/library/include/ck/library/reference_tensor_operation/gpu
    ${CMAKE_CK_CLONE_DIRECTORY}/library/include/ck/library/utility
)

add_custom_target(test)
function(add_executable_no_testing BINARY_NAME FILE_NAME)
    message( STATUS "adding binary ${BINARY_NAME}")
    add_executable(${BINARY_NAME} ${FILE_NAME})
    #target_link_libraries(${BINARY_NAME} ck_dep)
    target_link_libraries(${BINARY_NAME} PRIVATE hiptensor)
    #add_dependencies(${BINARY_NAME} ck)
    add_dependencies(test ${BINARY_NAME})
endfunction(add_executable_no_testing BINARY_NAME)


add_subdirectory(01_contraction)
