include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include/hiptensor/ck
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
    ${CMAKE_CK_CLONE_DIRECTORY}/include/ck/tensor_operation/gpu/element
    ${CMAKE_CK_CLONE_DIRECTORY}/include/ck/tensor_operation/gpu/device
    ${CMAKE_CK_CLONE_DIRECTORY}/library/include/ck/library/tensor_operation_instance/gpu
    ${CMAKE_CK_CLONE_DIRECTORY}/library/include
    ${CMAKE_CK_CLONE_DIRECTORY}/library/include/ck/library/host_tensor
    ${CMAKE_CK_CLONE_DIRECTORY}/library/include/ck/library/reference_tensor_operation/cpu
    ${CMAKE_CK_CLONE_DIRECTORY}/library/include/ck/library/reference_tensor_operation/gpu
    ${CMAKE_CK_CLONE_DIRECTORY}/library/include/ck/library/utility
)

add_subdirectory(core)
add_subdirectory(contraction)

add_library(hiptensor SHARED 
	    $<TARGET_OBJECTS:core_instance>
	    $<TARGET_OBJECTS:contraction_instance>
        $<TARGET_OBJECTS:ck_core_instance>)

add_library(hipTENSOR::hiptensor ALIAS hiptensor)

target_compile_options(hiptensor PRIVATE --offload-arch=gfx908)
target_compile_features(hiptensor PUBLIC)
target_link_libraries(hiptensor ck_dep_ht ck_dep_do)
#target_link_libraries(hiptensor PRIVATE composable_kernel::host_tensor composable_kernel::device_operations)

set_target_properties(hiptensor PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(hiptensor PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CK_CLONE_DIRECTORY}/library/include/ck/library/host_tensor>"
)

set(DEV_OPS_INC_DIRS
    ${PROJECT_SOURCE_DIR}/include/utility/
    ${PROJECT_SOURCE_DIR}/include/hiptensor/
    ${CMAKE_CK_CLONE_DIRECTORY}/library/include/ck/library/host_tensor/
)

rocm_install(
    TARGETS hiptensor
    EXPORT hiptensorTargets
)

rocm_install(DIRECTORY ${DEV_OPS_INC_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ht)
rocm_install (EXPORT hiptensorTargets
    FILE hipTENSORhiptensorTargets.cmake
    NAMESPACE hipTENSOR::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipTENSOR
)
