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

#[[
add_custom_target(src)
function(add_executable_no_testing BINARY_NAME FILE_NAME)
    message( STATUS "adding binary ${BINARY_NAME}")
    add_executable(${BINARY_NAME} ${FILE_NAME})
    target_link_libraries(${BINARY_NAME} ck_dep)
    target_sources(${BINARY_NAME} PRIVATE $<TARGET_OBJECTS:core_instance>)
    add_dependencies(${BINARY_NAME} ck)
    add_dependencies(src ${BINARY_NAME})
endfunction(add_executable_no_testing BINARY_NAME)
]]


add_subdirectory(core)

add_library(hiptensor SHARED 
	    $<TARGET_OBJECTS:core_instance>)
add_library(hipTENSOR::hiptensor ALIAS hiptensor)

set(HT_OPS_INC_DIRS
    ${PROJECT_SOURCE_DIR}/include/
)


target_compile_features(hiptensor PUBLIC)
set_target_properties(hiptensor PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(hiptensor PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hiptensor>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/utility>"
)

install(TARGETS hiptensor
	EXPORT hiptensorTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY ${HT_OPS_INC_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ht)

install(EXPORT hiptensorTargets
    FILE hipTENSORhost_tensorTargets.cmake
    NAMESPACE hipTENSOR::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipTENSOR
)


#add_subdirectory(contraction)
