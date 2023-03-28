
find_package( composable_kernel 1.0.0 REQUIRED PATHS /opt/rocm /opt/rocm/ck $ENV{CK_DIR}/lib/cmake COMPONENTS device_operations )
rocm_package_add_dependencies("composable_kernel >= 1.0.0" COMPONENT tests)

add_subdirectory(core)
add_subdirectory(contraction)

add_library(hiptensor SHARED 
	    $<TARGET_OBJECTS:core_instance>
	    $<TARGET_OBJECTS:contraction_instance>
        $<TARGET_OBJECTS:ck_core_instance>
        )

add_library(hiptensor::hiptensor ALIAS hiptensor)

target_compile_options(hiptensor PRIVATE ${CMAKE_CXX_FLAGS})
target_include_directories(hiptensor PUBLIC
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/hiptensor/ck>
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/hiptensor>
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/utility>
    $<INSTALL_INTERFACE:../include/hiptensor/ck>
    $<INSTALL_INTERFACE:../include/hiptensor>
    $<INSTALL_INTERFACE:../include/utility>
)
target_compile_features(hiptensor PUBLIC)

# Users of hiptensor will need HIP and CK libs
target_link_libraries(hiptensor INTERFACE hip::device hip::host composable_kernel::device_operations)
set_target_properties(hiptensor PROPERTIES POSITION_INDEPENDENT_CODE ON)

#set(DEV_OPS_INC_DIRS
#    ${PROJECT_SOURCE_DIR}/include/utility/
#    ${PROJECT_SOURCE_DIR}/include/hiptensor/
#)

rocm_install_targets(
    TARGETS hiptensor
    EXPORT hiptensorTargets
    INCLUDES ../library/include
)

#rocm_install(DIRECTORY ${DEV_OPS_INC_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ht)
rocm_install (EXPORT hiptensorTargets
    FILE hiptensorTargets.cmake
    NAMESPACE hiptensor::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hiptensor
)
