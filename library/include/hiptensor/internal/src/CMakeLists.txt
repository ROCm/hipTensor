
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

# Users of hiptensor will need HIP and CK libs
target_link_libraries(hiptensor INTERFACE hip::device hip::host composable_kernel::device_operations)
set_target_properties(hiptensor PROPERTIES POSITION_INDEPENDENT_CODE ON)

rocm_install_targets(
    TARGETS hiptensor
    EXPORT hiptensorTargets
    INCLUDE ${PROJECT_SOURCE_DIR}/internal/include
)

rocm_install (EXPORT hiptensorTargets
    FILE hiptensorTargets.cmake
    NAMESPACE hip::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hiptensor
)
