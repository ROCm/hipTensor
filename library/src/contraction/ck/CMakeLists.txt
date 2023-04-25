include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/library/include/hiptensor
    ${PROJECT_SOURCE_DIR}/library/include/hiptensor/internal
    ${PROJECT_SOURCE_DIR}/library/include/hiptensor/internal/ck
)

set(CK_CORE_SOURCE
   ht_ck_core.cpp
)

add_library(ck_core_instance OBJECT ${CK_CORE_SOURCE})
target_include_directories(ck_core_instance PRIVATE ${composable_kernel_INCLUDES})
target_link_libraries(ck_core_instance PRIVATE hip::host)
set_target_properties(ck_core_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
