include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include/hiptensor/ck
    ${PROJECT_SOURCE_DIR}/include/hiptensor
    ${PROJECT_SOURCE_DIR}/include/utility
)

set(CK_CORE_SOURCE
   ht_ck_core.cpp
)

add_library(ck_core_instance OBJECT ${CK_CORE_SOURCE})
if("${DEBUG_MODE}" STREQUAL "ON")
    target_compile_options(ck_core_instance PRIVATE -DHT_DEBUG_MODE)
endif()
target_compile_features(ck_core_instance PUBLIC)
set_target_properties(ck_core_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
