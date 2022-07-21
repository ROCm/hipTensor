include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include/hiptensor/ck
    ${PROJECT_SOURCE_DIR}/include/hiptensor
    ${PROJECT_SOURCE_DIR}/include/utility
)

set(CORE_INSTANCE_SOURCE
   ht_tensor.cpp
)

add_library(core_instance OBJECT ${CORE_INSTANCE_SOURCE}) 
if("${DEBUG_MODE}" STREQUAL "ON")
    target_compile_options(core_instance PRIVATE -DHT_DEBUG_MODE)
endif()
target_compile_features(core_instance PUBLIC)
set_target_properties(core_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
