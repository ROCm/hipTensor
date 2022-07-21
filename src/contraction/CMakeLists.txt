include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include/hiptensor/ck
    ${PROJECT_SOURCE_DIR}/include/hiptensor
    ${PROJECT_SOURCE_DIR}/include/utility
)

set(CONTRACTION_INSTANCE_SOURCE
   ht_contraction.cpp
)

add_library(contraction_instance OBJECT ${CONTRACTION_INSTANCE_SOURCE})
if("${DEBUG_MODE}" STREQUAL "ON")
    target_compile_options(contraction_instance PRIVATE -DHT_DEBUG_MODE)
endif()
target_compile_features(contraction_instance PUBLIC)
set_target_properties(contraction_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(ck)
