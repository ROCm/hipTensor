include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include/hiptensor
    ${PROJECT_SOURCE_DIR}/include/utility
)

set(CORE_INSTANCE_SOURCE
   ht_tensor.cpp;
)


add_library(core_instance OBJECT ${CORE_INSTANCE_SOURCE}) 
target_compile_features(core_instance PUBLIC)
set_target_properties(core_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
