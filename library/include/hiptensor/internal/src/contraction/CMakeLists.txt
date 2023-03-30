include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/library/include/hiptensor/internal/src/contraction
    ${PROJECT_SOURCE_DIR}/library/include/hiptensor
    ${PROJECT_SOURCE_DIR}/library/include/hiptensor/internal
)

set(CONTRACTION_INSTANCE_SOURCE
   ht_contraction.cpp
)

add_library(contraction_instance OBJECT ${CONTRACTION_INSTANCE_SOURCE})
set_target_properties(contraction_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(ck)
