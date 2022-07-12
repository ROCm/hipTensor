include_directories(BEFORE
	${PROJECT_SOURCE_DIR}/include/hiptensor/ck
	${PROJECT_SOURCE_DIR}/include/hiptensor
	${PROJECT_SOURCE_DIR}/include/utility
    	${CMAKE_CK_CLONE_DIRECTORY}/library/include/ck/library/host_tensor
)

find_package(hipTENSOR 1.0.0 COMPONENTS hiptensor)
find_package(hip REQUIRED PATHS /opt/rocm)
message(STATUS "Build with HIP ${hip_VERSION}")

add_subdirectory(01_contraction)
