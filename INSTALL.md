# Installing hipTENSOR package
## Pre-requisites
### Docker script to setup env
```bash
docker run                                     \
-it                                            \
--privileged                                   \
--group-add sudo                               \
-w /root/workspace                             \
-v ${PATH_TO_LOCAL_WORKSPACE}:/root/workspace  \
rocm/tensorflow:rocm5.1-tf2.6-dev              \
/bin/bash
```

### Install the new rocm-cmake version
https://github.com/RadeonOpenCompute/rocm-cmake

## Build steps
Create and move to the build directory.
```bash 
mkdir build && cd build 
```
### For building the hipTENSOR packages, a couple of options are available 
1. Regular build with the package with build type, AARCH, Compiler, Prefix, and Install paths.
```bash
# Need to specify target ID, example below is gfx908 and gfx90a
cmake                                                                 \
-D BUILD_DEV=OFF                                                      \
-D CMAKE_BUILD_TYPE=Release                                           \
-D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 --offload-arch=gfx90a -O3" \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                             \
-D CMAKE_PREFIX_PATH=/opt/rocm                                        \
-D CMAKE_INSTALL_PREFIX=${PATH_TO_CK_INSTALL_DIRECTORY}               \
..
```
2. To build the hipTENSOR in debug mode for extensive debugging of APIs add the flag -D DEBUG_MODE=ON.
```bash
# Need to specify target ID, example below is gfx908 and gfx90a
cmake                                                                 \
-D BUILD_DEV=OFF                                                      \
-D CMAKE_BUILD_TYPE=Release                                           \
-D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 --offload-arch=gfx90a -O3" \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                             \
-D CMAKE_PREFIX_PATH=/opt/rocm                                        \
-D CMAKE_INSTALL_PREFIX=${PATH_TO_CK_INSTALL_DIRECTORY}               \
-D DEBUG_MODE=ON                                                      \
..
```
### Build and run hipTENSOR
* NOTE: Parallel build isn't supported due to the unavailability of the parallel support with the backend i.e., composable_kernel (CK).
```bash
 make
```

## Install hipTENSOR package
* The package will be installed to the path specified in the CMAKE_INSTALL_PREFIX cmake flag.
```bash
 make install
```
