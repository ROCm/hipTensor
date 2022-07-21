# hipTENSOR
Building hipTENSOR:
## Docker script
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

# Install the new rocm-cmake version
https://github.com/RadeonOpenCompute/rocm-cmake

## Build
```bash
mkdir build && cd build
```

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
#To run the hipTENSOR in debug mode for extensive debugging API results.
```bash
# Need to specify target ID, example below is gfx908 and gfx90a
cmake                                                                 \
-D BUILD_DEV=OFF                                                      \
-D CMAKE_BUILD_TYPE=Release                                           \
-D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 --offload-arch=gfx90a -O3" \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                             \
-D CMAKE_PREFIX_PATH=/opt/rocm                                        \
-D CMAKE_INSTALL_PREFIX=${PATH_TO_CK_INSTALL_DIRECTORY}               \
-D DEBUG_MODE=ON                                                   \
..
```
### Build and run hipTENSOR
```bash
 make
```

### Install hipTENSOR package
```bash
 make install
```
