##
Client application links to hiptensor library, and therefore hiptensor library needs to be installed before building client applications.


## Build
```bash
mkdir -p samples/build
cd samples/build
```

```bash
hiptensor_DIR=${PATH_TO_HIPTENSOR_INSTALL_DIRECTORY}                  \
CK_DIR=${PATH_TO_CK_INSTALL_DIRECTORY}                                \
cmake                                                                 \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                             \
..
```

### Build client example
```bash
 make
```
