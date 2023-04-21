##
Client application links to hiptensor library, and therefore hiptensor library needs to be installed before building client applications.


## Build
```bash
mkdir -p samples/build
cd samples/build
```

```bash
cmake                                                                                                  \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                              \
-D CMAKE_PREFIX_PATH="/opt/rocm;${PATH_TO_HIPTENSOR_INSTALL_DIRECTORY};${PATH_TO_CK_INSTALL_DIRECTORY} \
..
```

### Build client example
```bash
 make
```
