# hipTensor

hiptensor is AMD's C++ library for accelerating tensor primitives based on the composable kernel
library. hiptensor uses general purpose kernel languages, such as HIP C++.

## Support

* AMD CDNA class GPU featuring matrix core support: gfx908, gfx90a, gfx940, gfx941, gfx942 (as 'gfx9')

    ```note
    Double-precision FP64 datatype support requires gfx90a, gfx940, gfx941, or gfx942
    ```

* Operations: Contraction Tensor
* Data types: FP32 , FP64

## Prerequisites

* ROCm stack version 5.7 and above
* ROCm-cmake version 0.8.0 and above for ROCm 5.7
* C++ 17
* CMake 3.6 and above
* Composable Kernel

Optional:

* Doxygen (for building documentation)

## Documentation

To build our documentation locally, use the following code.

```bash
cd docs
pip3 install -r .sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Building hipTensor with CMake

### Project options

| Option | Description | Default value |
|---|---|---|
| AMDGPU_TARGETS | Build code for specific GPU target(s) | gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx940;gfx941;gfx942 |
| HIPTENSOR_BUILD_TESTS | Build Tests | ON |
| HIPTENSOR_BUILD_SAMPLES | Build Samples | ON |

### Example configurations

By default, the project is configured as 'Release mode'. Here are some other configuration examples:

| Configuration | Command |
|---|---|
| Basic | `CC=hipcc CXX=hipcc cmake -B<build_dir> .` |
| Targeting gfx908 | `CC=hipcc CXX=hipcc cmake -B<build_dir> . -DAMDGPU_TARGETS=gfx908:xnack-` |
| Debug build | `CC=hipcc CXX=hipcc cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug` |
| Build without tests (default on) | `CC=hipcc CXX=hipcc cmake -B<build_dir> . -DHIPTENSOR_BUILD_TESTS=OFF` |

After you've selected your configuration, you can build using:

```bash
cmake --build <build_dir> -- -j<nproc>
```

## Running tests

To reduce test compile time, you can:

* Target a specific GPU (e.g., `gfx908:xnack-`)
* Use many threads (e.g., `-j64`)

### Unit tests

* Logger: Tests API implementation of logger verbosity and functionality

  * `<build_dir>/bin/logger_test`

### Contraction tests

* Bilinear contraction: Tests API implementation of bilinear contraction algorithm with validation
  * `<build_dir>/bin/bilinear_contraction_f32_test`
  * `<build_dir>/bin/bilinear_contraction_f64_test`

* Scale contraction: Tests API implementation of scale contraction algorithm with validation
  * `<build_dir>/bin/scale_contraction_f32_test`
  * `<build_dir>/bin/scale_contraction_f64_test`

* Samples: Stand-alone use-cases of the hipTensor contraction operations.

* F32 bilinear contraction: Demonstrates API implementation of bilinear contraction operation without
  validation.
  * `<build_dir>/bin/simple_contraction_bilinear_f32`

* F32 scale contraction: Demonstrates API implementation of scale contraction operation without
  validation.
  * `<build_dir>/bin/simple_contraction_scale_f32`

### Build samples as external client

You must install the hipTensor library before building client applications, because the client application
links to this library.

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

Build client example:

```bash
 make
```

## Contributing to hipTensor code

1. Create and track a hipTensor fork
2. Clone your fork

    ```bash
    git clone -b develop https://github.com/<your_fork>/hipTensor.git .
    .githooks/install
    git checkout -b <new_branch>
    ...
    git add <new_work>
    git commit -m "What was changed"
    git push origin <new_branch>
    ...
    ```

3. Create a pull request to ROCmSoftwarePlatform/hipTensor develop branch
4. Await CI and approval feedback
5. Once approved, you can merge your changes

```important
Don't forget to install githooks. There are triggers for Clang formatting in commits.
```
