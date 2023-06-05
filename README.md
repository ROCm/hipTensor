# hipTensor
hiptensor is AMD's C++ library for accelerating tensor primitives based on the composable kernels,
through general purpose kernel languages, like HIP C++.

## GPU Support
* AMD CDNA class GPU featuring matrix core support: gfx908, gfx90a as 'gfx9'

`Note: Double precision FP64 datatype support requires gfx90a`

## Minimum Software Requirements
* ROCm stack minimum version 5.7
* ROCm-cmake minimum version 0.8.0 for ROCm 5.7
* C++ 17
* CMake >=3.6
* Composable Kernel

Optional:
* doxygen (for building documentation)

## Documentation

Run the steps below to build documentation locally.

```
cd docs

pip3 install -r .sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Currently supported
Operations - Contraction Tensor
Data Types - FP32 , FP64

## Contributing to the code
1. Create and track a hipTensor fork.
2. Clone your fork:
```
git clone -b develop https://github.com/<your_fork>/hipTensor.git .
.githooks/install
git checkout -b <new_branch>
...
git add <new_work>
git commit -m "What was changed"
git push origin <new_branch>
...
```
3. Create a pull request to ROCmSoftwarePlatform/hipTensor develop branch.
4. Await CI and approval feedback.
5. Once approved, merge!

`Note: Please don't forget to install the githooks as there are triggers for clang formatting in commits.`

## Build with CMake

### Project options
|Option|Description|Default Value|
|---|---|---|
|AMDGPU_TARGETS|Build code for specific GPU target(s)|gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+|
|HIPTENSOR_BUILD_TESTS|Build Tests|ON|
|HIPTENSOR_BUILD_SAMPLES|Build Samples|ON|

### Example configurations
By default, the project is configured as Release mode.
Here are some of the examples for the configuration:
|Configuration|Command|
|---|---|
|Basic|`CC=hipcc CXX=hipcc cmake -B<build_dir> .`|
|Targeting gfx908|`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DAMDGPU_TARGETS=gfx908:xnack-` |
|Debug build|`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug` |
|Build without tests (default on)|`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DHIPTENSOR_BUILD_TESTS=OFF` |

After configuration, build with `cmake --build <build_dir> -- -j<nproc>`

### Tips to reduce tests compile time:
- Target a specific GPU (e.g. gfx908:xnack-)
- Use lots of threads (e.g. -j64)

## Running Unit Tests

### Logger tests
Tests API implementation of logger verbosity and functionality.
o	<build_dir>/bin/logger_test

## Running Contraction Tests

### Bilinear contraction tests
Tests the API implementation of bilinear contraction algorithm with validation.
o	<build_dir>/bin/bilinear_contraction_f32_test
o	<build_dir>/bin/bilinear_contraction_f64_test

### Scale contraction tests
Tests the API implementation of scale contraction algorithm with validation.
o	<build_dir>/bin/scale_contraction_f32_test
o	<build_dir>/bin/scale_contraction_f64_test

### Samples
These are stand-alone use-cases of the hipTensor contraction operations.

## F32 Bilinear contraction
Demonstrates the API implementation of bilinear contraction operation without validation.
o	<build_dir>/bin/simple_contraction_bilinear_f32

## F32 Scale contraction
Demonstrates the API implementation of scale contraction operation without validation.
o	<build_dir>/bin/simple_contraction_scale_f32

### Build Samples as external client
Client application links to hipTensor library, and therefore hipTensor library needs to be installed before building client applications.

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
