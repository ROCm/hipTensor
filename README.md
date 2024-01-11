# hipTensor

hiptensor is AMD's C++ library for accelerating tensor primitives
based on the composable kernel library,
through general purpose kernel languages, like HIP C++.

## GPU Support

* AMD CDNA class GPU featuring matrix core support:
gfx908, gfx90a, gfx940, gfx941, gfx942 as 'gfx9'

:::{note}
Double precision FP64 datatype support requires gfx90a, gfx940, gfx941 or gfx942
:::

## Minimum software requirements

* ROCm stack minimum version 5.7
* ROCm-cmake minimum version 0.8.0 for ROCm 5.7
* C++ 17
* CMake >=3.6
* Composable Kernel

Optional:

* doxygen (for building documentation)

## Documentation

Run the steps below to build documentation locally.

```bash
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Currently supported

### Operation: Contraction tensor

hiptensor supports a tensor contraction of the form $D = \alpha AB + \beta C$

Supported data-type combinations are:

| typeA       | typeB       | typeC       | typeCompute       | notes                              |
|-------------|-------------|-------------|-------------------|------------------------------------|
| bf16        | bf16        | bf16        | f32               |                                    |
| __half      | __half      | __half      | f32               |                                    |
| f32         | f32         | f32         | bf16              |                                    |
| f32         | f32         | f32         | __half            |                                    |
| f32         | f32         | f32         | f32               |                                    |
| f64         | f64         | f64         | f32               | f64 is only supported on gfx90a +  |
| f64         | f64         | f64         | f64               | f64 is only supported on gfx90a +  |
| cf32        | cf32        | cf32        | cf32              | cf32 is only supported on gfx90a + |
| cf64        | cf64        | cf64        | cf64              | cf64 is only supported on gfx90a + |

### Operation: Permutation tensor

Supported data-type combinations are:

| typeA     | typeB     | descCompute     | notes |
|-----------|-----------|-----------------|-------|
| f16       | f16       | f16             |       |
| f16       | f16       | f32             |       |
| f32       | f32       | f32             |       |

## Contributing to the code

1. Create and track a hipTensor fork.
2. Clone your fork:

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

<!-- markdownlint-disable ol-prefix -->
3. Create a pull request to ROCmSoftwarePlatform/hipTensor develop branch.
4. Await CI and approval feedback.
5. Once approved, merge!
<!-- markdownlint-enable ol-prefix -->

`Note: Please don't forget to install the githooks as there are triggers for clang formatting in commits.`

## Build with CMake

### Project options

| Option                  | Description                           | Default Value                                                  |
|-------------------------|---------------------------------------|----------------------------------------------------------------|
| AMDGPU_TARGETS          | Build code for specific GPU target(s) | gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx940;gfx941;gfx942 |
| HIPTENSOR_BUILD_TESTS   | Build Tests                           | ON                                                             |
| HIPTENSOR_BUILD_SAMPLES | Build Samples                         | ON                                                             |

### Example configurations

By default, the project is configured as Release mode.
Here are some of the examples for the configuration:
| Configuration                    | Command                                                                   |
|----------------------------------|---------------------------------------------------------------------------|
| Basic                            | `CC=hipcc CXX=hipcc cmake -B<build_dir> .`                                |
| Targeting gfx908                 | `CC=hipcc CXX=hipcc cmake -B<build_dir> . -DAMDGPU_TARGETS=gfx908:xnack-` |
| Debug build                      | `CC=hipcc CXX=hipcc cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug`       |
| Build without tests (default on) | `CC=hipcc CXX=hipcc cmake -B<build_dir> . -DHIPTENSOR_BUILD_TESTS=OFF`    |

After configuration, build with `cmake --build <build_dir> -- -j<nproc>`.

### Tips to reduce tests compile time

* Target a specific GPU (e.g. gfx908:xnack-)
* Use lots of threads (e.g. -j64)

## Running Unit Tests

### Logger tests

Tests API implementation of logger verbosity and functionality.

```bash
  <build_dir>/bin/logger_test
```

## Running contraction tests

* Bilinear contraction tests

Tests the API implementation of bilinear contraction algorithm with validation.

```bash
  <build_dir>/bin/bilinear_contraction_test
  <build_dir>/bin/complex_bilinear_contraction_test
```

* Scale contraction tests

Tests the API implementation of scale contraction algorithm with validation.

```bash
  <build_dir>/bin/scale_contraction_test
  <build_dir>/bin/complex_scale_contraction_test
```

## Running permutation tests

Test API implementation of the permutation algorithm with validation.

```bash
  <build_dir>/bin/permutation_test
```

## Samples

These are stand-alone use-cases of the hipTensor contraction operations.

### F32 bilinear contraction

Demonstrates the API implementation of bilinear contraction operation without validation.

```bash
  <build_dir>/bin/simple_bilinear_contraction_<typeA>_<typeB>_<typeC>_<typeD>_compute_<computeType>
```

### F32 scale contraction

Demonstrates the API implementation of scale contraction operation without validation.

```bash
  <build_dir>/bin/simple_scale_contraction_<typeA>_<typeB>_<typeD>_compute_<typeCompute>
```

### Permutation

Demonstrates the API implementation of permutation operation without validation.

```bash
  <build_dir>/bin/simple_permutation
```

### Build samples as external client

The client application links to the hipTensor library; therefore, you must install the
hipTensor library before building client applications.

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
