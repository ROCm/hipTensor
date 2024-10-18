# Changelog for hipTensor

Full documentation for hipTensor is available at [rocm.docs.amd.com/projects/hiptensor](https://rocm.docs.amd.com/projects/hipTensor/en/latest/index.html).

## (Unreleased) hipTensor 1.4.0 for ROCm 6.3.0

### Additions

* Added API support for tensor reduction of ranks 2, 3, 4, 5 and 6
* Added CPU reference for tensor reductions
* Added unit tests for tensor reductions
* Added documentation for tensor reductions
* Added support for environment variable HIPTENSOR_DEFAULT_STRIDES_ROW_MAJOR to use row-major convention when calculating default strides

### Changes

* Updated target archs for ASAN builds
* ASAN library builds now use -mcmodel=large to accommodate larger lib size
* Updated permute backend to accommodate changes to element-wise ops implementation
* Updated validation acceptance criteria to match CK backend tests
* Default strides calculations now follow column-major convention

### Fixes

* Fixed a bug in randomized tensor input data generation
* Various documentation formatting updates and fixes
* Split kernel instances to improve build times
* Fixed small memory leak by properly destroying HIP event objects in tests

## hipTensor 1.3.0 for ROCm 6.2.0

### Additions

* Added support for tensor permutation of ranks of 2, 3, 4, 5 and 6
* Added tests for tensor permutation of ranks of 2, 3, 4, 5 and 6
* Added support for tensor contraction of M6N6K6: M, N, K up to rank 6
* Added tests for tensor contraction of M6N6K6: M, N, K up to rank 6
* Added new test YAML parsing to support sequential parameters ordering

### Changes

* Documentation updates for installation, programmer's guide and API reference
* Prefer amd-llvm-devel package before system LLVM library
* Preferred compilers changed to CC=amdclang CXX=amdclang++
* Updated actor-critic selection for new contraction kernel additions

### Fixes

* Fixed LLVM parsing crash
* Fixed memory consumption issue in complex kernels
* Work-around implemented for compiler crash during debug build
* Allow random modes ordering for tensor contractions

## hipTensor 1.2.0 for ROCm 6.1.0

### Additions

* API support for permutation of rank 4 tensors: f16 and f32
* New datatype support in contractions of rank 4: f16, bf16, complex f32, complex f64
* Added scale and bilinear contraction samples and tests for new supported data types
* Added permutation samples and tests for f16, f32 types

### Fixes

* Fixed bug in contraction calculation with data type f32

## hipTensor 1.1.0 for ROCm 6.0.0

### Additions

* Architecture support for gfx940, gfx941, and gfx942
* Client tests configuration parameters now support YAML file input format

### Changes

* Doxygen now treats warnings as errors

### Fixes

* Client tests output redirections now behave accordingly
* Removed dependency static library deployment
* Security issues for documentation
* Compile issues in debug mode
* Corrected soft link for ROCm deployment

## hipTensor 1.0.0 for ROCm 5.7.0

### Additions

* Initial prototype enablement of hipTensor library that supports tensor operations
* Kernel selection support for Default and Actor-Critic algorithms
* API support for:
  * Definition and contraction of rank 4 tensors
  * Contextual logging and output redirection
  * Kernel selection caching
* Data type support for f32 and f64
* Architecture support for gfx908 and gfx90a
