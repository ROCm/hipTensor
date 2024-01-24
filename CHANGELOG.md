# Changelog for hipTensor

Full documentation for hipTensor is available at [rocm.docs.amd.com/projects/hiptensor](https://rocm.docs.amd.com/projects/hipTensor/en/latest/index.html).

## hipTensor 1.2.0 for ROCm 6.1.0

### Additions

* API support
  * Permutation of rank 4 tensors
* Data type support
  * Contraction - new supported data types f16, bf16, complex f32, complex f64
* Samples
  * Added scale and bilinear contraction samples and tests for new supported data types.
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
