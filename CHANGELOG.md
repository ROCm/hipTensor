# Change Log for hipTensor

Full documentation for hipTensor is available at [hiptensor.readthedocs.io](https://hiptensor.readthedocs.io/en/latest/).

## hipTensor 1.1.0  for ROCm 6.0.0

### Added
- Added architecture support for gfx940, gfx941 and gfx942
- Client tests configuration parameters now support YAML file input format

### Changed
- Doxygen now treats warnings as errors

### Fixed
- Client tests output redirections now behave accordingly
- Removed dependency static library deployment
- Security fixes for documentation
- Fixed compile issues in debug mode
- Fixed soft link for ROCm deployment

## hipTensor 1.0.0  for ROCm 5.7.0

### Added
- Initial prototype enablement of hipTensor library supporting tensor operations
- Kernel selection support for Default and Actor-Critic algorithms
- API support for definition and contraction of rank 4 tensors
- API support for contextual logging and output redirection
- API support for kernel selection caching
- Datatype support for f32 and f64
- Architecture support for gfx908 and gfx90a
