.. meta::
   :description: A high-performance HIP library for tensor primitives
   :keywords: hipTensor, ROCm, library, API, tool

.. _what-is-hiptensor:

============================================================================
What is hipTensor?
============================================================================

hipTensor is a work-in-progress (WIP) high-performance HIP library for tensor primitives. It is AMD's C++ library for accelerating tensor primitives which can leverage specialized GPU matrix cores on AMD's latest discrete GPUs. hipTensor is currently powered by the composable kernel library. The API is designed to be portable with the Nvidia cuTensor library, allowing those users to easily migrate to the AMD platform.

The hipTensor API offers functionality for defining tensor data objects and currently supports contraction, permutation and reduction operations on the tensor objects. Users may also control several available logging options. Under the hood, the hipTensor library is bundled with multitude of GPU kernels which are automatically selected and invoked to solve problems as defined by input parameters to the API. As hipTensor is currently a WIP, future tensor operation support may include reductions, element-wise operations and caching of selection instances.

Supporting host code is required for GPU device and memory management. The hipTensor code samples and tests provided are built and launched via
the Heterogeneous-Compute Interface for Portability (HIP) ecosystem within ROCm.
