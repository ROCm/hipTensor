.. meta::
   :description: A high-performance HIP library for tensor primitives
   :keywords: hipTensor, ROCm, library, API, tool

.. _index:

===========================
hipTensor documentation
===========================

hipTensor is a work-in-progress (WIP) high-performance Heterogeneous-Compute Interface for Portability (HIP) library for tensor primitives. It’s AMD’s C++ library for accelerating tensor primitives, which can leverage specialized GPU matrix cores on AMD’s latest discrete GPUs. hipTensor is currently powered by the composable kernel (CK) library backend. The API is designed to be portable with the NVIDIA CUDA cuTensor library, allowing those users to easily migrate to the AMD platform.

The hipTensor API offers functionality for defining tensor data objects and currently supports contraction, permutation and reduction operations on the tensor objects. Users may also control several available logging options. The hipTensor library is bundled with GPU kernel instances, which are automatically selected and invoked to solve problems as defined by the API input parameters. As hipTensor is a WIP, future tensor operation support might include additional element-wise operations and caching of selection instances.

Supporting host code is required for GPU device and memory management. The hipTensor code samples and tests provided are built and launched via the HIP ecosystem within ROCm.

hipTensor library code is open source and available via the `GitHub repository <https://github.com/ROCm/hipTensor>`_.

The documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :ref:`installation`

  .. grid-item-card:: Conceptual

    * :ref:`programmers-guide`

  .. grid-item-card:: API reference

    * :ref:`api-reference`

  .. grid-item-card:: Contribution

    * :ref:`contributors-guide`

To contribute to the documentation refer to
`Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

Licensing information can be found on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
