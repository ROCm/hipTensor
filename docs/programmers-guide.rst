.. meta::
   :description: A high-performance HIP library for tensor primitives
   :keywords: hipTensor, ROCm, library, API, tool

.. _programmers-guide:

===================
Programmer's guide
===================

This document provides an insight into the library source code organization and infrastructure.

--------------------------------
Library source code organization
--------------------------------

The hipTensor code is split into four major parts:

- The ``library`` directory contains all source code for the library.
- The ``samples`` directory contains real-world use-cases of the hipTensor API.
- The ``test`` directory contains all validation tests of hipTensor API.
- Infrastructure

The ``library`` directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``library`` directory contains the following include and source files:

- ``library/include/hiptensor/``: C++ include files for the hipTensor API. These files also contain Doxygen comments that document the API.

- ``library/include/hiptensor/internal``: Internal includes files for utility code and generate tensor utility

- ``library/src/``: Source files for Logger, device, and performance functions

- ``library/src/contraction/``: Source files for core composable kernel header functions and contraction initialization functions

- ``library/src/contraction/device``: Source files for hipTensor bilinear and scale instance functions

The ``samples`` directory
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``samples`` directory contains the sample codes for the following use cases:

- ``01_contraction/simple_bilinear_contraction_f32.cpp``: For calling bilinear contraction for ``fp32`` input, output and compute types

- ``01_contraction/simple_scale_contraction_f32.cpp``: For calling scale contraction for ``fp32`` input, output and compute types

The ``test`` directory
^^^^^^^^^^^^^^^^^^^^^^^

The ``test`` directory contains the test codes for testing the following functionalities:

- ``00_unit/logger``: Logger API functions of hipTensor

- ``01_contraction/bilinear_contraction_f32``: Bilinear contraction functionality and log metrics for F32 types

- ``01_contraction/bilinear_contraction_f64``: Bilinear contraction functionality and log metrics for F64 types
 
- ``01_contraction/scale_contraction_f32``: Scale contraction functionality and log metrics for F32 types

- ``01_contraction/scale_contraction_f64``: Scale contraction functionality and log metrics for F64 types

Infrastructure
^^^^^^^^^^^^^^^

- CMake is used to build and package hipTensor. There are ``CMakeLists.txt`` files throughout the code.

- ``Doxygen/Breathe/Sphinx/ReadTheDocs`` are used to produce documentation. The API documentation is generated using:

  - Doxygen comments in include files in the directory ``library/include``
  - files in the directory ``docs/``

- Jenkins is used to automate Continuous Integration (CI) testing.

- ``clang-format`` is used to format C++ code.
