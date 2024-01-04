===================
Programmer's Guide
===================

--------------------------------
Library Source Code Organization
--------------------------------

The hipTensor code is split into four major parts:

- The `library` directory contains all source code for the library.
- The `samples` directory contains real-world use-cases of the hipTensor API.
- The `test` directory contains all validation tests of hipTensor API.
- Infrastructure

The `library` directory
^^^^^^^^^^^^^^^^^^^^^^^

`library/include/hiptensor/`
''''''''''''''''''''''''''''

Contains C++ include files for the hipTensor API. These files also contain Doxygen
comments that document the API.

`library/include/hiptensor/internal`
''''''''''''''''''''''''''''''''''''

Internal include files for:

- Utility Code
- Generate Tensor Utility

`library/src/`
''''''''''''''

Contains logger, device and performance functions.

`library/src/contraction/`
''''''''''''''''''''''''''

Contains hipTensor core composable kernel header functions and contraction initialization functions.

`library/src/contraction/device`
''''''''''''''''''''''''''''''''

Contains hipTensor Bilinear and Scale instance functions

The `samples` directory
^^^^^^^^^^^^^^^^^^^^^^^
`01_contraction/simple_bilinear_contraction_f32.cpp`
''''''''''''''''''''''''''''''''''''''''''''''''''''

sample code for calling bilinear contraction for :code:`fp32` input, output and compute types


`01_contraction/simple_scale_contraction_f32.cpp`
'''''''''''''''''''''''''''''''''''''''''''''''''

sample code for calling scale contraction for :code:`fp32` input, output and compute types

The `test` directory
^^^^^^^^^^^^^^^^^^^^^^^

`00_unit/logger`
''''''''''''''''

Test code for testing logger API Functions of hipTensor

`01_contraction/bilinear_contraction_f32`
'''''''''''''''''''''''''''''''''''''''''

Test code for testing the bilinear contraction functionality and log metrics for F32 types.

`01_contraction/bilinear_contraction_f64`
'''''''''''''''''''''''''''''''''''''''''

Test code for testing the bilinear contraction functionality and log metrics for F64 types.

`01_contraction/scale_contraction_f32`
''''''''''''''''''''''''''''''''''''''

Test code for testing the scale contraction functionality and log metrics for F32 types.

`01_contraction/scale_contraction_f64`
''''''''''''''''''''''''''''''''''''''

Test code for testing the scale contraction functionality and log metrics for F64 types.

Infrastructure
^^^^^^^^^^^^^^

- CMake is used to build and package hipTensor. There are :code:`CMakeLists.txt` files throughout the code.
- `Doxygen/Breathe/Sphinx/ReadtheDocs` are used to produce documentation. Content for the documentation is from:

  - Doxygen comments in include files in the directory :code:`library/include`
  - files in the directory :code:`docs/`

- Jenkins is used to automate Continuous Integration testing.
- :code:`clang-format` is used to format C++ code.
