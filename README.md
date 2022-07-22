## Introduction:
hipTENSOR is a high-performance HIP library for tensor primitives based on the composable kernels, which is aa set of C++ templates that provide the ability to generate high-performance assembly kernels for mathematical operations.

## Limitations:
* The package supports only the FP32 inputs with FP32 Compute.
* Also, the inputs with the tensor suppports only the following input tensors layout assumption of the tensor contraction operation.
  - Input:  A[M0, M1, M2, ..., K0, K1, K2, ...], B[N0, N1, N2, ..., K0, K1, K2, ...]
  - Output: C[M0, M1, M2, ..., N0, N1, N2, ...]

## Features to be adapted in future development: <br>
  - Adapt the library to adapt the arbitrary input tensor layouts and type computes.
  - Adapt the hiptensorContractionFind API to different set of available algorithms on the different accelerators.
  - Adapt the hiptensorContractionGetWorkspace API as per the future backends.
  - Also, to adapt the library to handle for the FP64 tensors support. <br>
    (Pending due to compiler issue:  https://ontrack-internal.amd.com/browse/SWDEV-335738).
  - Need to make few modularisation in the ck part of the core logic handling all the datatypes.

