Overview
========

.. |br| raw:: html

   <br />

| **Advanced Micro Devices, Inc's tensor contraction library.**

* Sources and binaries can be found at `hipTENSOR GitHub repo <https://github.com/AMD-HPC/hipTENSOR>`_.
* Backend package can be found at `composable_kernel GitHub repo <https://github.com/ROCmSoftwarePlatform/composable_kernel>`_.
* This package implement a first tensor contraction using hipTENSOR. Our code will compute the following operation using single-precision arithmetic.
  
.. math::
   
   D_{M0,M1,N0,N1} = \alpha A_{M0,M1,K0,K1}B_{N0,N1,K0,K1} + \beta C_{M0,M1,N0,N1}


Introduction
------------

hipTENSOR is a high-performance HIP library for tensor primitives based on the composable kernels, which is a set of C++ templates that provide the ability to generate high-performanceassembly kernels for mathematical operations.

Limitations
-----------

* The backend (composable\_kernel) is tested only with the rocm-5.1 and the 9110 version of compiler.
* The package supports only the FP32 inputs with FP32 Compute.
* Also, the inputs with the tensor suppports only the following input tensors layout assumption of the tensor contraction operation. 
  |br| Input : A[M0, M1, M2, ..., K0, K1, K2, ...], B[N0, N1, N2, ..., K0, K1, K2, ...].
  |br| Output: C[M0, M1, M2, ..., N0, N1, N2, ...].

Future development
------------------
  
* Adapt the library to adapt the arbitrary input tensor layouts and type computes.
* Adapt the hiptensorContractionFind API to different set of available algorithms on the different accelerators.
* Adapt the hiptensorContractionGetWorkspace API as per the future backends.
* Also, to adapt the library to handle for the FP64 tensors support.
  |br| Pending due to compiler issue: `SWDEV-335738 <https://ontrack-internal.amd.com/browse/SWDEV-335738>`_.
* Need to make few modularisation in the ck part of the core logic handling all the datatypes.
