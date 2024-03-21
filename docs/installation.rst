.. meta::
   :description: A high-performance HIP library for tensor primitives
   :keywords: hipTensor, ROCm, library, API, tool, installation

.. _installation:

===============================
Installation
===============================

This document provides instructions for installing and configuring hipTensor library on Linux.
The quickest way to install is to build from source.

-------------
Prerequisites
-------------

-  A ROCm enabled platform, more information `here <https://github.com/ROCm/ROCm>`_.
-  ROCm-cmake, more information `here <https://github.com/RadeonOpenCompute/rocm-cmake/>`_.

---------------------------------
Building and installing hipTensor
---------------------------------

Use the following instructions to build hipTensor from source.

System requirements
^^^^^^^^^^^^^^^^^^^
As a general rule, 8GB of system memory is required for a full hipTensor build. This value can be lower if hipTensor is built without tests. This value may also increase in the future as more functions are added.

GPU support
^^^^^^^^^^^
AMD CDNA class GPU featuring matrix core support: `gfx908`, `gfx90a`, `gfx940`, `gfx941`, `gfx942` and `gfx9`.

.. note:: 
    Double precision FP64 datatype support requires `gfx90a`, `gfx940`, `gfx941` or `gfx942`.

Download hipTensor
^^^^^^^^^^^^^^^^^^

The hipTensor source code is available `here <https://github.com/ROCmSoftwarePlatform/hipTensor>`_. hipTensor requires ROCm version >= 5.7.
To check the ROCm Version on your system, use:

::

    apt show rocm-libs -a

For Centos use

::

    yum info rocm-libs

The ROCm version has major, minor, and patch fields, possibly followed by a build specific identifier. For example, a ROCm version 4.0.0.40000-23 corresponds to major = 4, minor = 0, patch = 0, and build identifier 40000-23.
There are GitHub branches at the hipTensor site with names ``rocm-major.minor.x`` where major and minor are the same as in the ROCm version. To download hipTensor on ROCm version 4.0.0.40000-23, use:

::

   git clone -b release/rocm-rel-x.y https://github.com/ROCmSoftwarePlatform/hipTensor.git
   cd hipTensor

Replace ``x.y`` in the above command with the version of ROCm installed on your machine. For example, if you have ROCm 5.0 installed, then replace release/rocm-rel-x.y with release/rocm-rel-5.0.

You can choose to build any of the following:

* library

* library and samples

* library and tests

You only need (library) for calling hipTensor from your code.
The client contains the tests and sample codes.

Below are the project options available to build hipTensor library with or without clients.

.. list-table::

    *   -   **Option**
        -   **Description**
        -   **Default Value**
    *   -   AMDGPU_TARGETS
        -   Build code for specific GPU target(s)
        -   ``gfx908:xnack-``; ``gfx90a:xnack-``; ``gfx90a:xnack+``; ``gfx940;gfx941;gfx942``
    *   -   HIPTENSOR_BUILD_TESTS
        -   Build Tests
        -   ON
    *   -   HIPTENSOR_BUILD_SAMPLES
        -   Build Samples
        -   ON

Build library
^^^^^^^^^^^^^^^^^^

ROCm-cmake has a minimum version requirement of 0.8.0 for ROCm 5.7.

Minimum ROCm version support is 5.7.

By default, the project is configured in Release mode.

To build the library alone, run:

.. code-block:: bash
    
    `CC=hipcc CXX=hipcc cmake -B<build_dir> . -DHIPTENSOR_BUILD_TESTS=OFF -DHIPTENSOR_BUILD_SAMPLES=OFF`

Here are some other example project configurations:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
|         Configuration             |                                          Command                                                                   |
+===================================+====================================================================================================================+
|            Basic                  |                        :code:`CC=hipcc CXX=hipcc cmake -B<build_dir> .`                                            |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
|        Targeting gfx908           |           :code:`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DAMDGPU_TARGETS=gfx908:xnack-`                          |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
|          Debug build              |                    :code:`CC=hipcc CXX=hipcc cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug`                       |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+

After configuration, build using:

.. code-block:: bash

    cmake --build <build_dir> -- -j

Build library and samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build library and samples, run:

.. code-block:: bash

    CC=hipcc CXX=hipcc cmake -B<build_dir> . -DHIPTENSOR_BUILD_TESTS=OFF -DHIPTENSOR_BUILD_SAMPLES=ON

After configuration, build using:

.. code-block:: bash

    cmake --build <build_dir> -- -j

The samples folder in ``<build_dir>`` contains executables in the table below.

=================================== ===================================================================================
Executable Name                     Description
=================================== ===================================================================================
simple_contraction_bilinear_f32     bilinear contraction using hipTensor API for single-precision floating point types
simple_contraction_scale_f32        scale contraction using hipTensor API for single-precision floating point types
=================================== ===================================================================================


Build library and tests
^^^^^^^^^^^^^^^^^^^^^^^^^

To build library and tests, run:

.. code-block:: bash

    CC=hipcc CXX=hipcc cmake -B<build_dir> .

After configuration, build using:

.. code-block:: bash

    cmake --build <build_dir> -- -j

The tests in ``<build_dir>`` contain executables as given in the table below.

====================================== ===================================================================================
Executable name                        Description
====================================== ===================================================================================
logger_test                            Unit test to validate hipTensor Logger APIs
scale_contraction_f32_test             Scale contraction using hipTensor API for single-precision floating point types
scale_contraction_f64_test             Scale contraction using hipTensor API for double-precision floating point types
bilinear_contraction_f32_test          Bilinear contraction using hipTensor API for single-precision floating point types
bilinear_contraction_f64_test          Bilinear contraction using hipTensor API for double-precision floating point types
====================================== ===================================================================================

Build library and documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build documentation locally, run:

.. code-block:: bash

    cd docs

    sudo apt-get update
    sudo apt-get install doxygen
    sudo apt-get install texlive-latex-base texlive-latex-extra

    pip3 install -r .sphinx/requirements.txt

    python3 -m sphinx -T -E -b latex -d _build/doctrees -D language=en . _build/latex

    cd _build/latex

    pdflatex hiptensor.tex

Running the above commands generates ``hiptensor.pdf``.
