===============================
Getting Started Guide for Linux
===============================

------------
Introduction
------------

This document contains instructions for installing, using, and contributing to hipTensor.
The quickest way to install is to build from source. The document also contains an API Reference Guide, Programmer's Guide, and Contributor's Guides.

Documentation Roadmap
^^^^^^^^^^^^^^^^^^^^^
The following is a list of hipTensor documents in the suggested reading order:

 - Getting Started Guide (this document): Describes how to install and configure the hipTensor library; designed to get users up and running quickly with the library.
 - API Reference Guide : Provides detailed information about hipTensor functions, data types and other programming constructs.
 - Programmer's Guide: Describes the code organization, Design implementation detail and those that should be considered for new development and Testing & Benchmarking detail.
 - Contributor's Guide : Describes coding guidelines for contributors.

-------------
Prerequisites
-------------

-  A ROCm enabled platform, more information `here <https://rocm.github.io/>`_.
-  ROCm-cmake, more information `here <https://github.com/RadeonOpenCompute/rocm-cmake/>`

---------------------------------
Building and Installing hipTensor
---------------------------------

The following instructions can be used to build hipTensor from source.

System Requirements
^^^^^^^^^^^^^^^^^^^
As a general rule, 8GB of system memory is required for a full hipTensor build. This value can be lower if hipTensor is built without tests. This value may also increase in the future as more functions are added.


GPU Support
^^^^^^^^^^^
AMD CDNA class GPU featuring matrix core support: gfx908, gfx90a as 'gfx9'

`Note: Double precision FP64 datatype support requires gfx90a`

Download hipTensor
^^^^^^^^^^^^^^^^^^

The hipTensor source code is available at the `hipTensor github page <https://github.com/ROCmSoftwarePlatform/hipTensor>`_. hipTensor has a minimum ROCm support version 5.7.
Check the ROCm Version on your system. For Ubuntu use

::

    apt show rocm-libs -a

For Centos use

::

    yum info rocm-libs

The ROCm version has major, minor, and patch fields, possibly followed by a build specific identifier. For example the ROCm version could be 4.0.0.40000-23, this corresponds to major = 4, minor = 0, patch = 0, build identifier 40000-23.
There are GitHub branches at the hipTensor site with names rocm-major.minor.x where major and minor are the same as in the ROCm version. For ROCm version 4.0.0.40000-23 you need to use the following to download hipTensor:

::

   git clone -b release/rocm-rel-x.y https://github.com/ROCmSoftwarePlatform/hipTensor.git
   cd hipTensor

Replace x.y in the above command with the version of ROCm installed on your machine. For example: if you have ROCm 5.0 installed, then replace release/rocm-rel-x.y with release/rocm-rel-5.0

The user can build either

* library

* library + samples

* library + tests

You only need (library) if you call hipTensor from your code.
The client contains the test samples and benchmark code.

Below are the project options available to build hipTensor library with/without clients.

.. tabularcolumns::
   |C|C|C|

+------------------------------+-------------------------------------+-------------------------------------------+
|Option                        |Description                          |Default Value                              |
+==============================+=====================================+===========================================+
|AMDGPU_TARGETS                |Build code for specific GPU target(s)|gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+  |
+------------------------------+-------------------------------------+-------------------------------------------+
|HIPTENSOR_BUILD_TESTS         |Build Tests                          |ON                                         |
+------------------------------+-------------------------------------+-------------------------------------------+
|HIPTENSOR_BUILD_SAMPLES       |Build Samples                        |OFF                                        |
+------------------------------+-------------------------------------+-------------------------------------------+


Build only library
^^^^^^^^^^^^^^^^^^

ROCm-cmake has a minimum version requirement 0.8.0 for ROCm 5.3.

Minimum ROCm version support is 5.4.

By default, the project is configured as Release mode.

To build only library, run the following comomand :

    CC=hipcc CXX=hipcc cmake -B<build_dir> . -DHIPTENSOR_BUILD_TESTS=OFF -DHIPTENSOR_BUILD_SAMPLES=OFF

Here are some other example project configurations:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
|         Configuration             |                                          Command                                                                   |
+===================================+====================================================================================================================+
|            Basic                  |                                CC=hipcc CXX=hipcc cmake -B<build_dir> .                                            |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
|        Targeting gfx908           |                   CC=hipcc CXX=hipcc cmake -B<build_dir> . -DAMDGPU_TARGETS=gfx908:xnack-                          |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
|          Debug build              |                    CC=hipcc CXX=hipcc cmake -B<build_dir> . -DCMAKE_BUILD_TYPE=Debug                               |
+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+

After configuration, build with

    cmake --build <build_dir> -- -j


Build library + samples
^^^^^^^^^^^^^^^^^^^^^^^

To build library and samples, run the following comomand :

    CC=hipcc CXX=hipcc cmake -B<build_dir> . -DHIPTENSOR_BUILD_TESTS=OFF -DHIPTENSOR_BUILD_SAMPLES=ON

After configuration, build with

    cmake --build <build_dir> -- -j

The samples folder in <build_dir> contains executables in the table below.

=================================== ===================================================================================
executable name                     description
=================================== ===================================================================================
test_bilinear_contraction_xdl_fp32  bilinear contraction using hipTensor API for single-precision floating point types
test_scale_contraction_xdl_fp32     scale contraction using hipTensor API for single-precision floating point types
=================================== ===================================================================================


Build library + tests
^^^^^^^^^^^^^^^^^^^^^

To build library and tests, run the following command :

    CC=hipcc CXX=hipcc cmake -B<build_dir> .

After configuration, build with

    cmake --build <build_dir> -- -j

The tests in <build_dir> contains executables in the table below.

====================================== ===================================================================================
executable name                        description
====================================== ===================================================================================
test_bilinear_contraction_xdl_fp32     bilinear contraction using hipTensor API for single-precision floating point types
test_scale_contraction_xdl_fp32        scale contraction using hipTensor API for single-precision floating point types
====================================== ===================================================================================

Build library + Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the steps below to build documentation locally.

    cd docs

    sudo apt-get update
    sudo apt-get install doxygen
    sudo apt-get install texlive-latex-base texlive-latex-extra

    pip3 install -r .sphinx/requirements.txt

    python3 -m sphinx -T -E -b latex -d _build/doctrees -D language=en . _build/latex

    cd _build/latex

    pdflatex hiptensor.tex

Generates hiptensor.pdf here
