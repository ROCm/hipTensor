.. meta::
   :description: A high-performance HIP library for tensor primitives
   :keywords: hipTensor, ROCm, library, API, tool

.. _api-reference:

********************
API reference guide
********************

hipTensor data types
====================

.. <!-- spellcheck-disable -->

hiptensorStatus_t
-----------------

.. doxygenenum::  hiptensorStatus_t

hiptensorComputeType_t
----------------------

.. doxygenenum::  hiptensorComputeType_t

hiptensorOperator_t
-------------------

.. doxygenenum::  hiptensorOperator_t

hiptensorAlgo_t
---------------

.. doxygenenum::  hiptensorAlgo_t

hiptensorWorksizePreference_t
-----------------------------

.. doxygenenum::  hiptensorWorksizePreference_t

hiptensorLogLevel_t
-------------------------------

.. doxygenenum::  hiptensorLogLevel_t

hiptensorHandle_t
-----------------

.. doxygenstruct::  hiptensorHandle_t
   :members:

hiptensorTensorDescriptor_t
---------------------------

.. doxygenstruct::   hiptensorTensorDescriptor_t
   :members:

hiptensorContractionDescriptor_t
--------------------------------

.. doxygenstruct::  hiptensorContractionDescriptor_t
   :members:

hiptensorContractionFind_t
--------------------------

.. doxygenstruct::  hiptensorContractionFind_t
   :members:

hiptensorContractionPlan_t
--------------------------

.. doxygenstruct::  hiptensorContractionPlan_t
   :members:

Helper functions
================

hiptensorGetVersion
-------------------

.. doxygenfunction::  hiptensorGetVersion

hiptensorCreate
---------------

.. doxygenfunction::  hiptensorCreate

hiptensorDestroy
----------------

.. doxygenfunction::  hiptensorDestroy

hiptensorInitTensorDescriptor
-----------------------------

.. doxygenfunction::  hiptensorInitTensorDescriptor

hiptensorGetAlignmentRequirement
--------------------------------

.. doxygenfunction::  hiptensorGetAlignmentRequirement

hiptensorGetErrorString
-----------------------

.. doxygenfunction::  hiptensorGetErrorString

Contraction operations
======================

hiptensorInitContractionDescriptor
----------------------------------

.. doxygenfunction::  hiptensorInitContractionDescriptor

hiptensorInitContractionFind
----------------------------

.. doxygenfunction::  hiptensorInitContractionFind

hiptensorInitContractionPlan
----------------------------

.. doxygenfunction::  hiptensorInitContractionPlan

hiptensorContraction
--------------------

.. doxygenfunction::  hiptensorContraction

hiptensorContractionGetWorkspaceSize
------------------------------------

.. doxygenfunction::  hiptensorContractionGetWorkspaceSize

Logging functions
=================

hiptensorLoggerSetCallback
--------------------------

.. doxygenfunction::  hiptensorLoggerSetCallback

hiptensorLoggerSetFile
----------------------

.. doxygenfunction::  hiptensorLoggerSetFile

hiptensorLoggerOpenFile
-----------------------

.. doxygenfunction::  hiptensorLoggerOpenFile

hiptensorLoggerSetLevel
-----------------------

.. doxygenfunction::  hiptensorLoggerSetLevel

hiptensorLoggerSetMask
----------------------

.. doxygenfunction::  hiptensorLoggerSetMask

hiptensorLoggerForceDisable
---------------------------

.. doxygenfunction::  hiptensorLoggerForceDisable

.. <!-- spellcheck-enable -->
