
************
Introduction
************

hiptensor Data Types
====================

hiptensorStatus_t
-----------------

.. doxygenenum::  hiptensorStatus_t

hiptensorDataType_t
-------------------

.. doxygenenum::  hiptensorDataType_t

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

hiptensorContractionOperation_t
-------------------------------

.. doxygenenum::  hiptesnorContractionOperation_t

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

tensor_attr
-----------

.. doxygenstruct::  tensor_attr
   :members:

hiptensorContractionDescriptor_t
--------------------------------

.. doxygenstruct::  hiptensorContractionDescriptor_t
   :members:

hiptensorContractionFind_t
--------------------------

.. doxygenstruct::  hiptensorContractionFind_t
   :members:

hiptensorContractionMetrics_t
-----------------------------

.. doxygenstruct::  hiptensorContractionMetrics_t
   :members:

hiptensorContractionPlan_t
--------------------------

.. doxygenstruct::  hiptensorContractionPlan_t
   :members:

Helper Functions
================

hiptensorGetVersion
-------------------

.. doxygenfunction::  hiptensorGetVersion

hiptensorInit
-------------

.. doxygenfunction::  hiptensorInit

hiptensorInitTensorDescriptor
-----------------------------

.. doxygenfunction::  hiptensorInitTensorDescriptor

hiptensorGetAlignmentRequirement
--------------------------------

.. doxygenfunction::  hiptensorGetAlignmentRequirement

hiptensorGetErrorString
-----------------------

.. doxygenfunction::  hiptensorGetErrorString

Contraction Operations
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

hipTensorContractionGetWorkspaceSize
------------------------------------

.. doxygenfunction::  hipTensorContractionGetWorkspaceSize

Logging Functions
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
