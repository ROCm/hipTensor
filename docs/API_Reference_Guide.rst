
************
Introduction
************

hiptensor Data Types
====================

:code:`hiptensorStatus_t`
-----------------

.. doxygenenum::  hiptensorStatus_t

:code:`hiptensorComputeType_t`
----------------------

.. doxygenenum::  hiptensorComputeType_t

:code:`hiptensorOperator_t`
-------------------

.. doxygenenum::  hiptensorOperator_t

:code:`hiptensorAlgo_t`
---------------

.. doxygenenum::  hiptensorAlgo_t

:code:`hiptensorWorksizePreference_t`
-----------------------------

.. doxygenenum::  hiptensorWorksizePreference_t

:code:`hiptensorLogLevel_t`
-------------------------------

.. doxygenenum::  hiptensorLogLevel_t

:code:`hiptensorHandle_t`
-----------------

.. doxygenstruct::  hiptensorHandle_t
   :members:

:code:`hiptensorTensorDescriptor_t`
---------------------------

.. doxygenstruct::   hiptensorTensorDescriptor_t
   :members:

:code:`hiptensorContractionDescriptor_t`
--------------------------------

.. doxygenstruct::  hiptensorContractionDescriptor_t
   :members:

:code:`hiptensorContractionFind_t`
--------------------------

.. doxygenstruct::  hiptensorContractionFind_t
   :members:

:code:`hiptensorContractionPlan_t`
--------------------------

.. doxygenstruct::  hiptensorContractionPlan_t
   :members:

Helper Functions
================

:code:`hiptensorGetVersion`
-------------------

.. doxygenfunction::  hiptensorGetVersion

:code:`hiptensorCreate`
---------------

.. doxygenfunction::  hiptensorCreate

:code:`hiptensorDestroy`
----------------

.. doxygenfunction::  hiptensorDestroy

:code:`hiptensorInitTensorDescriptor`
-----------------------------

.. doxygenfunction::  hiptensorInitTensorDescriptor

:code:`hiptensorGetAlignmentRequirement`
--------------------------------

.. doxygenfunction::  hiptensorGetAlignmentRequirement

:code:`hiptensorGetErrorString`
-----------------------

.. doxygenfunction::  hiptensorGetErrorString

Contraction Operations
======================

:code:`hiptensorInitContractionDescriptor`
----------------------------------

.. doxygenfunction::  hiptensorInitContractionDescriptor

:code:`hiptensorInitContractionFind`
----------------------------

.. doxygenfunction::  hiptensorInitContractionFind

:code:`hiptensorInitContractionPlan`
----------------------------

.. doxygenfunction::  hiptensorInitContractionPlan

:code:`hiptensorContraction`
--------------------

.. doxygenfunction::  hiptensorContraction

:code:`hiptensorContractionGetWorkspaceSize`
------------------------------------

.. doxygenfunction::  hiptensorContractionGetWorkspaceSize

Logging Functions
=================

:code:`hiptensorLoggerSetCallback`
--------------------------

.. doxygenfunction::  hiptensorLoggerSetCallback

:code:`hiptensorLoggerSetFile`
----------------------

.. doxygenfunction::  hiptensorLoggerSetFile

:code:`hiptensorLoggerOpenFile`
-----------------------

.. doxygenfunction::  hiptensorLoggerOpenFile

:code:`hiptensorLoggerSetLevel`
-----------------------

.. doxygenfunction::  hiptensorLoggerSetLevel

:code:`hiptensorLoggerSetMask`
----------------------

.. doxygenfunction::  hiptensorLoggerSetMask

:code:`hiptensorLoggerForceDisable`
---------------------------

.. doxygenfunction::  hiptensorLoggerForceDisable
