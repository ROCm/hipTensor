Use the DeviceMem structure under the new wrapper /member of the existing hiptensorTensorDescriptor\_t for moving the data from the input arrays for the core operations.
    --- Don't need to use the host tensors for this computation

Need to port all the contraction logic this location.


Write the overloading operator (<<) for the structures:
hiptensorTensorDescriptor\_t  -- printing the lengths and strides
hiptensorContractionPlan\_t  -- printing the metrics (avg\_time, flops, gb\_per\_sec)
