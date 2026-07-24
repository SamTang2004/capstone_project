# 14x14 MAC Array Prototype

This directory preserves prototypes that are not part of the PYNQ-Z2
deployment bitstream. The active implementation remains the 16x16 tiled
INT8 systolic array with the mixed BRAM plus DSP/LUT resource strategy.

The proposed 14x14 MAC array is retained here as a future exploration for a
different device or resource budget. Before it becomes active work, it needs a
complete interface contract, tiling/padding analysis, DSP and BRAM utilization
estimate, timing target, and golden RTL testbench.
