# PwPA CUDA optimizations
[test_optimizations.cu](test_optimizations.cu) contains all the PwPA CUDA kernels tested with different optimization strategies.

## Strategies
Optimization technics used:
- AoS to SoA re-arrangement of coefficient data structure, to avoid unconcealed global accesses. 
- Unrolling of For-loops.
- Data Reuse with per-thread registers arrays, to hide warp stalls due to pipe math throttlling of FMA executions.
- Re-writing if-branches to avoid thread divergence.
- Shared Memory.

## Evaluation method
To evaluate the kernels w.r.t different metrics [Nvidia Nsight Compute](https://developer.nvidia.com/nsight-compute) is used.

## Results
The results of running [test_optimizations.cu](test_optimizations.cu) are shown below in simple speedup charts.

A simple unrolling is the most effective optimization for this small kernel for both partitions and degree input parameters,
respectively using AoS and SoA coefficients' data structures.

*"AoS Basic Unrolling"* and *"SoA Basic Unrolling"* are the kernels integrated in this extension.

![immagine](https://github.com/user-attachments/assets/ca4d9b43-a8bf-4d0d-995a-7eb71be3eed6)

