---
layout: page
title: Glossary
permalink: /glossary/
---

## Nvidia GPU Jargon

__CUDA__: Compute Unified Device Architecture.

__Kernel__: a function executed on the GPU.

__CUDA Thread__: a lightweight thread that executes a sequential program. It the smallest unit of
execution for CUDA kernels.

__Block__: a 1D/2D/3D collection of threads.

__Grid__: a 1D/2D/3D collection of grids.

__Warp__: a collection of parallel CUDA threads in a thread block.

__PTX Instruction__: an instruction specified by a CUDA thread.

__SM (Streaming Multiprocessor)__: a multithreaded SIMT/SIMD processor which executes warps of CUDA
threads.

__CUDA Stream__: a sequence of operations that execute in issue-order on the GPU.

__Execution configuration__: the parameters specified within the angle brackets `<<< >>>`. The
execution configuration consists of grid dimension, block dimension, shared memory size (default
size = 0B) and stream (default = null stream).

__SIMT (Single instruction, multiple threads)__: an execution model used in parallel computing where
single instruction, multiple data (SIMD) is combined with multithreading. It is different from SPMD
in that all instructions in all threads are executed in lock-step.

### Memory

__Global memory__: also known as DRAM. It is accessible by all CUDA threads in any block in any
grid. 

__Local memory__: private thread local memory for a CUDA thread. Implemented as a cached region of
DRAM.

__Shared memory__: Fast SRAM memory shared by CUDA threads composing a thread block and private to
that thread block.

__Registers__: private registers for a CUDA thread.

## Computer Architecture

__Amdahl's Law__: Amdahl's law states that the performance improvement obtained by using a faster
mode of execution is limited to the fraction of time the faster mode can be used. 

__Gustafson's Law__: 

Strong scaling is a measure of how, for a fixed overall problem size, the time to solution decreases as more processors are added to a system. An application that exhibits linear strong scaling has a speedup equal to the number of processors used.

Weak scaling is a measure of how the time to solution changes as more processors are added to a system with a fixed problem size per processor; i.e., where the overall problem size increases as the number of processors is increased.


## Miscellaneous

PCIE

NVLink



## References

1. D. Kirk and W. W. Hwu, _Programming Massively Parallel Processors, A Hands on Approach_, Third
   Ed., Morgan Kaufmann 2017.
1. J. Henessey and D. Patterson, _Computer Architecture: A Quantitative Approach_, Sixth Ed., Morgan
   Kaufmann 2019.
