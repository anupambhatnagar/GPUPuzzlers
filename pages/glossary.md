---
layout: page
title: Glossary
permalink: /glossary/
---

## Nvidia GPU Terminology
__Block__: a 1D/2D/3D collection of threads.

__CUDA__: Compute Unified Device Architecture.

__CUDA Stream__: a sequence of operations that execute in issue-order on the GPU.

__CUDA Thread__: a lightweight thread that executes a sequential program. It the smallest unit of
execution for CUDA kernels.

__Execution configuration__: the parameters specified within the angle brackets `<<< >>>`. The
execution configuration consists of grid dimension, block dimension, shared memory size (default
size = 0B) and stream (default = null stream).

__Grid__: a 1D/2D/3D collection of grids.

__Kernel__: a function executed on the GPU.

__NVLink__:

__NVSwitch__:

__PTX Instruction__: an instruction specified by a CUDA thread.

__SIMT (Single instruction, multiple threads)__: an execution model used in parallel computing where
single instruction, multiple data (SIMD) is combined with multithreading. It is different from SPMD
in that all instructions in all threads are executed in lock-step.

__SM (Streaming Multiprocessor)__: a multithreaded SIMT/SIMD processor which executes warps of CUDA
threads.

__Warp__: a collection of parallel CUDA threads in a thread block.

### Memory

__Global memory__: also known as DRAM. It is accessible by all CUDA threads in any block in any
grid.

__Local memory__: private thread local memory for a CUDA thread. Implemented as a cached region of
DRAM.

__Registers__: private registers for a CUDA thread.

__Shared memory__: Fast SRAM memory shared by CUDA threads composing a thread block and private to
that thread block.
## Computer Architecture

__Amdahl's Law__: Amdahl's law states that the performance improvement obtained by using a faster
mode of execution is limited to the fraction of time the faster mode can be used.

__Architecture__:

__Arithmetic Intensity__: The ratio of floating point operations per byte of memory accessed.

__Cache consistency__:

__Core__:

__Core Level Parallelism (CLP)__:

__Gustafson's Law__: [Gustafson's law](http://www.johngustafson.net/pubs/pub13/amdahl.htm
) states that as the problem size scales with the number of
processors the maximum speedup $(S)$ a program can achieve is:

$$ S = N + (1-P) (1-N)$$

where $N$ is the number of processors and $P$ is the fraction of the total time spent in serial
execution. 

__Microarchitecture__:

__Memory Controller__:

__Load/Store architecture__:

__Instruction Level Parallelism (ILP)__:

__Parallelism__:

__PCIE__:

__Pipeline hazards__:

__Roofline analysis__:

__Strong scaling__: Strong scaling is a measure of how, for a fixed overall problem size, the time
to solution decreases as more processors are added to a system. An application that exhibits linear
strong scaling has a speedup equal to the number of processors used.

__Translation Lookaside Buffer (TLB)__:

__Weak scaling__: Weak scaling is a measure of how the time to solution changes as more processors
are added to a system with a fixed problem size per processor; i.e., where the overall problem size
increases as the number of processors is increased.

## Miscellaneous



## References

1. D. Kirk and W. W. Hwu, _Programming Massively Parallel Processors, A Hands on Approach_, Third
   Ed., Morgan Kaufmann 2017.
1. J. Henessey and D. Patterson, _Computer Architecture: A Quantitative Approach_, Sixth Ed., Morgan
   Kaufmann 2019.
