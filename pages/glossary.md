---
layout: page
title: Glossary
permalink: /glossary/
---

## Nvidia GPU Terminology
__Block__: a 1D/2D/3D collection of threads.

__CUDA__: Compute Unified Device Architecture.

__CUDA Core__: a single-precision (fp32) floating-point unit.

__CUDA Stream__: a sequence of operations that execute in issue-order on the GPU.

__CUDA Thread__: a lightweight thread that executes a sequential program. It the smallest unit of

__Device__: an alias for GPU.

__Execution Configuration__: a kernel invocation has to specify the grid dimension and block dimension;
and, optionally, the shared memory size (default size = 0B) and stream (default = null stream).
These parameters are called the execution configuration and are specified within the angle brackets `<<< >>>`.

__Grid__: a 1D/2D/3D collection of grids.

__Host__: alias for CPU.

__Kernel__: a function executed on the GPU. Its arguments are primitive types and pointers, and cannot be larger than 1 KB.

__NVLink__: an interconnection technology used to connect GPUs within a server.
<!--Note that GPUs within a server can also communicate via PCIE they are connected to. -->

__NVSwitch__: a switch used to connect GPUs within a single server. The connection between the switch and device is NVLink.

__PTX Instruction__: an instruction specified by a CUDA thread.

__SIMT (Single instruction, multiple threads)__: an execution model used in parallel computing where
single instruction, multiple data (SIMD) is combined with multithreading. It is different from SPMD
in that all instructions in all threads are executed in lock-step.

__SM (Streaming Multiprocessor)__: a multithreaded SIMT/SIMD processor which executes warps of CUDA
threads.

__Warp__: a collection of parallel CUDA threads in a thread block.

### GPU Memory

__Global memory__: also known as DRAM. It is accessible by all CUDA threads in any block in any
grid.

__Local memory__: private thread local memory for a CUDA thread. Implemented as a cached region of DRAM.
<!-- TODO: slightly oxymoronic since we say DRAM is global, but local memory is part of DRAM. also saying 
local memory - private thread local memory seems self referential. Some online reference to how
"thread-local global memory" is a better term:
A region of DRAM that's accessible only to a specific thread - "thread-local" global memory would be a more accurate term.
-->

__Registers__: private registers for a CUDA thread.
<!-- TODO: they are local to the thread, but are they private? they are visible to the programmer... -->

__Shared memory__: on-chip memory shared by CUDA threads. Since shared memory is on chip, and is built using SRAM, it has higher
bandwidth and lower latency compared to local/global memory.
<!-- basically there's two reasons, proximity and circuit technology: https://stackoverflow.com/questions/28804760/why-shared-memory-is-faster-than-global-memory -->
<!--
Used for communication among CUDA threads in a thread block at barrier synchronization points. Since
shared memory is on chip it has higher bandwidth and lower latency compared to local/global memory.
-->

## Computer Architecture

__Amdahl's Law__: the performance improvement obtained by using a faster
mode of execution is limited to the fraction of time the faster mode can be used.

__Architecture__: the part of a processor that's visible to a programmer - analogous to the API a data structure library presents.

__Arithmetic Intensity__: the ratio of the number of floating point operations executed by a piece of code to the number of bytes of memory it reads and writes.

__Cache consistency__: all views of the data at an address are the same.

__Core__: a processing unit that executes a single instruction stream. A multicore processor consists of a set of cores, each executing its own instruction stream.

__Gustafson's Law__: [Gustafson's law](http://www.johngustafson.net/pubs/pub13/amdahl.htm
) states that as the problem size scales with the number of
processors the maximum speedup $(S)$ a program can achieve is:

$$ S = N + (1-P) (1-N)$$

where $N$ is the number of processors and $P$ is the fraction of the total time spent in serial
execution.
<!-- TODO: Seems like rehash of Amdahl's law, worth including? Is the (1-N) correct? -->

__Microarchitecture__: hardware techniques used to implement an architecture efficiently. E.g., cache, pipeline, branch-predictor, prefetcher, parallel dispatch, etc.

__Memory Controller__: the hardware that reads and writes from DRAM performs DRAM maintainence events (for example, memory refresh)

__Load/Store Architecture__: an architecture where instructions either load/store from registers to RAM or perform operations in registers. This is the most prevalent computer architecture today.

__Instruction Level Parallelism (ILP)__: a microarchitectural innovation by which multiple instructions are executed in parallel.

__PCIE__: the technology used to connect the CPU and devices (GPU, network, hard drive, etc.) - both the interconnect as well as the switch.

<!-- __Pipeline Hazards__: dependencies between instructions that cause the pipeline to lose efficiency (stall, squash, duplicate work). These can be caused by resource conflicts, data dependencies (an instruction's input depends on output of a previous one that hasn't completed), and control (the next instruction depends on the result of the previous one). -->

__Roofline Analysis__: a graphical representation of the performance bounds of a processor in terms of flops and arithmetic intensity.

__Strong Scaling__: strong scaling is a measure of how, for a fixed overall problem size, the time
to solution decreases as more processors are added to a system.
<!-- An application that exhibits linear
strong scaling has a speedup equal to the number of processors used. -->

__Translation Lookaside Buffer (TLB)__: a hardware cache of the page table, i.e., the mapping from virtual to physical addresses.
<!-- This can also be used to implement memory protection, read-only memory, copy-on-write, etc. -->

__Weak Scaling__: weak scaling is a measure of how the time to solution changes as more processors
are added to a system with a fixed problem size per processor; i.e., where the overall problem size
increases as the number of processors is increased.

## Miscellaneous

__Kineto__: the library that traces GPU kernel calls in PyTorch programs - GPU kernels are executed asynchronously on the GPU so special NVIDIA libraries are needed to do the tracing.

__PyTorch Profiler__: the library that uses Kineto to generated host and device-side timeline traces for PyTorch programs.



## References

1. D. Kirk and W. W. Hwu, _Programming Massively Parallel Processors, A Hands on Approach_, Third
   Ed., Morgan Kaufmann 2017.
1. J. Henessey and D. Patterson, _Computer Architecture: A Quantitative Approach_, Sixth Ed., Morgan
   Kaufmann 2019.
