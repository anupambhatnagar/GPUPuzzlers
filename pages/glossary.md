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
<!-- TODO: feels a bit out of nowhere? -->

__Grid__: a 1D/2D/3D collection of grids.

__Kernel__: a function executed on the GPU. Its arguments are primitive types and pointers, and cannot be larger than 1KB.  
<!-- TODO: elaborate a bit more on significance of a kernel -->

__NVLink__: An interconnection technology used to connect GPUs to one-another within a server. Note that GPUs within a server can also communicate via the PCIE the are connected to.

__NVSwitch__: A switch used to connect GPUs within a single server node - the corresponding interconnect is NVLink.

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

__Local memory__: private thread local memory for a CUDA thread. Implemented as a cached region of
DRAM. 
<!-- TODO: a bit misleading, since local memory is actually in SRAM? -->

__Registers__: private registers for a CUDA thread.

__Shared memory__: Fast SRAM memory shared by CUDA threads composing a thread block and private to
that thread block.

## Computer Architecture

__Amdahl's Law__: The performance improvement obtained by using a faster
mode of execution is limited to the fraction of time the faster mode can be used.

__Architecture__: The part of a processor that's visible to a programmer - analagous to the API a data structure library presents.

__Arithmetic Intensity__: The ratio of the number of floating point operations executed by a piece of code to the number of byte of memory accessed it reads.

__Cache consistency__: All views of the data at an address are the same.

__Core__: A processing unit that executes a single instruction stream, as opposed to a multicore processor which consists of a set of cores, each executing its own instruction stream.

__Core Level Parallelism (CLP)__: 
<!-- TODO: what is this? -->

__Gustafson's Law__: [Gustafson's law](http://www.johngustafson.net/pubs/pub13/amdahl.htm
) states that as the problem size scales with the number of
processors the maximum speedup $(S)$ a program can achieve is:

$$ S = N + (1-P) (1-N)$$

where $N$ is the number of processors and $P$ is the fraction of the total time spent in serial
execution. 
<!-- TODO: Seems like rehash of Amdahl's law, worth including? Is the (1-N) correct? -->

__Microarchitecture__: The collection of techniques used to implement an architecture efficiently and performantly: e.g., caches, pipeline, branch-predictor, prefetcher, parallel dispatch, etc.

__Memory Controller__:  The logic needed to read and write from DRAM as well as perform DRAM maintainence events (refresh as well as more obscure actions).

__Load/Store architecture__: An architecture where instructions either load/store from registers to RAM or operate on registers. This is the most prevalent computer architecture today.

__Instruction Level Parallelism (ILP)__: A microarchitectural innovation by which multiple instructions are executed in parallel. The key challenges are dependencies between instructions, resource contention, and branches. Instruction reordering, hardware schedulers, and stalling/speculation + squashing/predication are used to addess these.

__Parallelism__: <!-- TODO - too general? -->

__PCIE__: The technology used to connect the CPU and devices (GPU, network, hard drive, etc.) - both the interconnect as well as the switch.

__Pipeline hazards__: Dependencies between instructions that cause the pipeline to lose efficiency (stall, squash, duplicate work). These can be caused by resource conflicts, data dependencies (an instruction's input depends on output of a previous one that hasn't completed), and control (the next instruction depends on the result of the previous one).

__Roofline analysis__: A graphical representation of the performance bound of some hardware, usually peak flops and memory bandwidth.

__Strong scaling__: Strong scaling is a measure of how, for a fixed overall problem size, the time
to solution decreases as more processors are added to a system. An application that exhibits linear
strong scaling has a speedup equal to the number of processors used.

__Translation Lookaside Buffer (TLB)__: A hardware cache of the page table, i.e., the mapping from virtual to physical addresses. This can also be used to implement memory protection, read-only memory, copy-on-write, etc.

__Weak scaling__: Weak scaling is a measure of how the time to solution changes as more processors
are added to a system with a fixed problem size per processor; i.e., where the overall problem size
increases as the number of processors is increased.

## Miscellaneous

__Kineto__: The library that enables tracing GPU kernel calls in PyTorch prgrams - since by default GPU kernels are executed asynchronously on the GPU, special NVIDIA libraries are needed to do the tracing. 

__PyTorch Profiler__: The library that enables host-side tracing of PyTorch programs - it is integrated with Kineto to generate host + GPU side timeline traces.



## References

1. D. Kirk and W. W. Hwu, _Programming Massively Parallel Processors, A Hands on Approach_, Third
   Ed., Morgan Kaufmann 2017.
1. J. Henessey and D. Patterson, _Computer Architecture: A Quantitative Approach_, Sixth Ed., Morgan
   thethe  Kaufmann 2019.
