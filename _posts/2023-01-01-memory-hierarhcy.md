---
title: GPU Memory Hierarchy
layout: post
permalink: /memory-hierarchy/
excerpt: An overview of GPU memory hierarchy
---

## Memory diagram 

![GPU Memory Diagram](/images/cuda_memory_model.gif)

__Device code can__
- R/W per-thread registers
- R/W per-thread local memory
- R/W per-block L1 cache/shared memory
- R/W per-grid global memory
- Read only per grid constant memory (on device low latency high bandwidth read only cache which stores constants and kernel arguments)
- Read only per grid texture memory (on device low latency read only cache for 2D/3D textures)

__Host code can__
- Transfer data to/from per grid global, constant and texture memories


## Memory feature and size in A100

| Memory   | Location | Access | Scope | Lifetime | Amount in A100 SXM |
| -------- | -------- |------- | ----- | -------- | ------------------ |
| Register | On chip  | Read/Write | 1 thread | Thread | 256 KB per SM |
| Local\*    | Off chip | Read/Write | 1 thread | Thread | -- |
| Shared\*\* | On chip  | Read/Write | All threads in a block | Block | up to 228 KB per SM |
| Global   | Off chip | Read/Write | All threads + host | Host allocation | 40 GB or 80 GB |
| Constant | Off chip | Read only  | All threads + host | Host allocation | 64 KB |
| Texture  | Off chip | Read only  | All threads + host | Host allocation | Depends on textures used

\* Local memory is not a physical type of memory, but an abstraction of global memory. It is used
only to hold automatic variables. The compiler makes use of local memory when it determines that
there is not enough register space to hold the variable. 

\*\* [Shared
  memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x) is
  configurable up to 228KB per SM, depending on the compute capability.

## Caches

| Type | Access | Amount in A100 SXM |
| ---- | ------ | ------------------ |
| L1 data cache | Read/Write  | 192 KB per SM |
| L2 cache      | Read/Write  | 40 MB         |

