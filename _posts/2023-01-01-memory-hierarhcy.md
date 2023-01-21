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


## Memory access and size in A100

| Memory                   | Access      | Amount in A100 SXM        |
| ------------------------ | ----------- |-------------------------- |
| Global memory            | Read/Write  | 40 GB or 80 GB            |
| L2 cache                 | Read/Write  | 40 MB                     |
| Texture memory           | Read only   | Depends on textures used  |
| Constant memory          | Read only   | 64KB                      |
| L1 data cache (shared memory\*)  | Read/Write  | 192 KB per SM     |
| Shared memory            | Read/Write  | Up to 164 KB per SM       |
| Register memory          | Read/Write  | 256 KB per SM             |

* Shared memory is configurable up to 164KB per SM.
