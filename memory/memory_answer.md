---
layout: page
title: Answer
permalink: /memorable-mysteries-answer/
---

### Puzzler 1

Direct Memory Access (DMA) is a feature by which hardware subsystems (GPU, NIC, drive, etc) can
access host-memory without going through the CPU. Without DMA, we have "CPU bounce" - the CPU has to
read from host-memory and write to the PCIE bus. This slows down the transfer and also keeps it from
doing other work (the extent of the slowdown depends on how fast the CPU is). Therefore it’s
advantageous to use DMA.

Host side memory comes in 2 flavors:
  - pageable
  - pinned (a.k.a. page-locked)

DMA can only be applied to pinned memory.

<p align = "center">
  <a href="/memory/data_transfer.png">
    <img src = "/memory/data_transfer.png">
  </a>
</p>

It’s easy to pin memory in PyTorch. `Acpu = torch.rand(N).pin_memory()` does the trick. With this the
transfer speed almost triples to 12 GB/sec.

### Puzzler 2

GPU memory is allocated and freed by calls to `cudaMalloc()` and `cudaFree()`. cudaMalloc() and
cudaFree() cause device-to-host syncs, which in turn result in the launch queues emptying, exposing
kernel launch overhead. PyTorch uses the CUDA Caching Allocator (CCA) to overcome this. It holds on
to memory allocated by cudaMalloc() and recycles it. Critically, CCA is smart enough to recycle
memory allocated for the single stream case.

Let’s say for iteration $i$, the CCA (running on the CPU) can already decide that (for example)
"Block 1" will be used for Asquare.

Now, for iteration $i+1$, the caching allocator has already run the free call for `Asquare` from
iteration $i$ since the Asquare tensor went out of scope at the end of iteration $i$. This happens
before the next allocation call for `Asquare` in iteration $i+1$.

Here’s the above argument applied to the test program:

```python
for i in range(3000):
  #  Consider iteration 100
  Asquare = torch.matmul(A,A)  # Say 0x6000 is the address of Asquare.
  # Iteration ends
  #    => Asquare goes out of  scope
  #    => Tensor destructor called, has CCA callback
  #    => CCA knows 0x6000 is available
  # Verify recycling by looking at Asquare.data_ptr
```

A big thanks to Andrew Gu and Zachary DeVito for providing clarifications on this material.

### Puzzler 3

One reason for transpose()’s poor performance is the fact that it’s launching kernels in a loop -
looking at the timeline trace, there’s huge gaps, coming from kernel launch overhead, between the
copy kernel calls that perform `result[i,:] = A[:,i]`.

<p align = "center">
  <a href="/memory/puzzler3_trace.png">
    <img src = "/memory/puzzler3_trace.png">
  </a>
</p>

Using [Holistic Trace
Analysis](https://hta.readthedocs.io/en/latest/source/features/temporal_breakdown.html), we see that
the GPU is 90% idle.

<p align = "center">
  <a href="/memory/hta_idle_time.png">
    <img src = "/memory/hta_idle_time.png">
  </a>
</p>

However, even accounting for this, we are left with a factor of 20x perf gap to explain. The reason
for this gap is quite deep.

Recall that an A100 has ~1.5 TB/sec memory bandwidth. Matrix transpose does not perform any
computation - it’s just moving bytes around. We read and then write $4 \cdot 4096 \cdot 4096$ bytes in 173ms.
Discounting the gaps from kernel launch, the naive `transpose()` takes 17 ms, so the effective memory
bandwidth `transpose()` achieved is 8 GB/sec.

Recall peak memory bandwidth comes when we move large (5120 bits) contiguous blocks of memory. This
is definitely not the case with transpose - specifically, there’s lots of fragmentation on the read
side since torch tensors are row major and A[:,i] is a column.

There are several optimizations that can be performed to improve the performance of transpose -
these all have to happen at the level of a CUDA kernel - they are not available from PyTorch. The
two key optimizations are:
1. Memory coalescing - load contiguous blocks from off-chip DRAM into the on-chip SRAM (the shared
   memory (SMEM)), and perform operations in SMEM, which is faster and doesn’t require very wide
   memory accesses
1. Bank conflict minimization - SMEM is made of individual banks of SRAMs, and each bank can support
   only a single read/write per cycle, so it’s important to place data to minimize concurrent
   accesses to a bank.

We give details on both of these in the discussion section.

## Discussion

__Why does DMA require pinned memory?__

Paged memory access requires the ability to translate virtual to physical addresses, which requires
a Memory Management Unit (MMU) which is on the CPU). For DMA to work, we need physical addresses,
since there’s no MMU - this is why the memory must be pinned.
([link](https://spdk.io/doc/memory.html))

__Why does the copy not achieve the full 16 GB/sec we’d expect from PCIE3?__

It’s due to a combination of protocol overhead and contention (the CPU connects to multiple devices
through the same PCIE bus).

__How does this relate to GPUDirect Storage and GPUDirect DMA?__

As previously discussed, CPU bounce is a performance killer when copying from host memory to GPU and
can be avoided using DMA. In the same spirit, copying directly from  storage to GPU and
communicating between GPUs without CPU intervention is beneficial, and this is what GPU Direct
Storage and GPUDirect RDMA provide.

Learn more about [GPUDirect Storage](https://developer.nvidia.com/blog/gpudirect-storage/) and
[GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)

<p align = "center">
  <a href="/memory/gpu_direct_storage.png">
    <img src = "/memory/gpu_direct_storage.png">
  </a>
  GPUDirect Storage
</p>

<br>
<p align = "center">
  <a href="/memory/gpu_direct_rdma.png">
    <img src = "/memory/gpu_direct_rdma.png">
  </a>
  GPUDirect RDMA
</p>

__Where do cudaMalloc() and cudaFree() run?__

Both run on the host - the CUDA context on the host has complete knowledge and control of memory on
the GPU.

__It seems counter-intuitive that memory can be reclaimed before it’s even been used?__

The key is to realize that the CCA serves as a middle layer to orchestrate which GPU memory a
kernel will use when it actually runs. This orchestration can be planned long in advance (in the
case of a fast CPU thread). It doesn’t matter if the GPU has actually performed the square - kernels
are strictly ordered within a stream, so CCA will know that the memory will be free when the ops
that consume the requested allocation execute. Therefore, CCA knows that it can
reuse those blocks. (Operationally, there will be a call in the tensor’s destructor into the
storage implementation, and then into the CCA.)

__What will happen if we were to record the result of each squaring, e.g., by appending each Asquare
to a list?__

CCA will not reclaim storage, and the program will OOM.

__What happens when tensors are used across streams?__

For multiple streams, the block cannot be marked as okay to reuse until the block's cross-stream
usage finishes on the GPU. Tracking this can be challenging and the solution is to rate limit, i.e.,
keep the queue from becoming too big.

__What is memory coalescing?__

Consider a 16x16 matrix as shown below.

<p align = "center">
  <a href="/memory/large_matrix1.png">
    <img src = "/memory/large_matrix1.png">
  </a>
</p>

RAM is one-dimensional, so the actual memory layout of this matrix is shown below. (PyTorch uses
“row-major” order.)

<p align = "center">
  <a href="/memory/row_matrix.png">
    <img src = "/memory/row_matrix.png">
  </a>
</p>

Naive transpose reads entries 0-15, and writes to indices 0, 16, 32, … 244. Since none of these
writes are contiguous, this is slow. (If we wanted to make the writes contiguous, e.g., by
writing result row-by-row, we’d have to read from indices 0, 16, 32, … 244 from the original array.)

The GPU has very fast, though relatively small, on-chip SRAM known as Shared Memory (SM). We can
read in a 4x4 submatrix, e.g., the top-right entries shown in boldface, and transpose them in the
SM, as shown below.

<p align = "center">
  <a href="/memory/submatrices.png">
    <img src = "/memory/submatrices.png">
  </a>
</p>

Now we can write the submatrix to the result. In particular, we only need 4 contiguous writes for
this submatrix, which are coded in color.

<p align = "center">
  <a href="/memory/large_matrix2.png">
    <img src = "/memory/large_matrix2.png">
  </a>
</p>

__What are bank conflicts?__

In reality, shared memory (SMEM) is not a true monolithic RAM - it’s composed of multiple memory “banks” to achieve
high bandwidth. So the picture is closer to the following:

<p align = "center">
  <a href="/memory/memory_banks.png">
    <img src = "/memory/memory_banks.png">
  </a>
</p>

We cannot efficiently generate the first row of the result, since that means we need to perform 4
reads out of B0; nor can we efficiently generate the first column of the result, since that entails
4 writes into B0’.

However we can "stagger" reads and writes to create bank-conflict free schedule. The colors encode
which valuer are written in a specific iteration.

<p align = "center">
  <a href="/memory/resolved_bank_conflicts.png">
    <img src = "/memory/resolved_bank_conflicts.png">
  </a>
</p>

These enhancements are enough to get transpose performance near roofline. This may sound surprising,
since we are doing 2 reads and 2 writes per element. However, one of the reads and one of the writes
is in the SMEM, which is very fast. Furthermore, SMEM read/write latency can be effectively hidden by
overlapping with HBM reads/writes.

#### General Memory Movement Optimization

Optimizing memory access is one of the most challenging aspects of CUDA programming. For example,
when we computed the [arithmetic intensity of square matrix
multiplication](/tensor-cores-answer/), we wrote it as
$\frac{2n^3}{3w \cdot n^2}$, where $n$ is the matrix dimension and $w$ is the word size.

The justification we gave was that to compute $A = B \times C$, we need to read each element of $B$ and $C$, and
write each element of $A$. However, if you look at the expression for the result, you’ll see each
element of $B$ and $C$ is read multiple ($n$) times!

$$
A_{0,0} = B_{0,0} \cdot C_{0,0} + B_{0,1} \cdot C_{1,0} + \ldots + B_{0,n-1} \cdot C_{n-1,0}
$$

$$
A_{0,1} = B_{0,0} \cdot C_{0,1} + B_{0,1} \cdot C_{1,1} + \ldots + B_{0,n-1} \cdot C_{n-1,1}
$$

It’s through clever use of the memory hierarchy (including the coalescing and banking techniques)
that we can get away with $n^2$ reads from $B$ and $C$ respectively. This
[page](/memory-hierarchy/) summarizes the GPU memory hierarchy and here’s
an [example of developing a fast GEMM](https://siboehm.com/articles/22/CUDA-MMM).

#### Transpose in O(1) time

`A.transpose(0,1)` is an $O(1)$ operation, not involving any GPU-side compute - it simply updates
the “strides” in the (host-side) tensor object for $A$. To be precise, a $m$-dimensional tensor $T$
contains a length $m$ tuple of “strides” ($s_1,s_2,\ldots s_m$). The entry $T[i_1] [i_2] \ldots [i_m]$
is at entry $s1 \cdot i_1 + s_2 \cdot i_2 + \dots + s_m \cdot i_m$ in the 1D array that stores $T$’s
entries. For an $n \times n$ matrix $A$, $s_1=n$, and
$s_2=1$. Using $s_1=1$ and $s_2=n$ has the effect of transposing $A$.

### What should you remember years to come?

GPU memory access patterns plays a critical to PyTorch performance. It’s important to have a good
mental model of the GPU memory hierarchy.

### Explore More

- [NVIDIA Blog on Efficient Matrix
  Transpose](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) (memory
  coalescing and removing bank conflicts)
- [NVIDIA Memory Optimizations course](https://courses.nvidia.com/courses/course-v1:DLI+L-AC-02+V1/)
- [Triton PhD thesis](https://dash.harvard.edu/bitstream/handle/1/37368966/ptillet-dissertation-final.pdf?sequence=1&isAllowed=y)
- [Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/index.html)
