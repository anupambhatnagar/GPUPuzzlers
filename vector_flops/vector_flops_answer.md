---
layout: page
title: Answer
permalink: /vector-flops-answer/
---

![Assorted Flops](/vector_flops/assorted_flops.jpg?raw=true "Assorted Flops")

### Solution

In the given program, operations other than matmul are bottlenecked on memory bandwidth - how fast we can read and write from DRAM - and not compute.

When root-causing the performance bottleneck, here's some things that stand out:
 - Summing matrix elements takes roughly half as long as multiplying a scalar into the matrix.
 - The operations `sqrt`, `sin`, `sigmoid`, `log10`, `pow` take roughly the same time as scalar multiplication.
 - Adding two matrices takes roughly 50% longer than scalar multiplication.

What's the difference between these operations?
  1. *Summation vs scalar multiplication:* for summation we read entries only once from DRAM (the sum is "accumulated" in local registers), for multiplication, we read and write back entries. 
  1. *Scalar multiplication vs adding matrices*: adding matrices requires an additional read.

Before the GPU can perform numerical operations on data, that data has to first be read into registers.
  - The peak memory bandwidth for a A100 (40GB) is 1.555 TB/sec. 
    - This means that an 8192x8192 matrix of fp32 values takes at least 4x(8192)^2/1.555e12 = 0.17 ms to read. 
    - => Even if numerical operations took no time, scalar multiplication takes at least 0.34 ms (read, write back)
    - => Scalar multiplcation achieves 0.197 TFLOPs - 1% of the peak.

From CPU-side profiling, we can estimate the number of flops needed to perform math operations:
  - `sqrt` (7), `sin` (3 single precision, 14 double precision), `sigmoid` (24), `log10` (24), `pow` (142). 
  - For all of these, excepting `pow`, we are strongly dominated by memory bandwidth.

Matrix multiplication achieves 19.5 TFLOPS/s. 
   - Once we read in a value, we operate on it multiple times, amortizing the cost of the read.

### Discussion

#### Arithmetic Intensity
- The ratio of flops performed by an operation to the bytes read/written is known as the **arithmetic intensity** of the operation. 
  - For the point-wise operations, the flop count is O(n), where n is the number of tensor entries; the bytes read/written is also O(n) so the arithmetic intensity is O(1).
  - For matrix multiplication, the flop count is O(n^3), so the arithmetic intensity is O(n^2).
  - An A100 (40GB) can perform roughly 50 flops in the time it takes to read or write 4 bytes (single precision float) 
    - Arithmetic intensity of a program < 50 => on an A100 (40 GB) it will be limited by memory bandwidth and not flops.
#### Understanding 19.5 TFLOPS/sec
- The maximum clock speed of an A100 is 1410 MHz - dividing the advertised flops by the clock speed, we get **13830 FLOPs/cycle**.
  - Looking at the spec sheet, we see it has 6912 "CUDA Cores" - this number is suspiciously close to 13830/2 = 6915.
  - A CUDA Core is essentially one single-precision floating point unit. 
  - Multiplying two *n*-bit numbers entails adding *n-1* partial products. Therefore a MAC (multiply-and-accumulate) operation has very little incremental cost over multiplication.
  - All hardware vendors consider a MAC to be two flops - so 6912 cores can perform 13824 FLOPs/cycle.
- Even for compute-bound workloads achieving peak flops is hard - **every CUDA Core has to perform a MAC every single cycle**. 
#### Achieving Peak Memory Bandwidth
- Just like TFLOPs, the **memory bandwidth** on GPU spec sheets can easily be misunderstood:
  - Best-case memory layouts, specifically when copies are aligned with cache dimensions. 
  - GPUs memory bandwidth comes from very wide memory bus (5120 bits, 10X that of a CPU). 
  - Works well when reading/writing long contiguous segments of memory.
  - Tensors are ideal use-case, but still have challenges.
    - Consider 2D matrix transpose: column-major order -> writing to rows is fragmented
    - Solution: clever memory access (blocking, coalescing, banking)

### What will you remember in 10 years?

Though we focus on compute GPUs have additional performance constraints.
 - Arithmetic Intensity - the ratio of flops to bytes read/writtten - is the key to determining if a GPU program is compute- or memory-bandwidth bound.

### Explore more

#### How do you compute the flops needed by a program?

Various approaches: analytical, NVIDIA tooling (coarse grain, fine grain)
  - Slick approach: get from CPU side using `perf stat`
