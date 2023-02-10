---
layout: page
title: Answer
permalink: /vector-flops-answer/
---

![Assorted Flops](/vector_flops/assorted_flops.jpg?raw=true "Assorted Flops")

### Solution

Some things that stand out:
 - Summing elements of a matrix takes roughly half as long as multiplying a scalar into a matrix 
 - sqrt, sin, sigmoid, log10, pow all roughly take as long as scalar multiplication 
 - Adding two matrices takes roughly 50% longer than scalar multiplcation

What's the difference between these three classes of operations?
  - Summation, and multiplication by a scalar: in the former we only read entries (the sum is "accumulated" in local registers), in the latter we read read and write back entries. 
  - Adding two matrices requires an additional read.

This suggests the bottleneck is reading memory - before the GPU can perform numerical operations can on data, that data has to first be read into registers. 
  - The peak memory bandwidth for a A100, 40GB is 1.555 TB/sec. 
  - This means that an 8192x8192 matrix of fp32 values takes at least (4x8192^2)/1.555e12 = 0.17 ms. 
  - Therefore, even if numerical operations were infinitely fast, the time to multiply by a scalar is at least 0.34 ms (read the data, write back the updated data)
    - This yields 0.197 TFLOPs, i.e., 1% of the peak.

From some clever CPU-side profiling, we can get good estimates of the number of flops needed to perform sqrt (7), sin (3 single precision, 14 double precision), sigmoid (24), log10 (24), pow (142). For all of these, excepting pow, we are strongly dominated by memory bandwidth.

Matrix multiplication achieves close to advertised performance. This is because once we read in a value, we operate on it multiple times, amortizing the cost of the read.


### Discussion

- The ratio of flops performed by an operation to the bytes read/written is known as the **arithmetic intensity** of the operation. 
  - For the unary and binary operations we saw, the flop count is O(n), where n is the number of tensor entries; the bytes read/written is also O(n) so the arithmetic intensity is O(1).
  - For matrix multiplication, the flop count is O(n^3), so the arithmetic intensity is O(n^2).
  - An A100 40GB can perform roughly 50 flops in the time it takes to read or write 4 bytes (a single precision float) - therefore, if the arithmetic intensity of a program is under 50, on an A100 40 GB it will be limited by memory bandwidth and not flops.
- The maximum clock speed of an A100 is 1410 MHz - dividing the advertised flops by the clock speed, we get **13830 FLOPs/cycle**.
  - Looking at the spec sheet, we see it has 6912 "CUDA Cores" - this number is suspiciously close to 13830/2 = 6915.
  - A CUDA Core is essentially a single-precision floating point unit. 
  - Multiplying two n-bit numbers entails adding n-1 partial products. Therefore a MAC (multiply-and-accumulate) operation has very little incremental cost over multiplication.
  - All hardware vendors consider a MAC to be two flops - so 6912 cores can perform 13824 FLOPs/cycle.
- Even for compute-bound workloads achieving advertised performance is incredibly hard - **every CUDA Core has to perform a MAC** every single cycle. 
- The **advertised memory bandwidth** is misleading: 
  - It's for best-case memory layouts, specifically when copies are aligned with cache dimensions. 
  - GPUs get their enormous memory bandwidth by having a very wide memory bus (5192 bits, 10+X that of a CPU). This works well when copying long contiguous segments of memory, i.e., operating on tensors, but not for applications like sorting that do random access at the word level.
  - We see this in PyTorch's torch.sort(): it has abysmal performance compared to CPU.
  - Embedding table lookup is a common operation in recommendation models, and it requires sorting the indices. This is achieved via a custom radix-sort, that has better locality of reference.
