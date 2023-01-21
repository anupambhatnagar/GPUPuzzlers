---
layout: page
title: Answer
permalink: /vector-flops-answer/
---

![Assorted Flops](/vector_flops/assorted_flops.jpg?raw=true "Assorted Flops")

Before the GPU can perform numerical operations can on data, that data has to first be read into
registers. The peak memory bandwidth for an A100 is approximately 2 TB/sec. This means that a $20000
\times 20000$ tensor of FP32 values takes at least $(4 \times 20000^2) /2e12 = 0.8$ ms (to do what?).
Therefore, even if numerical operations were infinitely fast, the time to perform an elementary
operation such as multiplication by a scalar would take at least 1.6ms (read the data, write back
the updated data), effectively achieving 0.25 TFLOPS, i.e., 1.3% of the peak advertised TFOPS.

This is why all the unary operations take roughly the same time - the number of adds and multiplies
is irrelevant. The binary ops take 50% longer, since there's 2 Reads and 1 Write per result.

Matrix multiplication does infact achieve close to advertised performance - 19.1 TFLOPS (needs
clarification). This is because once we read in a value, we operate on it multiple times, thereby
amortizing the cost of the read.


## Discussion

- The ratio of flops performed by an operation to the bytes read/written is known as the compute
  intensity of the operation. 
 - For the unary and binary operations we saw, the flop count is O(n), where n is the number of
   tensor entries; the bytes read/written is also O(n) so the compute intensity is O(1).
 - For matrix multiplication, the flop count is O(n^3), so the compute intensity is O(n^2).
- The advertised memory bandwidth is misleading: it's for best-case memory layouts, specifically
  when copies are aligned with cache dimensions. It assumes wide transfers, i.e., copying long
  contiguous segments of memory - this is true when dealing with tensors, but not for applications
  like sorting.
- Within Meta, the top kernel types are vectorized functors (as we saw above), followed by embedding
  bag lookups, followed by matrix multiplication. ![CUDA Launch Queue
  Microarchitecture](cuda_launch_queue_uarch.jpg?raw=true "CUDA Launch Queue Microarchitecture")
  <!--- from https://slideplayer.com/slide/8211225/ --> <!--- see also
  http://xzt102.github.io/publications/2018_GPGPU_Sooraj.pdf -->

## Key Takeaways



## Futher Reading ??

--- 
## Hint (this should be removed)

[Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law): "the overall performance improvement gained by optimizing a single part of a system is limited by the fraction of time that the improved part is actually used".


