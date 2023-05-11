---
layout: page
title: Answer
permalink: /tensor-cores-answer/
---

1. The first matrix multiplication does not use tensor cores and the three remaining ones use tensor
   cores. This can be observed in the trace collected using Nsight Systems (nsys cli requires
   wrapping the code with `torch.cuda.cudart().cudaProfilerStart()` and
   `torch.cuda.cudart().cudaProfilerStop()`) or using the command line utility
   [dcgmi](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/getting-started.html).

   - Let $w$ denote the word size (4 for FP32 and 2 for FP16 and BF16). The arithmetic intensity of
     matrix multiplication is $\frac{2n^3}{3 w n^2} = \frac{2n}{3w}$ as there are approximately
     $2n^3$ operations required to multiply two square matrices of size $n$ and the number of reads
     and writes is $3n^2$.

    <p align = "center">
      <a href="/tensor_cores/arithmetic_intensity.png">
        <img src="/tensor_cores/arithmetic_intensity.png">
      </a>
    </p>

    - The machine balance is computed as the ratio of flops to memory bandwidth. In the roofline
    diagram the machine balance (i.e. the cusp in the roofline). For $n=512$,

    | Numeric format | Word size | Machine balance | Arithmetic intensity (2n/3w)|
    | --- | --- | --- | --- |
    | FP32 | 4  | 19.5/1.55 ~ 12 | ~ 85 |
    | TF32 | 4 | 156/1.55 ~ 100 | ~ 85 |
    | FP16 | 2 | 312/1.55 ~ 201 | ~ 170 |
    | BF16 | 2 | 312/1.55 ~ 201 | ~ 170 |

    Comparing the arithmetic intensity with the machine balance we see that the first matrix
    multiplication is compute bound and the remaining three are memory bandwidth bound.

2. bf16 has 8 bits for the exponent and 7 bits for precision. On the other hand fp16 has 5 bits for
   the exponent and 10 bits for precision. Since A is initialized with random values between 0 and
   1, so the accuracy of bf16 compared to fp16 is 3 bits
   less. As $2^3 = 8$, the accuracy loss is 8 times when using bf16.

## Discussion

**Can CUDA cores and Tensor cores be used concurrently?**

CUDA cores and tensor cores can be utilized simultaneously by distributing the operations across
multiple CUDA streams. A few things to keep in mind which trying to achieve this are:

  1. CUDA_LAUNCH_BLOCKING should be set to `False` to take advantage of concurrent execution.
  1. Kernels used on one stream should not saturate the GPU.
  1. The kernels executing on CUDA cores and tensor cores should run long enough so
     that the launch overhead and synchronization of streams does not increase total time.

**Is there a difference in performance when using fp16 and bf16?**

While the durations when using fp16 and bf16 are comparable for tensor cores but there is a
difference when using vectorized operations.  there is a clear gap is accuracy
when using the two precision formats as the vectorized operations like multiplication, square root,
sine and pow show in the `vector_ops` function.

**When are tensor cores utilized?**

Tensor cores are automatically utilized when doing matrix multiplication but the performance
(flops) achieved by the tensor cores is highly dependent on the matrix sizes and precision format.

| Numeric Format | cuBLAS ≥ 11.0 and cuDNN ≥ 7.6.3                                  |
| ---  | ---                                                                        |
| INT8 | Always but most efficient with multiples of 16; on A100, multiples of 128. |
| FP16 | Always but most efficient with multiples of 8; on A100, multiples of 64.   |
| TF32 | Always but most efficient with multiples of 4; on A100, multiples of 32.   |
| FP64 | Always but most efficient with multiples of 2; on A100, multiples of 16.   |

See
[here](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#improved-tensor-core-operations)
for more details pertaining to different matrix instruction formats.

As an example, consider a transformer based language model built by stacking several encoder/decoder
layers. We summarize below the limitations of using tensor cores for each layer used in the
transformer architecture during the training phase.

| Operation categories      | Uses tensor cores | Usually bound by  |
| ---                       | ---               | ---               |
| Matmul                    | Yes               | Compute           |
| Elementwise (Activation)  | No                | Memory bandwidth  |
| Reduction(Pooling)        | No                | Memory bandwidth  |

**How can good utilization of tensor cores be ensured?**

In order to extract better performance from the GPU, matrix multiplication of large matrices is broken down into smaller tiles as shown below:

<p align = "center">
  <a href="/tensor_cores/tiled_matmul.png">
    <img src="/tensor_cores/tiled_matmul.png">
  </a>
</p>

Some common tile sizes are:

- 256x128 and 128x256 (most efficient)
- 128x128
- 256x64 and 64x256
- 128x64 and 64x128
- 64x64 (least efficient)

When the output matrix dimensions ($M, N$) are not evenly divisible by the tile size or the
number of thread blocks are not evenly divisible by the number of Streaming Multiprocessors (SM)
then there are wasted cycles which lead to inefficiency.

**Tile Quantization**

[Tile
quantization](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#tile-quant) means that the work is quantized to the size of the tile. It happens when the matrix dimensions are not evenly divisible by the thread block size.

**Wave Quantization**

[Wave quantization](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#wave-quant) means that the work is quantized to the size of the GPU. It happens when the number of thread blocks is not evenly divisible by the number of SM's.

## What should you remember in years to come?

Using tensor cores and quantization/mixed precision is not sufficient to get the best performance (flops)
from the GPU. Matrix dimensions play a critical role to achieve high throughput from tensor cores.

## Explore more

1. To see tensor core activity, collect a trace for a PyTorch program using Nsight Systems. Here's a command to get you started:

   ```
   nsys profile --stats true -o report_name --gpu-metrics-device=all python3 file_name.py
   ```

1. Measure tensor core usage using DCGMI:

    ```
    dcgmi dmon -e 1004
    ```
