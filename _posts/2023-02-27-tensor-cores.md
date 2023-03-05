---
layout: post
title: Quantization Quirks
permalink: /posts/tensor-cores/
excerpt: This puzzle uses tensor cores for matrix multiplcation. Can you find when the matrix multiplication is compute bound and when it is memory bandwidth bound on an A100 (40GB)?
---

[Tensor cores](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)
are high performance programmable matrix multiply and accumulate units. The code snippet below
utilizes tensor cores for some operations and does computation in different precision formats [fp32,
fp16, bf16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)(Brain Floating Point).

1. There are four matrix multiplications in the program. Which ones are compute bound and which are
  memory bandwidth bound on an Nvidia A100(40 GB)?

2. For the given vector operations, let `error_bf16` be the error introduced when using bf16 and
  `error_fp16` be the error introduced when using fp16 - the ratio `error_bf16:error_fp16` is **8:1** -
  why?

``` python
def matmul(A):
    torch.backends.cuda.matmul.allow_tf32 = False
    fp32 = torch.matmul(A, A)

    torch.backends.cuda.matmul.allow_tf32 = True
    tf32 = torch.matmul(A, A)

    A_fp16 = A.half()
    fp16 = torch.matmul(A_fp16, A_fp16)

    A_bf16 = A.to(dtype=torch.bfloat16)
    bf16 = torch.matmul(A_bf16, A_bf16)

def vector_ops(A):
    mul_fp32 = A.mul(0.5)
    sqrt_fp32 = torch.sqrt(A)
    sin_fp32 = torch.sin(A)
    pow_fp32 = torch.pow(A, 3.14)

    A_fp16 = A.half()
    mul_fp16 = A_fp16.mul(0.5)
    sqrt_fp16 = torch.sqrt(A_fp16)
    sin_fp16 = torch.sin(A_fp16)
    pow_fp16 = torch.pow(A_fp16, 3.14)

    A_bf16 = A.to(dtype=torch.bfloat16)
    mul_bf16 = A_bf16.mul(0.5)
    sqrt_bf16 = torch.sqrt(A_bf16)
    sin_bf16 = torch.sin(A_bf16)
    pow_bf16 = torch.pow(A_bf16, 3.14)

    loss = torch.nn.L1Loss(reduction='sum')

    error_fp16_mul = loss(mul_fp32, mul_fp16)
    error_fp16_sqrt = loss(sqrt_fp32, sqrt_fp16)
    error_fp16_sin = loss(sin_fp32, sin_fp16)
    error_fp16_pow = loss(pow_fp32, pow_fp16)

    error_bf16_mul = loss(mul_fp32, mul_bf16)
    error_bf16_sqrt = loss(sqrt_fp32, sqrt_bf16)
    error_bf16_sin = loss(sin_fp32, sin_bf16)
    error_bf16_pow = loss(pow_fp32, pow_bf16)

    print(f"BF16/FP16 relative error \n"
          f"mul: {error_bf16_mul/error_fp16_mul:.2E}\n"
          f"sqrt: {error_bf16_sqrt/error_fp16_sqrt:.2E}\n"
          f"sin: {error_bf16_sin/error_fp16_sin:.2E}\n"
          f"pow: {error_bf16_pow/error_fp16_pow:.2E}")

A = torch.rand((2**9, 2**9), device=torch.device('cuda'), dtype=torch.float32)
matmul(A)
vector_ops(A)
```

[See answer and discussion](/tensor-cores-answer)
