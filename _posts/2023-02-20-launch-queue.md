---
layout: post
title: Order of Kernels
permalink: /order-of-kernels/
excerpt: The order of operations matters on the GPU. Can you find the faster ordering?
tags: [CUDA Launch Queue, PyTorch Dispatch Overhead, CUDA Graphs]
---

In the code snippet below the small_large function does ten 32x32 matrix multiplications, followed
by a 1024x1024 matrix multiplication. The large_small function reverses the order of
multiplications. Which function is faster and why?

``` python
small_matrix = torch.rand((2**5, 2**5), device = torch.device('cuda:0'))
large_matrix = torch.rand((2**10, 2*10), device = torch.device('cuda:0'))

def small_large():
    for _ in range(10):
       torch.matmul(small_matrix, small_matrix)
    torch.matmul(large_matrix, large_matrix)

def large_small():
    torch.matmul(large_matrix, large_matrix)
    for _ in range(10):
       torch.matmul(small_matrix, small_matrix)
```

[See answer and discussion](/order-of-kernels-answer/)
