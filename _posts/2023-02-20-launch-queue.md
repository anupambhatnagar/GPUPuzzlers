---
layout: post
title: Order of Kernels
permalink: /order-of-kernels/
excerpt: The order of operations matters on the GPU. Can you find the faster ordering?
---

In the code snippet below the `small_large` function squares a 32x32 matrix ten times, followed by
squaring a 1024x1024 matrix. The `large_small` function does the matrix multiplications in the
reverse order i.e. square the large matrix followed by squaring the small matrix ten times. Which
function has a smaller execution time and why?

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
