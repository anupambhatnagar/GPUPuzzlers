---
layout: post
title: Counting TFLOPS
excerpt: Is the GPU faster at addition or multiplication?
---

Marketing literature for GPUs stresses their high FLOPS. An [A100 is advertised as capable of
achieving 19.5 FP32 TFLOPS](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/).
The code below performs a number of numerical operations on tensors: addition, scalar
multiplication, transcendental functions, and matrix multiplications etc.

``` python
import time
import torch

def sync_and_pause():
torch.cuda.synchronize()
  time.sleep(1e-6)

size = 2^15
ones = torch.ones((size, size), device=torch.device('cuda'))

ones.mul_(0.5)
sync_and_pause()

result = ones.mul(0.5)
sync_and_pause()

result += ones
sync_and_pause()

total = ones + result
sync_and_pause()

result = torch.sin(ones)
sync_and_pause()

result = torch.sigmoid(ones)
sync_and_pause()

torch.sqrt(ones, out=result)
sync_and_pause()

torch.log10(ones, out=result)
sync_and_pause()

torch.pow(ones, 3.14159, out=result)
sync_and_pause()

result = torch.matmul(ones, ones)
sync_and_pause()
```

The trace shown below indicates that other than matrix multiplication, all of the other operations
are a tiny fraction (~1%) of the advertised flops. Furthermore, simple operations like addition take
exactly as long as complex ones like sine and log. Why?

![trace](/vector_flops/assorted_flops.jpg?raw=true)

[See answer and discussion](/vector-flops-answer/)
