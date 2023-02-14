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
import torch
from torch.profiler import profile, ProfilerActivity

def do_work():
    SIZE = 2**12
    ones = torch.ones((SIZE, SIZE), device=torch.device('cuda'))

    torch.matmul(ones, ones, out=ones)

    # In place multiplication
    ones.mul_(0.5)

    result = ones.mul(0.5)

    total = ones + result

    result = torch.sum(ones)

    # sqrt takes 7 ops.
    result = torch.sqrt(ones)

    # sin takes 17 ops (14 fp64, 3 fp32).
    result = torch.sin(ones)

    # sigmoid takes 24 ops.
    result = torch.sigmoid(ones)

    # log10 takes 24 ops.
    result = torch.log10(ones)

    # pow takes 142 ops.
    result = torch.pow(ones, 3.14159)

do_work()

with profile(
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory = True,
    record_shapes = True
    #with_stack = True
) as prof:

    do_work()

filename = f"./flops.json"
prof.export_chrome_trace(filename)
```

The trace shown below indicates that other than matrix multiplication, all of the other operations
are a tiny fraction (~1%) of the advertised flops. Furthermore, simple operations like addition take
exactly as long as complex ones like sine and log. Why?

<a href = "/vector_flops/assorted_flops.jpg">
  <img src= "/vector_flops/assorted_flops.jpg" text="vector flops trace">
</a>

[See answer and discussion](/vector-flops-answer/)
