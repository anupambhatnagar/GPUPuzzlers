---
layout: post
title: Swimming in Streams
permalink: /posts/swimming-in-streams/
excerpt: Concurrent kernel execution with CUDA streams
---

A CUDA stream is a sequence of commands (possibly issued by different host threads) that execute in
order. Applications can manage concurrent execution of kernels through multiple streams.

This puzzler executes kernels and does data transfer concurrently using multiple streams. There are two
different implementations provided. Which implementation achieves better overlap and why? The
attached trace contains the answer. Check if it matches your intuition.

```python
def streams1():
    with torch.cuda.stream(s1):
        for i in range(len(matrix_on_gpu)):
            torch.matmul(matrix_on_gpu[i], matrix_on_gpu[i])

    with torch.cuda.stream(s2):
        for i in range(len(data_on_gpu)):
            data_on_gpu[i].to(cpu, non_blocking=True)

    with torch.cuda.stream(s3):
        for i in range(len(data_on_cpu)):
            data_on_cpu[i].to(cuda, non_blocking=True)

def streams2():
    with torch.cuda.stream(s1):
        for i in range(len(matrix_on_gpu)):
            torch.matmul(matrix_on_gpu[i], matrix_on_gpu[i])

    with torch.cuda.stream(s2):
        for i in range(len(data_on_gpu)):
            data_on_gpu[i].to(cpu, non_blocking=False)

    with torch.cuda.stream(s3):
        for i in range(len(data_on_cpu)):
            data_on_cpu[i].to(cuda, non_blocking=False)

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
s3 = torch.cuda.Stream()

cpu = torch.device("cpu")
cuda = torch.device("cuda")

data_on_gpu = [torch.rand((1024, 1024), device=cuda) for _ in range(100)]
data_on_cpu = [torch.rand((1024, 1024), device=cpu) for _ in range(100)]
matrix_on_gpu = [torch.rand((1024, 1024), device=cuda) for _ in range(1000)]
```

[Trace file](/streams/swimming-in-streams.json.gz). Note that, overlap in trace may vary by GPU.
