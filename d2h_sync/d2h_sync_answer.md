---
layout: page
title: Answer 
permalink: /faster-way-to-add-answer/
---

The `first_sum` implementation is the slowest and the `third_sum` implementation is the
fastest.
<!-- mention that none of them are acceptable perf wise? btw, i tried cpu and gpu .sum(),
small sizes are not getting close to peak mem bw, but 2^26 does achieve it on both, and gpu
is ~10 faster as expected -->

<p align = "center">
  <img src = "/d2h_sync/annotated_d2h_sync_trace.png">
</p>
The trace above shows that duration of each functions and highlights the numerous device to host
memory copy kernels triggered in the `first_sum` implementation and the various vector ops taking
place in the `second_sum` function. The trace file is available [here]().

### Analyzing first_sum

``` python
def first_sum(tensor):
    total = 0.0
    for i in range(tensor.size()[0]):
        total += tensor[i].cpu()
    return total
```

In the `first_sum` function the number of copies from the device to the host is equal to the size of
the tensor. These are incurred due to the `.cpu()` call in the for loop. Each `.cpu()` call moves a
small amount of data from the GPU to the CPU and takes about ~26 microseconds. These repeated calls,
highighted in the zoomed trace below make the `first_sum` implementation the slowest one.
<!-- is the 26 just the kernel time or does it include the gap to the next kernel call? -->


### Analyzing second_sum 

``` python
def second_sum(tensor):
    total = torch.zeros(1, device='cuda')
    for i in range(tensor.size()[0]):
        total += tensor[i]
    return total
```

The `second_sum` function initializes and places the `total` variable on the GPU. Addition of each
element of `tensor` to `total` triggers a vector op which is inefficient. 
<!-- I am uncomforable with calling the vector op inefficient, since it divert attention from the much
bigger problem here, which is calling a kernel to do a single add, this has huge overhead -->
The arithmetic intensity
of vector ops is low and such operations should be performed on the CPU, when possible. 
<!-- we talked about doing vector ops on CPU as being something that should be done within reason, 
i.e., don't move all vector ops to CPU, that will be much worse because of copy cost -->
While there are fewer instances of device to host copies and synchronizations in the `second_sum`
implementation, the GPU is considerably underutilized.


### Analyzing third_sum

``` python
def third_sum(tensor):
    total = 0.0
    tensor_on_cpu = tensor.cpu()
    for i in range(tensor_on_cpu.size()[0]):
        total += tensor_on_cpu[i]
    return total
```

Finally, the `third_sum` implementation copies the entire tensor to the CPU and pays a small one
<!-- relatively samll onetime cost - quantify it, and mention the implied PCIE BW, which i think is 
(4 * 4096 / 2) * 1e6/1e9 = 8.192 GB/sec -->
time cost to transfer data. Additionally, the low arithmetic intensity operations (i.e. summing
values) is done on the CPU as `total` and the elements of the tensor are on the CPU. 

<!-- moved to discussion? -->
### Analyzing with Holistic Trace Analysis

Looking at the trace, readers may note that the `cudaMemcpyAsync` event on the CPU is a lot more
expensive than the corresponding operation (MemcpyDtoH) on the GPU. In particular the duration of
the CPU event is longer than the kernel execution on the GPU. This may not be easy to find.

The [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) (HTA) library
provides a convenient way to identify this behavior using the [CUDA Kernel Launch
Statistics](https://hta.readthedocs.io/en/latest/source/features/cuda_kernel_launch_stats.html)
feature. Using the PyTorch Profiler traces, HTA provides insights for performance debugging. In
particular, the [`get_cuda_kernel_launch_stats`](https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_cuda_kernel_launch_stats)
function displays the distribution of GPU kernels (in particular, `CudaLaunchKernel`, `CudaMemcpyAsync`
and `CudaMemsetAsync`) whose duration is less than the corresponding CPU event.

<p align = "center">
  <img src = "/d2h_sync/d2h_sync_short_gpu_kernels.png">
</p>

The graph above was generated from the following three lines of code

``` python
from hta.trace_analysis import TraceAnalysis
analyzer = TraceAnalysis(trace_dir = "/path/to/trace/folder")
kernel_stats = analyzer.get_cuda_kernel_launch_stats(runtime_cutoff=20)
```

One of the outputs from the `get_cuda_kernel_launch_stats` function is the graph above which shows that
there are thousands of GPU events with duration shorter than the corresponding CPU event, thus
highlighting the issues with the `first_sum` and `second_sum` implementations.

<!-- add to discssion:
- what's the fastest way to add?
- why are sync points bad? 
  - inside a for loops are dangerous:
    - obvious: sync is bad
    - less obvious: no sync, still bad (many small kernel calls)
      - solution: foreach kernel wrapper, fusion, operate on single tensor
      - soltion: cudagraphs
  - forward reference to next lecture, stall the queue
- how to find sync points?
- naturally occurring sync points: item(); logging; few kernels
- difference between device sync/stream sync/event sync (may be too stream focused, which we haven't talked about? 
- HTA
-->

### Key Takeaways

When possible,

1. Trade multiple small device to host copies with a few large device to host copies. More
   generally, make yourself aware of device to host synchronization points.
   <!-- how? via reading trace? -->
1. Avoid low arithmetic intensity operations on the GPU. The GPU should be used for large matrix
   multiplications which have high arithmetic intensity.
   <!-- I don't agree with this, what are you supposed to do if you have sigmod, pow, etc? move them to CPU?
   our data centers are >50% low arithm intensity (TBE, vector ops), and we are fine with that, since GPU also offers tremendous mem BW -->

### Food for thought

Highlight other ways in which device to host synchronization may be a bottleneck.
<!-- show code snippets with item(), other kernels, logging, tell them to try out -->
