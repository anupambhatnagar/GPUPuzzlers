---
layout: page
title: Answer
permalink: /faster-way-to-add-answer/
---

The `first_sum` implementation is the slowest and the `third_sum` implementation is the
fastest.

<p align = "center">
  <a href="/d2h_sync/annotated_d2h_sync_trace.png">
    <img src = "/d2h_sync/annotated_d2h_sync_trace.png">
  </a>
</p>

The trace above shows that duration of each functions and highlights the numerous device to host
memory copy kernels triggered in the `first_sum` implementation and the various vector ops taking
place in the `second_sum` function. The trace file is available
[here](/d2h_sync/addition_d2h_sync_final.json.gz).

### Analyzing first_sum - numerous device to host copies

``` python
def first_sum(a_tensor):
    total = 0.0
    for i in range(a_tensor.size()[0]):
        total += a_tensor[i].cpu()
    return total
```


<p align = "center">
  <a href="/d2h_sync/first_sum_zoomed.png">
    <img src = "/d2h_sync/first_sum_zoomed.png">
  </a>
</p>

<p align = "center"> Zoomed view - first_sum </p>


In the `first_sum` function the number of copies from the device to the host is equal to the size of
the tensor. These are incurred due to the `.cpu()` call in the for loop. Each `.cpu()` call moves a
small amount of data from the GPU to the CPU and takes about 1 microseconds. These repeated calls,
highighted in the zoomed trace above make the `first_sum` implementation the slowest one.

### Analyzing second_sum - no device to host copies

``` python
def second_sum(a_tensor):
    total = torch.zeros(1, device='cuda')
    for i in range(a_tensor.size()[0]):
        total += a_tensor[i]
    return total
```

<p align = "center">
  <a href="/d2h_sync/second_sum_zoomed.png">
    <img src = "/d2h_sync/second_sum_zoomed.png">
  </a>
</p>

<p align = "center"> Zoomed view - second_sum </p>

The `second_sum` function initializes and places the `total` variable on the GPU. Addition of each
element of `tensor` to `total` triggers vector op kernels. The addition is slowed due to the launch
overhead of these kernels. The arithmetic intensity of vector ops involving small tensors is low and
such operations can be done on the CPU, when reasonable. Even though __there are no instances of
device to host copies and synchronizations__ in the `second_sum` implementation, the GPU is
considerably underutilized. This can be seen by the ~20 microsecond gaps between consecutive kernels
on the stream 7.

### Analyzing third_sum - addition on the CPU

``` python
def third_sum(a_tensor):
    total = 0.0
    tensor_on_cpu = a_tensor.cpu()
    for i in range(tensor_on_cpu.size()[0]):
        total += tensor_on_cpu[i]
    return total
```

Finally, the `third_sum` implementation copies the entire tensor to the CPU and pays a small one
time cost to transfer data. Precisely speaking, 4 * 4096 bytes are transferred in 2 microseconds.
Thus the achieved PCIe bandwidth is approximately 8 GB/sec. The summation is done on the CPU as
`total` and the elements of the tensor are on the CPU.

## Discussion

__What is the fastest way to add in PyTorch?__

`torch.sum` has better performance than the above implementations. The functions above are contrived
examples to demonstrate device to host synchronization. The table below summarizes the time taken by
the three functions above and `torch.sum`:

| function| CPU time (ms) | GPU time(ms) |
|--- | --- | --- |
| first_sum | 183 | 181 |
| second_sum | 87 | 86  |
| third_sum | 25 | 0.001 |
| torch.sum() (tensor on GPU) | 0.161 | 0.009|
| torch.sum() (tensor on CPU) | 1.8 | NA |

__Synchronization points__

- Iterating over a for loop as in `first_sum` causes synchronization and leads to performance
  degradation. These synchronization points are visible and avoidable. A better implementation would
  parallelize the computation by launching multiple threads rather than iterating over it.

- As seen in the `second_sum` example an absence of synchronization points does not guarantee good
  performance. Executing multiple small kernels iteratively does not utilize the GPU completely.

- Synchronization points can stall the CUDA launch queue which can make the job CPU bound. More
  about this in the next post.

__Some naturally occurring synchronization points__

- Call to `torch.cuda.synchronize()`
- `.item()`, `.cpu()`, `torch.nonzero`, `torch.masked_select`
- Logging statements from the GPU


__Finding synchronization points__

Use
[`torch.cuda.set_sync_debug_mode()`](https://pytorch.org/docs/stable/generated/torch.cuda.set_sync_debug_mode.html),
when possible. Currently, this feature does not support torch.distributed and torch.sparse
namespaces.

__Device synchronization vs. Stream synchronization vs. Event synchronization__

A _stream_ is simply a sequence of operations that are performed in order on the device. Operations in
different streams can be interleaved and in some cases overlapped - a property that can be used to
hide data transfers between the host and the device.

A CUDA _event_ is a synchronization marker that can be used to monitor the device’s progress, to
accurately measure timing, and to synchronize CUDA streams.

- Using `torch.cuda.synchronize()` leads to device synchronization which blocks execution on the CPU
thread until the device has completed all preceding tasks. It waits for the GPU to finish kernel
execution on __all streams__.

- CUDA streams can be synchronized using `torch.cuda.Stream.synchronize()`. Execution is blocked on the
CPU thread until the device has executed all kernels on the __specified stream__.

- Event synchronization can be triggered using `torch.cuda.Event.synchronize()`. It prevents the CPU
thread for proceeding until the event is completed.

### Analyzing with Holistic Trace Analysis

Looking at the trace, you may note that the `cudamemcpyasync` event on the cpu is a longer duration
than the corresponding operation (memcpydtoh) on the gpu. This may not be easy to find in general.

The [Holistic Trace Analysis](https://github.com/facebookresearch/holistictraceanalysis) (HTA hta)
library provides a convenient way to identify this behavior using the [Cuda Kernel Launch
Statistics](https://hta.readthedocs.io/en/latest/source/features/cuda_kernel_launch_stats.html)
feature. Using the pytorch profiler traces, HTA provides insights for performance debugging. In
particular, the
[`get_cuda_kernel_launch_stats`](https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html#hta.trace_analysis.traceanalysis.get_cuda_kernel_launch_stats)
function displays the distribution of gpu kernels (in particular, `cudaLaunchKernel`,
`cudaMemcpyAsync` and `cudaMemsetAsync`) whose duration is less than the corresponding CPU event.

Profiling each function as an independent python program we generated three traces and analyzed them
with the `get_cuda_kernel_launch_stats` feature. One of the outputs from the
`get_cuda_kernel_launch_stats` function are the graphs below which show that there are thousands of
GPU events with duration shorter than the corresponding CPU event, thus highlighting the issues with
the `first_sum` and `second_sum` implementations.

<p align = "center">
  <a href="/d2h_sync/first_sum_kernel_launch_stats.png">
    <img src = "/d2h_sync/first_sum_kernel_launch_stats.png">
  </a>
</p>

<p align = "center"> Cuda Kernel Launch Stats - first_sum</p>

<p align = "center">
  <a href="/d2h_sync/second_sum_kernel_launch_stats.png">
    <img src = "/d2h_sync/second_sum_kernel_launch_stats.png">
  </a>
</p>

<p align = "center"> Cuda Kernel Launch Stats - second_sum</p>

<p align = "center">
  <a href="/d2h_sync/third_sum_kernel_launch_stats.png">
    <img src = "/d2h_sync/third_sum_kernel_launch_stats.png">
  </a>
</p>

<p align = "center"> Cuda Kernel Launch Stats - third_sum</p>
the graphs above were generated using the following code snippet:

``` python
from hta.trace_analysis import traceanalysis
analyzer = traceanalysis(trace_dir = "/path/to/trace/folder")
kernel_stats = analyzer.get_cuda_kernel_launch_stats()
```

Here are the traces for the [first_sum](/d2h_sync/addition_first_sum.json.gz),
[second_sum](/d2h_sync/addition_second_sum.json.gz) and
[third_sum](/d2h_sync/addition_third_sum.json.gz) functions. <!--and a [notebook]() showing how to use HTA. -->

### Key Takeaway

When possible, trade multiple small device to host copies (kernels) with a few large device to host
copies (kernels) and be aware of device to host synchronization points.

### Explore More

- Find the kernels taking the most time in your model using the Kernel Breakdown feature in
Holistic Trace Analysis.
- Build a roofline model to find if the kernels are compute bound or memory bandwidth bound.
