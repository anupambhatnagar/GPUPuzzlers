---
layout: page
title: Answer
permalink: /order-of-operations-answer/
---

The image shown below indicates that it is faster to do the large matrix squaring first. Precisely
speaking, the trace shows that `large_matmul_first` function takes 449 microseconds and the
`small_matmul_first` function takes 858 microseconds.

![Annotated trace](/launch_queue/kernel_launch_annotated_trace.png)

To explain this behavior we begin by introducting a couple of terms:

_CUDA Kernel Launch Queue_ - The GPU maintains a queue of kernel calls in the order they are made
by the host. After the kernels are dispatched, this ensures that the CPU is not blocked.

_PyTorch Dispatch Overhead_ - Time taken to launch a kernel and can dominate the time
taken to perform the actual computation on the GPU.

``` python
def small_matmul_first():
    for _ in range(10):
       torch.matmul(small_matrix, small_matrix)
    torch.matmul(large_matrix, large_matrix)
```

In the `small_matmul_first` function, the GPU completes each small matrix multiply before the next one is
ready. The PyTorch dispatch overhead is larger than the time taken to do the small matrix
multiplications and thus the GPU idles between the small matrix multiplications.

``` python
def large_matmul_first():
    torch.matmul(large_matrix, large_matrix)
    for _ in range(10):
       torch.matmul(small_matrix, small_matrix)
```

In the `large_matmul_first` function, the GPU takes longer to perform the large matrix multiply,
allowing the CUDA launch queue can fill up. While the large matrix multipcation executes the small matrix
multiplications have been dispatched and the GPU can immediately compute the small matrix
multiplications leading to a lower wall clock time.

## Discussion

### Kernel Launch Queue

Rewrite the section as follows:
1. explain the Microarchitecture diagram
![CUDA Launch Queue Microarchitecture](/launch_queue/cuda_launch_queue_uarch.jpg?raw=true "CUDA
Launch Queue Microarchitecture")
  - Each queue entry is constrained to be very small: under 1 KB. (why? and how?) It's basically the
    function pointer, and arguments, which are pointers to tensors. Notably, a host-side tensor
    cannot be an argument - tensors have to be explicitly copied to and from device.
  - If the kernel launch queue reaches a threshold (around 1000 entries), the host will block on
    calling a compute kernel. This can be problematic if there's other work the host could be doing,
    and should be avoided.

1. What to do in order to write high performance programs
  - operator fusion
  - reorder independent operations (as in this example)
1. What are some things to watch out for
  - don't let the queue flush out
1. Empirical observation
  - performing more small matmuls

1. Discuss notable exception cases (to be used as a teaser for future lessons)
1. Important: the GPU maintains one queue per stream. (actually, the kernel launch queue on a stream
  is subdivided into execution queue, D2H copy queue and H2D copy queue within the stream.


---
Rough notes below



One of the guiding principles for desinging high performance PyTorch programs is to keep the
kernel launch queue non-empty. Some ways to achieve this are:

1. Operator fusion, wherein we do more work in a single kernel.
1. Avoiding operations that force the queue to be flushed - a common example is a GPU to CPU copy,
   which leads to a read-after-write data hazard. (Flushing the queue is analogous to stalling the
   pipeline in a pipelined processor.)
1. Reordering independent operations to bring the slower operations to the front of the queue (this
    example).

1. There are some exceptions to asynchronous kernel launches, notably around CPU-GPU memory copies
  and synchronization primitives; we'll discuss these in another unit.

1. Asynchronous launches can be disabled by setting `CUDA_LAUNCH_BLOCKING=1`. This is useful for
   debugging, especially in the context of multiple streams - [see the CUDA Toolkit Documentation
   for details](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-execution-host-device).


1. If we performed several more small matrix multiplications after the large one, we would expect at
   some point the CUDA launch queue will empty out (since the service rate is higher than the
   arrival rate). Empirically, this happens if we have 40 or more small matrix multiplications.

1. In this unit, we're working with a single
   [stream](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams). In general
   the GPU maintains multiple queues, one per stream. (so?)

<!--- from https://slideplayer.com/slide/8211225/ -->
<!--- see also http://xzt102.github.io/publications/2018_GPGPU_Sooraj.pdf -->
<!--
- TODO: from Yueming, add NSIGHT traces, understand what is happening there (sending multiple kernels in one shot?)
- TODO: cudnn optimization enable, see if that leads to pytorch matching CUDA code
- TODO: summarize jason/kimish insights into launch overhead
- TODO: see if we can trace PCIE to see how much that contributes and if CUDA graph/CUDA code do group transactions
- TODO: explain need for Kineto and CUPTI - profiler is not enough
-->


![CUDA Launch Queue Trace](/launch_queue/cuda_launch_queue.jpg?raw=true "CUDA Launch Queue Trace")

