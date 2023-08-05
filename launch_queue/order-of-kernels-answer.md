---
layout: page
title: Answer
permalink: /order-of-kernels-answer/
---

To keep the CPU from blocking when it dispatches compute kernels, the GPU maintains a queue of
kernel calls - the CUDA launch queue. This can be thought of as an example of the producer-consumer
pattern, where the CPU is the producer and the GPU is the consumer.

It takes time to send the kernel from CPU to GPU (the “kernel launch delay”), and for small kernels,
this time can exceed the time taken to execute the kernel on GPU.

Equipped with this knowledge, we can interpret the [PyTorch Profiler
trace](https://www.gpupuzzlers.com/launch_queue/more_small_kernels.json) as follows:

- In Block 1, the GPU takes long enough to perform the large multiply for the queue to fill up with
  the small kernels. When the large matrix multiply finishes, the GPU immediately starts executing
  the enqueued small matrix multiplies.
- In Block 2, the GPU completes each small matrix multiply before the next one is available, so it
  idles between the small kernels.

If we increase the number of small matrix multiplications in large_small, after roughly 35 small
matrix multiplies the gaps reappear. This is because the queue eventually drains, exposing the
kernel launch delay again.

<p align = "center">
  <a href="/launch_queue/more_small_kernels.jpg">
    <img src = "/launch_queue/more_small_kernels.jpg">
  </a>
  With more small multiplications, the launch queue drains
</p>

## Discussion

__What is the CUDA kernel launch queue?__

<p align = "center">
  <a href="/launch_queue/cuda_launch_queue_uarch.jpg">
    <img src = "/launch_queue/cuda_launch_queue_uarch.jpg">
  </a>
  CUDA kernel launch queue
</p>

The GPU maintains a queue of kernel calls in the order they are made by the CPU. Some salient facts
about this queue include:
- The queue is hardware-managed: kernels launch when required resources become available.
- If the launch queue reaches a threshold (1024 entries), the CPU will block on calling a compute
  kernel. This can be problematic if there is other work the CPU could be doing, and should be
  avoided.
- Each queue entry is quite small: under 4 KB of kernel params. (Even though a kernel may operate on
  huge tensors, the queue entry will use their GPU memory addresses.)

__What is the root-cause of the kernel dispatch overhead?__

Specifically, what happens in the time from when a Python method is invoked to when the kernel
actually starts executing on the GPU? A common guess is that the overhead comes from PCIE, but this
is incorrect. Proof comes from comparing the Kineto trace for our PyTorch program to the NSight
Systems (NSYS) trace for the functionally equivalent CUDA C program.

<p align = "center">
  <a href="/launch_queue/native_cuda.jpg">
    <img src = "/launch_queue/native_cuda.jpg">
  </a>
Big and small matrix multiplications in CUDA C
</p>

For the CUDA program we see almost no difference between launching the large gemm first and the
small gemms first. In particular the gaps between kernels is very small in both cases.
- Note that the actual kernel calls are identical and have the same duration
- Big: ampere_sgemm_64x32_sliced1x4_nn, Small: ampere_sgemm_32x32_sliced1x4_nn
- NSYS [trace file](https://www.gpupuzzlers.com/launch_queue/launchqueue.qdrep), CUDA [source
  code](https://www.gpupuzzlers.com/launch_queue/launchqueue.cpp)

PyTorch is supposed to be a thin veneer around NVIDIA library code. However, the reality is that
the combination of PyTorch, PyBind, Python, and, ultimately, the C++ dispatcher is very expensive.
There are many operations involved in dispatch:

- Parsing arguments
- Finding the right function to dispatch (v-table, signature scanning)
- Object construction in Python, C++, as well as the GPU. This involves memory allocations - note
  that there's lots of memory overhead for small objects.
- Stride calculation (done on host side, made complex by possibility of aliasing)

TensorFlow on GPU is even slower than PyTorch: it adds one more layer of indirection via protobuf.
(NSYS [trace file](/launch_queue/tf_profile.qdrep), TensorFlow [source
code](https://www.gpupuzzlers.com/launch_queue/tf_launch_overhead.py))

<p align = "center">
  <a href="/launch_queue/tensorflow.jpg">
    <img src = "/launch_queue/tensorflow.jpg">
  </a>
  Big and small matrix multiplications in TensorFlow
</p>


__How can we reduce the kernel launch overhead?__

1. Don’t let the queue empty

    A non-empty queue hides the launch overhead. We can keep the queue nonempty by avoiding
    synchronization/ganging up synchronization events wherever possible As an example, the CUDA Caching
    Allocator avoids calling cudaFree() which is a synchronizing event by recycling memory within
    streams (Details: [source
    code](https://github.com/pytorch/pytorch/blob/master/c10/cuda/CUDACachingAllocator.cpp), also
    Zachary DeVito’s [blog post](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html)).

    Reordering independent kernels to send big kernels before small ones also serves to keep the queue
    from emptying, just like in our working example.

1. Replace lots of small kernels with a few big ones

    The key idea of fusion is to [combine
    kernels](https://pytorch.org/blog/optimizing-production-pytorch-performance-with-graph-transformations/).
    There are two basic approaches to fusion:

    - Horizontal Fusion, in which independent kernels (i.e., kernels with no data-dependencies) are
      merged.
      - E.g., `torch._foreach_mul(P, Q)` where P, Q are lists of matrices
      - See this link for the comprehensive list of `torch._foreach` methods (link).
      - There’s more manual approaches to horizontal fusion, e.g., keep matrices A, B, C as a single 3D
        tensor.
    - Vertical Fusion in which interdependent kernels are “in-lined”, e.g.,
      `A.pow_(3.14).mul_(2.71).add_(0.577)` can be replaced by a single kernel.

    Fusion doesn’t just reduce launch overhead: it can also avoid repeated memory reads/writes, and
    is a great way to increase arithmetic intensity for vector operations. See the [To Fuse or Not
    to Fuse?](https://www.gpupuzzlers.com/posts/fusion/) puzzler for an example.

1. Use CUDA Graphs

    <p align = "center">
      <a href="/launch_queue/cudagraph_blogpost.jpg">
        <img src = "/launch_queue/cudagraph_blogpost.jpg">
      </a>
      CUDA Graph Overview
    </p>

    Real ML programs are repetitive, e.g., training has millions of iterations. The data changes but the
    computation remains the same.

    The basic idea of CUDA Graphs is to record computation and memory
    addresses and replay them. Here’s CUDA Graphs applied to this puzzler: [source
    code](https://www.gpupuzzlers.com/launch_queue/cudagraph_mwe.py), and the PyTorch Profiler
    [trace](https://www.gpupuzzlers.com/launch_queue/cudagraph_mwe.json).

    Reference: [Accelerating PyTorch with CUDA
    Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)

    <p align = "center">
      <a href="/launch_queue/cudagraph_mwe.jpg">
        <img src = "/launch_queue/cudagraph_mwe.jpg">
      </a>
    CUDA Graph applied to big and small matrix multiplication
    </p>

1. Eschew PyTorch

    The [AI Template](https://github.com/facebookincubator/AITemplate) approach is to do it all in CUDA
    C - currently, it’s inference only.

    The [tch-rs](https://github.com/LaurentMazare/tch-rs) project adds Rust bindings for the C++ API of
    PyTorch. Rust is a “modern systems programming language focusing on safety, speed, and concurrency;
    it accomplishes these goals by being memory safe without using garbage collection.” However, tch-rs
    is not much better than PyTorch with respect to dispatch overhead because it inherits the C++
    dispatcher from PyTorch, which is where the majority of dispatch overhead comes from.

    <p align = "center">
      <a href="/launch_queue/rust.jpg">
        <img src = "/launch_queue/rust.jpg">
      </a>
      Big and small matrix multiplications in Rust
    </p>

## What should you remember in years to come?

PyTorch kernel launch overhead can kill performance - keeping queues from emptying, kernel fusion,
CUDA Graphs, and eschewing PyTorch are solutions.

## Explore more

- With PyTorch 2.0, `torch.compile()` does many fusions automatically. It’s also exceptional at [GPU
  code generation](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).

- You can also try writing your own fused kernel in CUDA C and [binding it to
  PyTorch](https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension).
