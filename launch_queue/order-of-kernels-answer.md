---
layout: page
title: Answer
permalink: /order-of-kernels-answer/
---

To keep the CPU from blocking when it dispatches compute kernels, the GPU maintains a queue of kernel calls - the CUDA launch queue. (You can think of this as an example of the producer-consumer pattern, where the CPU is the producer and the GPU is the consumer.)

It takes time to send the kernel from CPU to GPU, and for small kernels, this time can exceed the time taken to execute the kernel on GPU. 

- In Block 1, the GPU takes long enough to perform the large multiply for the queue to fill up with the small kernels. 
  - When the large matrix multiply finishes, the GPU immediately start executing the enqueued small matrix multiplies.
- In Block 2, the GPU completes each small matrix multiply before the next one is available, so it idles between the small kernels.

More evidence that this is what's happening: increase the number of small kernels and after roughly 35 small matrix multiplies the gaps reappear.

<p align = "center">
  <a href="/launch_queue/files/more_small_kernels.jpg">
    <img src = "/launch_queue/files/more_small_kernels.jpg">
  </a>
</p>
<p align = "center">More small multiples</p>

  - Kineto trace [file](/launch_queue/files/more_small_kernels.json "More Small Kernels Trace File")

## Discussion

### The CUDA Kernel Launch Queue 

<p align = "center">
  <a href="/launch_queue/files/cuda_launch_queue_uarch.jpg">
    <img src = "/launch_queue/files/cuda_launch_queue_uarch.jpg">
  </a>
</p>
<p align = "center">CUDA Kernel Launch Queue Overview</p>

The GPU maintains a queue of kernel calls in the order they are made by the CPU. 
  - The queue is hardware-managed: kernels launch when required resources become available.
  - Each queue entry is quite small: under 4 KB of params. 
  - Each stream has its own queue, can set priorities to individual streams. 
  - If the launch queue reaches a threshold (1024 entries), the CPU will block on
    calling a compute kernel. 
    - This can be problematic if there's other work the CPU could be doing, and should be avoided.

  - *[Advanced]* Asynchronous launches can be disabled by setting `CUDA_LAUNCH_BLOCKING=1`. 
    - This is useful for debugging, especially in the context of multiple 
   streams - see the [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-execution-host-device) 
   for details. 
  - *[Advanced]* Letting the CPU run too far ahead of the GPU can also be problematic: excessive allocations lead to OOM.
    - Nuanced - for single stream, the CUDA caching allocator is smart enough to recycle.
    - Multiple streams, this has happened in practice: solution is to use a [rate limiter](https://pytorch.s3.amazonaws.com/posters/ptc2022/E03.pdf).

### Kernel Dispatch Overhead

What happens in the time from when Python method is invoked to when the kernel actually starts executing on GPU? 

 - Obvious answer: PCIE (Wrong!)
   - Proof: compare Kineto trace for PyTorch program to NSYS trace for functionally equivalent CUDA C program.

<p align = "center">
  <a href="/launch_queue/files/native_cuda.jpg">
    <img src = "/launch_queue/files/native_cuda.jpg">
  </a>
</p>
<p align = "center">Big and small matrix multiplies in CUDA C</p>

CUDA program: almost no difference between launching large gemm first and small gemms first - very little gap between the kenels
 - Actual kernel calls are identical: same kernel, same duration
 - Big: `ampere_sgemm_64x32_sliced1x4_nn`, Small: `ampere_sgemm_32x32_sliced1x4_nn`
 - NSYS trace [file](/launch_queue/files/launchqueue.qdrep), CUDA source [code](/launch_queue/files/launchqueue.cpp)

PyTorch is *supposed* to be a thin veneer around NVIDIA library code
 - Reality: the combination of PyTorch, PyBind, Python is expensive
 - Many operations involved in dispatch
   - Parsing arguments
   - Finding right function to dispatch (v-table, signature scanning)
   - Object construction (memory allocation, lots of memory overhead for small objects)
   - Stride calculation (host side, made complex by possibility of aliasing)

TensorFlow is even worse!
 - Adds one more layer of indirection: protobufs

<p align = "center">
  <a href="/launch_queue/files/tensorflow.jpg">
    <img src = "/launch_queue/files/tensorflow.jpg">
  </a>
</p>
<p align = "center">Big and small matrix multiplies in TensorFlow</p>

   - NSYS trace [file](/launch_queue/files/tf_profile.qdrep), TensorFlow source [code](/launch_queue/files/tf_launch_queue.py)

### How to Reduce Launch Overhead?

#### Don't Let The Queue Empty

Nonempty queue -> hides launch overhead!
- Avoid synchronization / gang up syncs where possible
  - Example: CUDA Caching Allocator - avoids calling cudaFree() which is blocking by recycling memory within streams. 
    - Details: source code ([link](https://github.com/pytorch/pytorch/blob/master/c10/cuda/CUDACachingAllocator.cpp)), also Zach DeVito's blog post ([link](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html))
- Reorder kernels big to small (uncommon)

#### Launch Fewer Kernels

Fusion: combine kernels (["Optimizing Production PyTorch Models’ Performance with Graph Transformations"](https://pytorch.org/blog/optimizing-production-pytorch-performance-with-graph-transformations/))
  - "Horizontal Fusion" - `torch._foreach_mul(P, Q)`, here `P`, `Q` are lists of matrices 
    - Comprehensive list of `torch._foreach` methods ([link](https://pytorch.org/cppdocs/api/file_build_aten_src_ATen_Functions.h.html#file-build-aten-src-aten-functions-h))
    - Many more manual tricks, e.g., keep matrices A1, A2, A3 as a single 3D tensor
  - "Vertical Fusion" - `A.pow_(3.14).mul_(2.71).add_(0.577)` ([link](https://pytorch.org/cppdocs/api/file_build_aten_src_ATen_Functions.h.html#file-build-aten-src-aten-functions-h))
    - Not just less launches: more significantly, avoids repeated memory reads/writes - great solution to low flops for vector operations!
  - PyTorch 2.0: `torch.compile()` does fusions automatically, also exceptional at GPU code generation ([link](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html))
  - Write your own kernel in CUDA C, bind it to PyTorch ([link](https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension))
    - Nontrivial: see this [113 slide deck](https://www.csd.uwo.ca/~mmorenom/HPC-Slides/Optimizing_CUDA_Code-2x2.pdf) that builds up to an efficient matrix transpose.

#### CUDA Graphs

Overhead comes from search: find functions, find memory
- Real ML programs: very repetitive, e.g., training has 1M iterations
  - Data changes, compute remains the same
- Idea: record pointers & computation; replay

<p align = "center">
  <a href="/launch_queue/files/cudagraph_blogpost.jpg">
    <img src = "/launch_queue/files/cudagraph_blogpost.jpg">
  </a>
</p>
<p align = "center">CUDA Graph overview</p>

Applied to the working example: MWE [source](/launch_queue/files/cudagraph_mwe.py), Kineto trace [file](/launch_queue/files/cudagraph_mwe.json), 

<p align = "center">
  <a href="/launch_queue/files/cudagraph_mwe.jpg">
    <img src = "/launch_queue/files/cudagraph_mwe.jpg">
  </a>
</p>
<p align = "center">CUDA Graph applied to big and small matrix multiplies</p>

Reference: [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)

#### Eschew PyTorch

- AI Template: do it all in CUDA C - currently, inference only ([link](https://github.com/facebookincubator/AITemplate))
- [tch-rs](https://github.com/LaurentMazare/tch-rs): Rust bindings for C++ API of PyTorch
  - "Rust is a modern systems programming language focusing on safety, speed, and concurrency. It accomplishes these goals by being memory safe without using garbage collection."
  - Surprising: shows no gain over PyTorch

<p align = "center">
  <a href="/launch_queue/files/rust.jpg">
    <img src = "/launch_queue/files/rust.jpg">
  </a>
</p>
<p align = "center">Big and small matrix multiplies in Rust</p>


### What should you remember in 10 years?

Kernel launch overhead can kill performance - nonempty queues, fusion, CUDA Graphs, eschewing PyTorch are solutions.
