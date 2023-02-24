---
layout: page
title: Answer
permalink: /order-of-kernels-answer/
---

To keep the CPU from blocking when it dispatches compute kernels, the GPU maintains a queue of kernel calls - the CUDA launch queue. (You can think of this as an example of the producer-consumer pattern, where the CPU is the producer and the GPU is the consumer.)

It takes time to send the kernel from CPU to GPU, and for small kernels, this time can exceed the time taken to execute the kernel on GPU. 

- In Block 1, the GPU takes long enough to perform the large multiply for the queue to fill up with the small kernels. Therefore, after the large matrix multiply finishes, the GPU can immediately start executing the small matrix multiplies which have queued up.
- In Block 2, the GPU completes each small matrix multiply before the next one is available, so it idles between the small kernels.

More evidence that this is what's happening: increase the number of small kernels and after roughly 35 small matrix multiplies the gaps reappear.

![More Small Kernels - Separation Starts](/launch_queue/files/more_small_kernels.jpg?raw=true "More Small Kernels")

  - Kineto trace [file](/launch_queue/files/more_small_kernels.json "More Small Kernels Trace File")

## Discussion

### The CUDA Kernel Launch Queue 

![CUDA Launch Queue Microarchitecture](/launch_queue/files/cuda_launch_queue_uarch.jpg?raw=true "CUDA
Launch Queue Microarchitecture")

  - The GPU maintains a queue of kernel calls in the order they are made 
    by the CPU. After the kernels are dispatched, this ensures that the CPU is not blocked.
  - The queue is hardware-managed: kernel launched when required resources become available.
  - Each queue entry is quite small: under 4 KB of params. 
  - Each stream has its own queue, can set priorities to individual streams. (Actually 3 queues: compute, D2H, H2D.)
  - If the launch queue reaches a threshold (1024 entries), the CPU will block on
    calling a compute kernel. 
    - This can be problematic if there's other work the CPU could be doing, and should be avoided.
  - Letting the CPU run too far ahead of the GPU can also be problematic: excessive allocations lead to OOM.
    - Actual story is more nuanced - for single stream, the CUDA caching allocator is smart enough to recycle.
    - With multiple streams, this has happened in practicel solution is to use a [rate limiter](https://pytorch.s3.amazonaws.com/posters/ptc2022/E03.pdf).
  - Asynchronous launches can be disabled by setting CUDA\_LAUNCH\_BLOCKING=1. 
    - This is useful for debugging, especially in the context of multiple 
   streams - see the [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-execution-host-device) 
   for details. 

### Kernel Dispatch Overhead

Where happens in the time from when Python method is invoked to when the kernel actually starts executing on GPU? (Assume launch queue is empty.)

 - Obvious answer: PCIE (Wrong!)
 - Proof: compare Kineto trace for PyTorch program to NSYS trace for functionally equivalent CUDA C program.
![Native CUDA Launch Overhead](/launch_queue/files/native_cuda.jpg?raw=true "Native Cuda Launch Overhead")
 - CUDA program: almost no difference between launching large gemm first and small gemms first - very little gap between the kenels
   - Actual kernel calls are identical: same kernel, same duration
   - Big: ampere\_sgemm\_64x32\_sliced1x4\_nn, Small: ampere\_sgemm\_32x32\_sliced1x4\_nn
   - NSYS trace [file](/launch_queue/files/launchqueue.qdrep), CUDA source [code](/launch_queue/files/launchqueue.cpp)

 - PyTorch is *supposed* to be a thin veneer around NVIDIA library code
   - Reality: the combination of PyTorch, PyBind, Python is expensive
   - Many operations involved in dispatch
     - Parsing arguments
     - Finding right function to dispatch (v-table, signature scanning)
     - Object construction (lots of memory overhead, call to CudaCachingAllocator)
     - Stride calculation (host side, made complex by possibility of aliasing)
 - How does TensorFlow handle launch overhead?
   - It's even worse!
   - Reason: adds protobufs
![TensorFlow Launch Overhead](/launch_queue/files/tensorflow.jpg?raw=true "TensorFlow Launch Overhead")
   - NSYS trace [file](/launch_queue/files/tf_profile.qdrep), TensorFlow source [code](/launch_queue/files/tf_launch_queue.py)

### How to Reduce Launch Overhead?

#### Don't Let The Queue Empty

Nonempty queue -> hides launch overhead!
- Avoid synchronization / gang up syncs where possible
  - Example: CUDA Caching Allocator - avoids calling cudaFree() which is blocking by recycling memory within streams. (Details ([link](https://github.com/pytorch/pytorch/blob/master/c10/cuda/CUDACachingAllocator.cpp)), also Zach's beautiful blog post ([link](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html)).)
- Move up big kernels (uncommon)

#### Launch Fewer Kernels

- Fusion: combine kernels (["Optimizing Production PyTorch Models’ Performance with Graph Transformations"](https://pytorch.org/blog/optimizing-production-pytorch-performance-with-graph-transformations/))
  - "Horizontal Fusion" - `torch._foreach_mul(P, Q)`, here `P`, `Q` are lists of matrices 
    - Comprehensive list of `torch._foreach` methods ([link](https://pytorch.org/cppdocs/api/file_build_aten_src_ATen_Functions.h.html#file-build-aten-src-aten-functions-h))
    - Many more manual tricks, e.g., keep matrices A1, A2, A3 as a single 3D tensor
  - "Vertical Fusion" - `A.pow(3.14).mul(1.41).add(2.71)` ([link](https://pytorch.org/cppdocs/api/file_build_aten_src_ATen_Functions.h.html#file-build-aten-src-aten-functions-h))
     - Not just less launches: more significantly, avoids repeated memory reads/writes - great solution to low flops for vector operations!
- PT2: `torch.compile()` does fusions automatically (also exceptional code generation) ([link](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html))

#### CUDA Graphs

- Overhead comes from search: find functions, find memory
- Real ML programs: very repetitive, e.g., training has 1M iterations
  - Data changes, compute remains the same
- Idea: record pointers & computation - then replay
![CUDAGraph Idea](/launch_queue/files/cudagraph_blogpost.jpg?raw=true "CUDAGraph")
- Can see the amortized benefit of forming the graph 
![CUDAGraph Applied to MWE](/launch_queue/files/cudagraph_mwe.jpg?raw=true "CUDAGraph")
- Kineto trace [file](/launch_queue/files/cudagraph_mwe.json), CUDAGraph MWE [file](/launch_queue/files/cudagraph_mwe.py)
- Reference: [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)

#### Eschew PyTorch

- AI Template: do it all in CUDA C - currently, inference only ([link](https://github.com/facebookincubator/AITemplate))
- [tch-rs](https://github.com/LaurentMazare/tch-rs): Rust bindings for C++ API of PyTorch
  - "Rust is a modern systems programming language focusing on safety, speed, and concurrency. It accomplishes these goals by being memory safe without using garbage collection."
  - Surprisingly, showed no gain over PyTorch
![Rust Launch Overhead](/launch_queue/files/rust.jpg?raw=true "Rust")

### What will you remember in 10 years?

Kernel launch overhead can kill performance - full queues, fusion, CUDA Graphs, eschewing PyTorch are solutions.
