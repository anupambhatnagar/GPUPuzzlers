---
layout: page
title: Answer
permalink: /order-of-kernels-answer/
---

To keep the CPU from blocking when it dispatches compute kernels, the GPU maintains a queue of kernel calls - the CUDA launch queue. (You can think of this as an example of the producer-consumer pattern, where the CPU is the producer and the GPU is the consumer.)

It takes time to send the kernel from CPU to GPU, and for small kernels, this time can exceed the time taken to execute the kernel on GPU. 

- In Block 1, the GPU takes long enough to perform the large multiply for the queue to fill up with the small kernels. Therefore, after the large matrix multiply finishes, the GPU can immediately start executing the small matrix multiplies.
- In Block 2, the GPU completes each small matrix multiply before the next one is available, so it idles between the small kernels.

Further evidence that this is what's happening comes from increasing the number of small kernels after the big kernel - 

![More Small Kernels - Separation Starts](/launch_queue/files/more_small_kernels.jpg?raw=true "More Small Kernels")
[Kineto Trace](/launch_queue/files/more_small_kernels.json "More Small Kernels Trace File")

## Discussion

### The CUDA Kernel Launch Queue 

![CUDA Launch Queue Microarchitecture](/launch_queue/files/cuda_launch_queue_uarch.jpg?raw=true "CUDA
Launch Queue Microarchitecture")

_CUDA Kernel Launch Queue_ - The GPU maintains a queue of kernel calls in the order they are made 
    by the CPU. After the kernels are dispatched, this ensures that the CPU is not blocked.
  - The queue is hardware-managed: kernel launched when required resources become available.
  - Each queue entry is quite small: under 4 KB of params. 
  - Each stream has its own queue, can set priorities to individual streams. (Actually 3 queues: compute, D2H, H2D.)
  - If the launch queue reaches a threshold (1024 entries), the CPU will block on
    calling a compute kernel. This can be problematic if there's other work the CPU could be doing,
    and should be avoided.
  - If the CPU is running too far ahead of the GPU, out-of-memory can happen: the CPU thread issues 
    the next allocation before the previous free completes on the GPU, so the CUDA caching 
    allocator cannot reuse that block to serve the allocation. This has happened in practice, and the
    solution is natural - use a [rate limiter](https://pytorch.s3.amazonaws.com/posters/ptc2022/E03.pdf).
 - Asynchronous launches can be disabled by setting CUDA\_LAUNCH\_BLOCKING=1. This is useful for debugging, especially in the context of multiple 
   streams - see the [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-execution-host-device) 
   for details. There are some exceptions to asynchronous kernel launches, notably around CPU-GPU memory copies and synchronization primitives; 
   we'll discuss these in another unit.

### Kernel Dispatch Overhead

Time from when Python method is invoked to when kernel actually starts executing on GPU (assuming launch queue is empty).

Where does it come from?

 - Obvious answer: PCIE (Wrong!)
 - Proof: compare Kineto trace for PyTorch program to NSYS trace for functionally equivalent CUDA C program.
 - CUDA program: almost no difference between launching large gemm first and small gemms first - very little gap between the kenels
   - Actual kernel calls are identical: same kernel, same duration
   - Big: ampere\_sgemm\_64x32\_sliced1x4\_nn, Small: ampere\_sgemm\_32x32\_sliced1x4\_nn
![Native CUDA Launch Overhead](/launch_queue/files/native_cuda.jpg?raw=true "Native Cuda Launch Overhead")
[NSYS trace](/launch_queue/files/launchqueue.qdrep)
[CUDA Source](/launch_queue/files/launchqueue.cpp)

 - PyTorch is supposed to be a thin veneer around NVIDIA library code
   - Reality: the combination of PyTorch, PyBind, Python is quite expensive
   - Many operations involved in dispatch
     - Parsing arguments
     - Finding right function to dispatch (v-table, signature scanning)
     - Object construction (lots of memory overhead, call to CudaCachingAllocator)
     - Stride calculation
 - How does TensorFlow handle launch overhead?
   - It's even worse!
   - Reason: uses protobufs
![TensorFlow Launch Overhead](/launch_queue/tensorflow.jpg?raw=true "TensorFlow Launch Overhead")
[NSYS trace](/launch_queue/files/tf_profile.qdrep)
[CUDA Source](/launch_queue/files/tf_launch_queue.py)

### How to Reduce Launch Overhead?

#### Don't Let the Queues Empty

Empty queues -> GPU idle
- Avoid synchronization / gang up syncs where possible
- Move up big kernels (uncommon)

#### Launch Fewer Kernels

- "Horizontal Fusion" - `torch.\_foreach\_mul(P, Q)`, here P, Q are lists of matrices 
  - Many more manual tricks, e.g., keep matrices A1, A2, A3 as a single 3D tensor
- "Vertical Fusion" - `A.pow(3.14).mul(1.41).add(2.71)`
   - Not just less launches - more significantly, avoids repeated memory reads/writes
- PT2: torch.compile() does fusions automatically (also exceptional code generation)

#### CUDAGraphs

- Overhead comes from search: find functions, find memory
- Real ML programs: very repetitive, e.g., training has 1M iterations
  - Data changes, compute remains the same
- Idea: record pointers, computation - then replay
![CUDAGraph Idea](/launch_queue/files/cudagraph_blogpost.jpg?raw=true "CUDAGraph")
- Can see the amortized benefit of forming the graph 
![CUDAGraph Applied to MWE](/launch_queue/files/cudagraph_mwe.jpg?raw=true "CUDAGraph")
[Kineto Trace](/launch_queue/files/cudagraph_mwe.json)
[CUDAGraph MWE](/launch_queue/files/cudagraph_mwe.py)


Rewrite the section as follows:
1. explain the Microarchitecture diagram

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

- NVIDIA offers a timeline viewer called NSIGHT that's analogous to Kineto, though less tightly coupled to PyTorch. Here's an NSIGHT trace that provides ground truth for the program we studied - it shows Kineto has high fidelity.
![NSIGHT Trace for Kernel Launch](nsight-launch-queue.jpg?raw=true "NSIGHT Trace for Kernel Launch")


- TODO: from Yueming, add NSIGHT traces, understand what is happening there (sending multiple kernels in one shot?)
- TODO: cudnn optimization enable, see if that leads to pytorch matching CUDA code
- TODO: summarize jason/kimish insights into launch overhead
- TODO: see if we can trace PCIE to see how much that contributes and if CUDA graph/CUDA code do group transactions
- TODO: explain need for Kineto and CUPTI - profiler is not enough
-->


![CUDA Launch Queue Trace](/launch_queue/cuda_launch_queue.jpg?raw=true "CUDA Launch Queue Trace")

