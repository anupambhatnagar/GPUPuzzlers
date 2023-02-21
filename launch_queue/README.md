## The CUDA Launch Queue

### Context 

A kernel is a function that's executed on the GPU. When the CPU sends a compute kernel to the GPU, the default behavior is that the CPU won't block on the GPU, i.e., control returns to the CPU before the GPU completes the requested task. This leaves the CPU free for other tasks.


### Problem

This [program](cuda_launch_queue.py) squares 10 100x100 matrices, followed by squaring a 1600x1600 matrix. After a pause, it does the large squaring, followed by the small squarings. The [timeline trace](N=1600-cuda-queue-puzzlers.trace.json), shown below, indicates that it's faster to do the large matrix squaring first - why?

 
```python
# A is a 1600x1600 matrix, the B[j]s are 10x10 matrices.

for j in range(10):
    Br[j] = torch.matmul(B[j],B[j])
Ar = torch.matmul(A,A)
torch.cuda.synchronize()
time.sleep(1e-3)

Ar = torch.matmul(A,A)
for j in range(10):
    torch.matmul(B[j],B[j])
```

#### Kineto Trace
![CUDA Launch Queue Trace](cuda_launch_queue.jpg?raw=true "CUDA Launch Queue Trace")

### Hint

Since CUDA kernel calls don't block on the host, GPU operations must be queued up to be executed by the CUDA driver, as GPU resources become available.

### Solution

To keep the CPU from blocking when it dispatches compute kernels, the GPU maintains a queue of kernel calls - the CUDA launch queue - in the order in which they are made by the host. 

It takes time to launch a kernel - the PyTorch dispatch overhead - and for small kernels this time can dominate the time taken to perform the actual computation on the GPU. In the first case, the GPU completes each small matrix multiply before the next one is ready, so it idles between the small multiplies. 

In the second case, the GPU takes longer to perform the large matrix multiply, so the CUDA launch queue can fill up. After the large matrix multiply finishes, the GPU can immediately turn to the small matrix multiplies.


### Discussion

- If we performed more small matrix multiplications after the large one, we would expect at some point the CUDA launch queue will empty out (since the service rate is higher than the arrival rate). Empirically, this happens if we have 40 or more small matrix multiplications.
- Not letting the CUDA launch queue empty out is a guiding principle in designing high performance PyTorch programs. Some ways to achieve this:
  - Operator fusion, wherein we do more work in a single kernel.
  - Avoiding operations that force the queue to be flushed - a common example is a GPU to CPU copy, which leads to a read-after-write data hazard. (Flushing the queue is analogous to stalling the pipeline in a pipelined processor.)
  - Reordering independent operations to bring the slower operations to the front of the queue (this example).
- This graphic shows the launch queues.
![CUDA Launch Queue Microarchitecture](cuda_launch_queue_uarch.jpg?raw=true "CUDA Launch Queue Microarchitecture")
  - Each queue entry is constrained to be very small: the kernel parameters are limited to 4 KB. It's basically the function pointer, and arguments, which are pointers to tensors. Notably, a host-side tensor cannot be an argument - tensors have to be explicitly copied to and from device.
  - If the CUDA launch queue reaches a threshold (around 1000 entries), the host will block on calling a compute kernel. This can be problematic if there's other work the host could be doing, and should be avoided.
- There are some exceptions to asynchronous kernel launches, notably around CPU-GPU memory copies and synchronization primitives; we'll discuss these in another unit. 
  - Asynchronous launches can be disabled by setting `CUDA_LAUNCH_BLOCKING=1`. This is useful for debugging, especially in the context of multiple streams - [see the CUDA Toolkit Documentation for details](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-execution-host-device).
- In this unit, we're working with a single [stream](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams). In general the GPU maintains multiple queues, one per stream.
<!--- from https://slideplayer.com/slide/8211225/ -->
<!-- launching time breakdown: from above slide 28
software: 
  runtime: resource allocation, e.g., params, streams, kernel information
  driver: sw-hw interactive data structure allocation
hw:
  kernel scheduling
-->
<!-- the 4kb for params is in the slides, also in https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
-->
<!--- see also http://xzt102.github.io/publications/2018_GPGPU_Sooraj.pdf -->
<!--
- TODO: from Yueming, add NSIGHT traces, understand what is happening there (sending multiple kernels in one shot?)
- TODO: cudnn optimization enable, see if that leads to pytorch matching CUDA code
- TODO: summarize jason/kimish insights into launch overhead
- TODO: see if we can trace PCIE to see how much that contributes and if CUDA graph/CUDA code do group transactions
- TODO: explain need for Kineto and CUPTI - profiler is not enough
-->
<!-- launch overhead from notes with taylor
parese args
I 
find right function to interpret pyobj as C__ obj elements (scan signatures)
actual dispatch is like a customm v-table, lots of bitmapping and indirections
index into a list of cuntion pointers
II
thread local -> RDTLC lookup is needed, ~10 libc calls (not just used for profiling) (is is rdtsc?!)
III
cascade of functions (redispatch)
IV
construction of o/p term object
240 bytes for atomic
C = A * B -> make empty (kernel itself allocates vi torch.empty), then inplace add dependency with C.. CU makes host side call to CudaCachingAllocator

PT : 1-2 us
CUDA PT Op: 5+ us not PCIE, more CUDA talking to driver(
also overheads for strides calculation (not device compute)
Fially pyobject also: TF-eager is even worse, 50 us, goes via protobuf to C++
pure CUDA: there's no lookups and strides
--> 
