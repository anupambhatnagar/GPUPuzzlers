---
layout: page
title: Answer
permalink: /swimming-in-streams-answer/
---

The key difference between `non_blocking_streams` and `blocking_streams` functions is that the former sets `non_blocking=True` whereas the latter sets `non_blocking=False`.

```python
with torch.cuda.stream(s1):
    for i in range(len(matrix_on_gpu)):
        torch.matmul(matrix_on_gpu[i], matrix_on_gpu[i])

with torch.cuda.stream(s2):
    for i in range(len(data_on_gpu)):
        data_on_gpu[i].to(cpu, non_blocking=True)
```

In both functions, stream *s1* executes some computation kernels and stream *s2* transfers data from the device to the host. When `non_blocking=True` the stream utilizes page-locked (pinned) memory whereas when `non_blocking=False` it uses pageable memory. As expected, the data transfer is much faster when page-locked memory is used (33 ms) in comparison to when pageable memory (78 ms) is used.
While transferring data from the device to the host both streams overlap.

```python
with torch.cuda.stream(s3):
    for i in range(len(data_on_cpu)):
        data_on_cpu[i].to(cuda, non_blocking=True)
```

Stream *s3* transfers tensors from the host to the device which can be overlapped with other data transfers only when page-locked host memory is used. Thus, all three streams overlap in the `non_blocking_streams` function whereas only *s1* and *s2* overlap in the `blocking_streams` function.


## Discussion

TODO: Show how to measure overlap using HTA. Add trace files and notebooks which implement both functions independently.


__Which operations can run concurrently on the GPU with one another?__

- Computation on the host
- Computation on the device
- Memory transfers from the host to the device
- Memory transfers from the device to the host
- Memory transfers within the memory of a device
- Memory transfers among devices

__Which operations are asynchronous with respect to the host?__

- Kernel launches
- Memory copies within a single device's memory
- Memory copy from host to device of a memory block if 64KB or less
- Memory copies performed by functions that are suffixed with Async (e.g. cudaMemcpyAsync,
  cudaMemcpyPeerAsync, cudaMemsetAsync etc.)
- Memory set function calls (cudaMemset)

__How many streams can be launched simultaneously?__

At most 128 kernels can run concurrently on the P100, V100, A100 and H100 GPUs. Assuming each
kernel is executing on a unique stream, the number of streams that can run concurrently is 128.
In ML applications achieving concurrent execution of more than a few streams is difficult to achieve as each kernel should run long enough to achieve overlap and not saturate the GPU memory and CUDA/Tensor cores.

__Can data transfers to and from the device be overlapped?__

Yes, assuming the GPU has the capability. In particular, the GPU must have `asyncEngineCount` equal
to 2.  This value can be checked using the code snippet below:

```cpp
int deviceCount; cudaGetDeviceCount(&deviceCount);
int device;

for (device = 0; device < deviceCount; ++device) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  printf("Number of asynchronous engines %d", deviceProp.asyncEngineCount);
}
```

Additionally, *any host memory involved must be page-locked.*

__Can data transfer and kernel execution be overlapped?__

Yes, assuming the GPU has `asyncEngineCount` equal to 2. *If host memory is involved in the data transfer it must be
page-locked.*

__How can kernels across streams be synchronized?__

There are various ways to *explicitly* synchronize streams with each other.

1. `cudaDeviceSynchronize()` - waits until all preceding commands in all streams of all host threads
   have completed.
1. `cudaStreamSynchronize()` - takes a stream as a parameter and waits until all preceding commands
   in the given stream have completed. This can be used to synchronize the host with a specific
   stream allowing other streams to continue executing on the device.
1. `cudaStreamWaitEvent()` - takes a stream and event as parameters and makes the stream wait for the
   event to complete.
1. `cudaStreamQuery()` - provides applications to query an asynchronous stream for completion
   status.

This table provides a mapping between the PyTorch and CUDA API for the explicit synchronization
function calls

| CUDA | PyTorch |
| ---  | ---     |
| cudaDeviceSynchronize | torch.cuda.synchronize |
| cudaStreamSynchronize | torch.cuda.Stream.synchronize |
| cudaStreamWaitEvent   | torch.cuda.Stream.wait_event|
| cudaStreamQuery       | None |

__Can streams block each other?__

Yes, if any of the following operations is issued in-between them by the host thread:

- Pinned host memory allocation
- Device memory allocation
- Device memory set
- Memory copy between two addresses to the same device memory
- Any CUDA command to the default (a.k.a. NULL) stream
- A switch between the L1/shared memory configurations

In other words, the operations above cause implicit synchronization.

__Can Tensor cores and CUDA cores be used simultaneously on different streams?__

Here's a code snippet demostrating the simultaneous use of both Tensor cores and CUDA cores.

```python
def streams():
    torch.backends.cuda.matmul.allow_tf32 = True
    with torch.cuda.stream(s1):
        for i in range(len(matrix_on_gpu)):
            torch.matmul(matrix_on_gpu[i], matrix_on_gpu[i])

    with torch.cuda.stream(s2):
        for i in range(len(matrix_on_gpu)):
            torch.pow(matrix_on_gpu[i], 3)

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
cuda = torch.device("cuda")

matrix_on_gpu = [torch.rand((1024, 1024), device=cuda) for _ in range(1000)]
```

## What should you remember in 10 years?

Streams provide a great mechanism to execute kernels concurrently but watch out for implicit
synchronization events as they can be major blockers to achieving concurrency.

## Explore More

Like queues, streams can be assigned a priority. Stream priority can be set in PyTorch when the
stream is initialized. E.g. `s = torch.cuda.Stream(priority=1)`. The possible values are 1 (high
priority) and 0 (low priority). By default, streams have priority 0.

In the streams function above set the priority for *s2* to 1 and see how the kernel execution
changes.
