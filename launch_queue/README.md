## Understanding the CUDA Launch Queue

### Context 

When the CPU sends a kernel to the GPU, it doesn't bock on the GPU, i.e., control returns to the host thread before the GPU completes the requested task. This means many GPU operations can be queued up to be executed by the CUDA driver, as GPU resources become available, which leaves the CPU free for other tasks. (There are some exceptions to this asynchronicity, notably around CPU-GPU memory copies and synchronization primitives; we'll discuss these in another unit.)

### Problem

This [program](cuda_launch_queue.py) squares 10 100x100 matrices, followed by squaring a 1600x1600 matrix. After a pause, it does the large squaring, followed by the small squarings. The [timeline trace](N=1600-cuda-queue-puzzlers.trace.json), shown below, indicates that it's faster to do the large matrix squaring first - why?

 
```python
# A is a 1600x1600 matrix, the B[i]s are 10x10 matrices.

for j in range(10):
    Br[j] = torch.matmul(B[j],B[j])
Ar = torch.matmul(A,A)
torch.cuda.synchronize()
time.sleep(1e-3)

Ar = torch.matmul(A,A)
for j in range(10):
    torch.matmul(B[j],B[j])
```

![CUDA Launch Queue Trace](cuda_launch_queue.jpg?raw=true "CUDA Launch Queue Trace")

### Hint

Since CUDA kernel calls don't block on the host, the GPU must buffer calls.

### Solution

To keep the CPU from blocking when it dispatches compute kernels, the GPU maintains a queue of kernel calls - the CUDA launch queue - in the order in which they are made by the host. 

It takes time to launch a kernel - the CPU has to initiate a PCI-E transaction with the GPU - and this time can dominate the time taken to perform the actual computation on the GPU. In the first case, the GPU completes each small matrix multiply before the next one is ready, so it idles between the small multiplies. 

In the second case, the GPU takes longer to perform the large matrix multiply, so the CUDA launch queue can fill up. After the large matrix multiply finishes, the GPU can immediately turn to the small matrix multiples.

<!--- from https://slideplayer.com/slide/8211225/ -->
<!--- see also http://xzt102.github.io/publications/2018_GPGPU_Sooraj.pdf ->
This graphic shows the launch queues.
![CUDA Launch Queue Microarchitecture](cuda_launch_queue_uarch.jpg?raw=true "CUDA Launch Queue Microarchitecture")


### Discussion

- If we had a very large number of small matrix multiplications after the large one, we would expect at some point the CUDA launch queue will empty out (since the service rate is higher than the arrival rate). This is exactly what happens if we have 40 or more small matrix multiplications.
- Not letting the CUDA launch queue empty out is a guiding principle in designing high performance PyTorch programs. Some ways to achieven this:
  - Operator fusion, wherein we do more work in a single kernel (this will be the subject of a different unit)
  - Reordering independent operations, do bring the slower operations to the front of the queue
- Technically, the GPU maintains multiple queues, one per stream, but we can ignore that in this single stream case.
- Each queue entry is constrained to be very small: under 1 KB. It's basically the function pointer, and arguments, which are pointers to tensors. Notably, a host-side tensor cannot be an argument - tensors have to be explicitly copied to and from device.
- If the CUDA launch queue reaches 1000 entries, the host will block on calling a compute kernel. This can be problematic if there's other work the host could be doing, and should be avoided.
