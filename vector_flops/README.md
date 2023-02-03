## Flops

### Context 

Marketing literature for GPUs stresses their high FLOPs. An A100 is advertised as performing 19.5 fp32 TFLOPs - by contrast, the Intel SKL CPU that's used in T1s comes in at just 921.6 GFLOPs.


### Problem

This [program](flops_bw.py) does performs a number of numerical operations on tensors: addition, scalar multiplication, transcendental functions, and matrix multiplications. The [timeline trace](N=flops.trace.json), shown below, indicates that other than matrix multiplication, all of the other operations are a tiny fraction (~1%) of the advertised flops. Furthermore, simple operations like addition take exactly as long as complex ones like sine and log. Why?
<table>
<tr>
<td>
```python
    def sync_and_pause():
        torch.cuda.synchronize()
        time.sleep(DELAY)

    N = 20000
    A = torch.ones((N,N), device=torch.device('cuda'))
    
    A.mul_(0.5)
    sync_and_pause()
    
    B = A.mul(0.5)
    sync_and_pause()
    
    B += A
    sync_and_pause()
    
    C = B + A
    sync_and_pause()
```
</td>
<td>
```python
    B = torch.sin(A)
    sync_and_pause()
    
    B = torch.sigmoid(A)
    sync_and_pause()
    
    torch.sqrt(A, out=B)
    sync_and_pause()
    
    torch.log10(A, out=B)
    sync_and_pause()

    torch.pow(A, 3.14159, out=B)
    sync_and_pause()

    B = torch.matmul(A, A)
    sync_and_pause()
```
</td>
</tr>
</table>

#### Kineto Trace
![Assorted Flops](assorted_flops.jpg?raw=true "Assorted Flops")

### Hint

[Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law): "the overall performance improvement gained by optimizing a single part of a system is limited by the fraction of time that the improved part is actually used".

### Solution

Before the GPU can perform numerical operations can on data, that data has to first be read into registers. The peak memory bandwidth for an A100 is ~2 TB/sec. This means that a 20000x20000 tensor of fp32 values takes at least (4x20000^2)/2e12 = 0.8ms. Therefore, even if numerical operations were infinitely fast, the time to perform an elementary operation like multiply by scalar is at least 1.6ms (read the data, write back the updated data), effectively 0.25 TFLOPs, i.e,. 1.3% of the peak.

This is why all the unary operations take roughly the same time - the number of adds and multiplies is irrelevant. The binary ops take 50% longer, since there's 2 Reads and 1 Write per result.

Matrix multiplication does infact achieve close to advertised performance - 19.1 TFLOPs. This is because once we read in a value, we operate on it multiple times, thereby amortizing the cost of the read.


### Discussion

- The ratio of flops performed by an operation to the bytes read/written is known as the compute intensity of the operation. 
 - For the unary and binary operations we saw, the flop count is O(n), where n is the number of tensor entries; the bytes read/written is also O(n) so the compute intensity is O(1).
 - For matrix multiplication, the flop count is O(n^3), so the compute intensity is O(n^2).
- The advertised memory bandwidth is misleading: it's for best-case memory layouts, specifically when copies are aligned with cache dimensions. It assumes wide transfers, i.e., copying long contiguous segments of memory - this is true when dealing with tensors, but not for applications like sorting.
- Within Meta, the top kernel types are vectorized functors (as we saw above), followed by embedding bag lookups, followed by matrix multiplication.
![CUDA Launch Queue Microarchitecture](cuda_launch_queue_uarch.jpg?raw=true "CUDA Launch Queue Microarchitecture")
<!--- from https://slideplayer.com/slide/8211225/ -->
<!--- see also http://xzt102.github.io/publications/2018_GPGPU_Sooraj.pdf -->

