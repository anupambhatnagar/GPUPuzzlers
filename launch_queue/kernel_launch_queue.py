import time
import torch
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity

def launch_kernels(large_matrices, small_matrices):
    # Block 1: execute small matmuls followed by large matmul
    for i in range(10):
       torch.matmul(small_matrices[i], small_matrices[i])
    torch.matmul(large_matrices[0], large_matrices[0])

    torch.cuda.synchronize()
    time.sleep(delay)

    # Block 2: execute large matmul followed by small matmuls
    torch.matmul(large_matrices[1], large_matrices[1])
    for j in range(10, 20):
       torch.matmul(small_matrices[j], small_matrices[j])


# set matrix size and device
large = 2**13
small = 2**6
delay = 0.001 # in seconds
cuda = torch.device('cuda')

# create small and large matrices
large_matrices = []
for _ in range(2):
    large_matrices.append(torch.rand((large, large), device='cuda'))

small_matrices = []
for _ in range(20):
    small_matrices.append(torch.rand((small, small), device='cuda'))

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             on_trace_ready=tensorboard_trace_handler(dir_name=f"./log_{large}_{small}"),
             record_shapes=True,
             with_stack = True) as prof:

    launch_kernels(large_matrices, small_matrices)
