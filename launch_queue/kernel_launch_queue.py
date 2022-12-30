import time
import torch
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity


def launch_kernels(large_matrices, small_matrices):
    # Block 1: execute small matmuls followed by large matmul
#    block1_start = time.time_ns()
    for i in range(10):
       torch.matmul(small_matrices[i], small_matrices[i])
    torch.matmul(large_matrices[0], large_matrices[0])
#    block1_end = time.time_ns()

    # sleep
    torch.cuda.synchronize()
    time.sleep(delay)

    # Block 2: execute large matmul followed by small matmuls
#    block2_start = time.time_ns()
    torch.matmul(large_matrices[1], large_matrices[1])
    for j in range(10, 20):
       torch.matmul(small_matrices[j], small_matrices[j])
#    block2_end = time.time_ns()
#
#    print(f"block 1 start = {block1_start}, block 1 end = {block1_end}, block 2 start = {block2_start}, block 2 end = {block2_end}")
#    print(f"Block 1 takes = {block1_end - block1_start}, Block 2 takes = {block2_end - block2_start} ")


large = 2**12
small = 2**6
delay = 0.001 # in seconds
cuda = torch.device('cuda')

# Create a big and small matrix
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
