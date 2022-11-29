import os
import sys
import time

import torch
from torch.autograd.profiler import profile

'''
Shows how queuing order and RAW hazards impact performance.
'''

N = 1600
M = 100
num_launches_base = 1
DELAY = 1e-2 # In seconds.

cuda = torch.device('cuda')

# set_sync_debug_mode() is a recent experimental feature; it can 
# help debug the challenges we'll be looking at in this exercise.
# torch.cuda.set_sync_debug_mode(1)

# Allocate a big and small matrix, and a small 1d tensor:
A = torch.rand((N,N), device='cuda')
B = torch.rand((M,M), device='cuda')
C = torch.rand(M, device='cuda')

s = torch.cuda.Stream()

def do_work(X, Y, Z):
    start_time = time.time()
    for i in range(num_launches_base):
        with torch.cuda.stream(s):

            # Challenge 1: which block is faster?

            # Block 1
            for j in range(10):
                torch.matmul(B,B)
            torch.matmul(A,A) 
            torch.cuda.synchronize()
            time.sleep(DELAY)

            # Block 2: reverse order of big gemm and small gemms
            torch.matmul(A,A) 
            for j in range(10):
                torch.matmul(B,B)
            time.sleep(DELAY)
            torch.cuda.synchronize()

            # Challenge 2: why is second set of small gemms slower?
            torch.matmul(A,A) 
            # Set 1
            for j in range(10):
                torch.matmul(B,B)
            C.to('cpu')
            # Avoid the stall by using non_blocking=True (but opens
            # you up to potential races).
            # More details here: https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
            #C.to('cpu', non_blocking=True)

            # Set 2
            for j in range(10):
                torch.matmul(B,B)
            time.sleep(DELAY)
            torch.cuda.synchronize()

    C.cumsum(dim=0)
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time

# Warmup.
do_work(A, B, C)

with profile(use_cuda=True, use_kineto=True, 
                record_shapes=True, with_stack=True) as p:
    time_taken = do_work(A, B, C)

print(f"Time with profiler & with_stack & record_shapes = {str(time_taken)}")

filename = f"./N={N}-cuda-queue-puzzlers.trace.json"
p.export_chrome_trace(filename)
print("printed to " + filename)
