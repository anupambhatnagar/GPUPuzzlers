import os
import sys
import time

import torch
from torch.autograd.profiler import profile

'''
Perform some matrix math on CPU.
'''


N = 1000
num_launches_base = 40

# Use different size matrices to identify calls on CPU side in Kineto trace.
A = torch.rand((N,N), device='cpu')
B = torch.rand((N+1,N+1), device='cpu')
C = torch.rand((N-1,N-1), device='cpu')

def do_work(X, Y, Z):
    start_time = time.time()
    for i in range(num_launches_base):
            torch.add(X,X) 
            torch.matmul(X,X) 
            torch.add(X,X) 
            torch.matmul(Y,Y) 
            torch.add(Y,Y) 
            torch.matmul(Y,Y) 
            torch.add(Y,Y) 
    end_time = time.time()
    return end_time - start_time

# Warmup.
do_work(A, B, C)

with profile(use_cuda=False, use_kineto=True, record_shapes=False) as p:
    time_taken = do_work(A, B, C)
print("with profiler = " + str(time_taken))

with profile(use_cuda=False, use_kineto=True, record_shapes=True) as p:
    time_taken = do_work(A, B, C)
print("with profiler & record_shapes = " + str(time_taken))

with profile(use_cuda=False, use_kineto=True, record_shapes=True, with_stack=True) as p:
    time_taken = do_work(A, B, C)
print("with profiler & with_stack & record_shapes = " + str(time_taken))

time_taken = do_work(A, B, C)
print("without profiler = " + str(time_taken))


print("N = " + str(N))

p.export_chrome_trace(f"./cpu-N={N}-cpu.trace.json")
