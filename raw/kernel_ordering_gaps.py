import os
import sys
import time

import torch
from torch.autograd.profiler import profile

# Disable tensorcore so that it doesn't cause streams to block.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

N = 10000
M = 100
#num_launches_base = 400
num_launches_base = 1

cuda = torch.device('cuda')
torch.cuda.set_sync_debug_mode(1)

# Create streams that we'll use to execute in parallel. This is not a 
# common use case: usually everything is executed in the default stream, 
# which has id 7.)

s,t,u = [torch.cuda.Stream() for _ in range(3)]

# Use different size matrices to identify calls on CPU side in Kineto trace.
A = torch.rand((N,N), device='cuda')
B = torch.rand((M,M), device='cuda')
C = torch.rand(N*N, device='cuda')

tmp = torch.rand(3, device="cuda")

def do_work(X, Y, Z):
    start_time = time.time()
    for i in range(num_launches_base):
        with torch.cuda.stream(s):
            torch.matmul(A,A) 
            for j in range(10):
                torch.matmul(B,B)
            C.to('cpu')
            for j in range(10):
                torch.matmul(B,B)
            time.sleep(0.2)
            for j in range(20):
                torch.matmul(B,B)
            torch.matmul(A,A) 
            time.sleep(0.2)
    C.cumsum(dim=0)
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time

# Warmup.
do_work(A, B, C)

'''
with profile(use_cuda=True, use_kineto=True, record_shapes=False) as p:
    time_taken = do_work(A, B, C)
print("with profiler = " + str(time_taken))

with profile(use_cuda=True, use_kineto=True, record_shapes=True) as p:
    time_taken = do_work(A, B, C)
print("with profiler & record_shapes = " + str(time_taken))

'''
with profile(use_cuda=True, use_kineto=True, record_shapes=True, with_stack=True) as p:
    time_taken = do_work(A, B, C)
print("with profiler & with_stack & record_shapes = " + str(time_taken))

#time_taken = do_work(A, B, C)
#print("without profiler = " + str(time_taken))


print("N = " + str(N))

filename = f"./N={N}-gaps.trace.json"
p.export_chrome_trace(filename)
print("printed to " + filename)
