import os
import sys
import time

import torch
from torch.autograd.profiler import profile

# Disable tensorcore so that it doesn't cause streams to block.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

N = 50000
#num_launches_base = 400
num_launches_base = 6

cuda = torch.device('cuda')
torch.cuda.set_sync_debug_mode(1)

# Create streams that we'll use to execute in parallel. This is not a 
# common use case: usually everything is executed in the default stream, 
# which has id 7.)

s,t,u = [torch.cuda.Stream() for _ in range(3)]

# Use different size matrices to identify calls on CPU side in Kineto trace.
A = torch.rand((N,N), device='cuda')
B = torch.rand((N+1,N+1), device='cuda')
C = torch.rand((N-1,N-1), device='cuda')
tmp = torch.rand(3, device="cuda")

def do_work(X, Y, Z):
    start_time = time.time()
    for i in range(num_launches_base):
        with torch.cuda.stream(s):
            #X = 1.4 * X
            #Y = 1.4 * X
            #Z = 1.4 * X
            #torch._foreach_mul_((X,Y,Z), 1.4)
            #torch.add(X,X) 
            #torch.matmul(X,X) 
            #torch.add(X,X) 
            pass
        with torch.cuda.stream(t):
            #X = 1.4 * X
            #Y = 1.4 + X
            #Z = 1.4 * X
            #torch.matmul(Y,Y) 
            #torch.matmul(Y,Y) 
            #torch.add(Y,Y) 
            torch.empty(N,N).pin_memory().to(cuda)
            torch.empty(N,N).pin_memory().to(cuda)
            pass
        with torch.cuda.stream(u):
            #torch.matmul(Z,Z) 
            #torch.add(Z,Z) 
            #torch.zeros(N,N).to(cuda)
            #torch.zeros(N,N).pin_memory().to(cuda)
            torch.empty(N,N).pin_memory().to(cuda)
            torch.empty(N,N).pin_memory().to(cuda)
            #torch.randn(N,N).to(cuda)
            #torch.randn(N,N).pin_memory().to(cuda)
        if i % 10 == 0:
            # Not calling sync
            tmp = (torch.sum(X), torch.sum(Y), torch.sum(Z))
            # Does call sync
            #tmp = torch.tensor([torch.sum(X), torch.sum(Y), torch.sum(Z)])
            # Does call sync, and is v slow
            # tmp = X.tolist()
            # Does call sync, but is less slow (and I cannot see it in the trace?)
            # tmp = X[0].tolist()
            pass
            #torch.sum(X)
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

p.export_chrome_trace(f"./N={N}-with_stack-with_record_shapes.trace.json")
