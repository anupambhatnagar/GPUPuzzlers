import os
import sys
import time
import gc

import torch
from torch.profiler import profile, record_function, ProfilerActivity

#from torch.autograd.profiler import profile

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
# We are turning off so as to check whether big gemms achieve peak flops by looking at 
# duration of gemms in the trace.
torch.backends.cuda.matmul.allow_tf32 = False

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False

# With a large N (e.g., 32K) gemms approach fp32 roofline of 19.5 TFLOPS.
N = 1024
M = 2000
#M = 1

T = []
print("hello!")


T.append(("0 profile start", time.time()))
A = torch.rand((N,N), device='cuda') 
torch.cuda.synchronize() 
print(torch.cuda.memory_allocated())
B = torch.rand((N,N), device='cuda') 
gc.disable()
print(torch.cuda.memory_allocated())
A = B
print(torch.cuda.memory_allocated())

print(torch.cuda.max_memory_allocated())
#print(torch.cuda.memory_stats())
print(torch.cuda.memory_snapshot())

with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    A = torch.rand((N,N), device='cuda') 

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


def timeline(T):
    res = ""
    for i in range(1, len(T)):
        res = res + " " + "{:.2E}".format((T[i][1] - T[i-1][1])) + " " + T[i][0] + "; "
    return res
        

def delta(T):
    res = []
    for i in range(1, len(T)):
        res.append(((T[i][1] - T[i-1][1]), T[i][0]))
    return res
        
sys.exit(0)
