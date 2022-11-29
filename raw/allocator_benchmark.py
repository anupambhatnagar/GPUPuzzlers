import os
import sys
import time
import gc

import torch
from torch.profiler import profile, record_function, ProfilerActivity

'''
I don't recall exactly what this was for, maybe benchmarking the allocator?
'''

N = 1024

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
