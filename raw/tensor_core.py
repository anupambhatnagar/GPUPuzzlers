import os
import sys
import time

import torch
from torch.autograd.profiler import profile

''' 
Shows the relative difference in performance across fp32, tf32 and fp16 gemms.
TODO: show how to convert types, talk to complexity of that. 
(Include something with int8 tensors with scale and offset?)
'''

#N = 4096
N = 10000
#num_launches_base = 200
num_launches_base = 200

cuda = torch.device('cuda')
torch.cuda.set_sync_debug_mode(1)

s,t = [torch.cuda.Stream() for _ in range(2)]

A = torch.rand((N,N), device='cuda')
Ad = torch.rand((N,N), device='cuda')
B = torch.rand((N,N), device='cuda').to(dtype=torch.float16)
Bd = torch.rand((N,N), device='cuda').to(dtype=torch.float16)
C = torch.rand((N,N), device='cuda').to(dtype=torch.bfloat16)
Cd = torch.rand((N,N), device='cuda').to(dtype=torch.bfloat16)

def do_work():
    start_time = time.time()
    with torch.cuda.stream(s):
        for i in range(num_launches_base):
            torch.backends.cuda.matmul.allow_tf32 = False
            #torch.matmul(A,A) 
            torch.matmul(B,B) 
            #torch.matmul(C,C) 
            pass
    with torch.cuda.stream(t):
        for i in range(num_launches_base):
            torch.backends.cuda.matmul.allow_tf32 = True
            #torch.matmul(Ad,Ad) 
            #torch.matmul(Bd,Bd) 
            #torch.matmul(Cd,Cd) 
            pass
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time

# Warmup.
do_work()

with profile(use_cuda=True, use_kineto=True, record_shapes=True, with_stack=True) as p:
    time_taken = do_work()
print("with profiler & with_stack & record_shapes = " + str(time_taken))

filename = f"./N={N}-tensor_core.trace.json"
p.export_chrome_trace(filename)
print("printed to " + filename)
