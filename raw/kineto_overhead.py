import os
import sys
import time

import torch

from torch.autograd.profiler import profile

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
# We are turning off so as to check whether big gemms achieve peak flops by looking at 
# duration of gemms in the trace.
torch.backends.cuda.matmul.allow_tf32 = False

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False

# With a large N (e.g., 32K) gemms approach fp32 roofline of 19.5 TFLOPS.
N = 1024
M = 50000
#M = 1

T = []


T.append(("0 profile start", time.time()))
A = torch.rand((N,N), device='cuda') 
torch.cuda.synchronize() 
time.sleep(2)
T.append(("0 profile after init arrays", time.time()))
for _ in range(M):
    A = torch.matmul(A,A) + A

#T.append(("before sync", time.time()))
torch.cuda.synchronize() 
T.append(("0 after sync", time.time()))

print(A)

T.append(("before profile start", time.time()))
with profile(use_cuda=True, use_kineto=False, record_shapes=False) as p:
    T.append(("profile start", time.time()))
    A = torch.rand((N,N), device='cuda') 
    torch.cuda.synchronize() 
    T.append(("profile after init arrays", time.time()))
    for _ in range(M):
        A = torch.matmul(A,A) + A
    
    #T.append(("before sync", time.time()))
    torch.cuda.synchronize() 
    T.append(("after sync", time.time()))

print(A)

T.append(("2 profile start", time.time()))
A = torch.rand((N,N), device='cuda') 
torch.cuda.synchronize() 
T.append(("2 profile after init arrays", time.time()))
for _ in range(M):
    A = torch.matmul(A,A) + A

#T.append(("before sync", time.time()))
torch.cuda.synchronize() 
T.append(("2 after sync", time.time()))

print(A)

T.append(("3 profile start", time.time()))
A = torch.rand((N,N), device='cuda') 
torch.cuda.synchronize() 
T.append(("3 profile after init arrays", time.time()))
for _ in range(M):
    A = torch.matmul(A,A) + A

#T.append(("before sync", time.time()))
torch.cuda.synchronize() 
T.append(("3 after sync", time.time()))

print(A)

def timeline(T):
    res = ""
    for i in range(1, len(T)):
        res = res + " " + T[i][0] + " " + "{:.2E}".format((T[i][1] - T[i-1][1])) + "; "
    return res
        

print(timeline(T))

try:
    os.mkdir("result")
except Exception:
    pass
p.export_chrome_trace("./result/kineto_overhead.pt.trace.json")

sys.exit(0)
