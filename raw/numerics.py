import os
import sys
import time

import torch
from torch.autograd.profiler import profile

torch.use_deterministic_algorithms(False)

# Disable tensorcore so that it doesn't cause streams to block.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

N = 5000
M = 16
num_launches_base = 100

s,t,u = [torch.cuda.Stream() for _ in range(3)]

cuda = torch.device('cuda')

def do_cumsum():
    base = torch.rand((M,N,N), device=cuda)
    A = torch.clone(base)
    B = torch.clone(base)
    C1 = torch.cumsum(A, dim=0)
    C2 = torch.cumsum(B, dim=0)
    for i in range(len(C1)):
        D = 1000000.0*(C1 - C2)
        print(f"min, max, mean(abs) = {torch.min(D)}, {torch.max(D)}, {torch.mean(torch.abs(D))}")

def do_bmm():
    base = torch.rand((M,N,N), device=cuda)
    A = torch.clone(base)
    B = torch.clone(base)
    C1 = torch.bmm(A, B) 
    C2 = torch.bmm(A, B)
    for i in range(len(C1)):
        D = 1000000.0*(C1 - C2)
        print(f"min, max, mean(abs) = {torch.min(D)}, {torch.max(D)}, {torch.mean(torch.abs(D))}")
    

def do_work():
    start_time = time.time()
    A = torch.rand((N,N), device='cuda')
    B = torch.clone(A)
    C = torch.clone(A)
    with torch.cuda.stream(s):
        for i in range(num_launches_base):
            A = torch.sin(A*A + A)
    with torch.cuda.stream(t):
        for i in range(num_launches_base):
            B = torch.sin(B*B + B)
    torch.cuda.synchronize()
    D = 1000000.0*(A - B)
    print(f"min, max, mean(abs) = {torch.min(D)}, {torch.max(D)}, {torch.mean(torch.abs(D))}")
    end_time = time.time()
    return D

do_cumsum()
