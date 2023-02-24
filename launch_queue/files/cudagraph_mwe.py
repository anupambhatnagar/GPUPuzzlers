import time

import torch
from torch.autograd.profiler import profile

def do_work():
    a = torch.zeros((1600, 1600), device='cuda')
    B = [torch.zeros((100, 100), device='cuda') for i in range(10)]
    c = None
    D = [None for _ in range(len(B))]

    # Needed for warmup.
    for i in range(len(B)):
        D[i] = torch.matmul(B[i], B[i])
    c = torch.matmul(a, a)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for j in range(len(B)):
            D[j] = torch.matmul(B[j], B[j])
        c = torch.matmul(a, a)

    Y = torch.tensor((1600, 1600), device='cuda').pow(3.14)
    time.sleep(0.001)

    tmp1 = torch.ones((1600, 1600), device='cuda')
    tmp2 = [torch.zeros((100, 100), device='cuda').add(i * 1.01) for i in range(10)]

    for _ in range(20):
        a.copy_(tmp1)
        for i in range(len(B)):
            B[i].copy_(tmp2[i])
        g.replay()
        # c and D are updated 

    time.sleep(0.001)
    torch.cuda.synchronize()
    
with profile(use_cuda=True, use_kineto=True, record_shapes=True,
             with_stack=True, with_flops=True) as p:
    time_taken = do_work()

filename = f"./cudagraph_mwe.json"
p.export_chrome_trace(filename)
