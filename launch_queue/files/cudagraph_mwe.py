import time

import torch
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity

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
    
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             on_trace_ready=tensorboard_trace_handler(dir_name=f"./cudagraph_mwe", use_gzip=True),
             record_shapes=True,
             with_stack = True) as prof:
    time_taken = do_work()
