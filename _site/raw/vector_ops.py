import time

import torch
from torch.autograd.profiler import profile

'''
Achieved flops and mem BW for various numerical operations.
'''

num_launches_base = 4

cuda = torch.device('cuda')


def do_work():
    start_time = time.time()
    N = 20000
    A = torch.ones((N,N), device=torch.device('cuda'))
    B = None
    #B = torch.empty((N,N), device=torch.device('cuda'))
    # inplace updates of A
    A.mul_(0.5)
    time.sleep(1e-4)

    # difference in mallocs between these:
    B = A.mul(0.5)
    time.sleep(1e-4)

    torch.mul(A,0.5, out=B)
    time.sleep(1e-4)

    # difference in mallocs between these:
    B += A
    time.sleep(1e-4)
    B = B + A
    time.sleep(1e-4)
    #print("A = " + str(A))
    #print("B = " + str(B))
    B = torch.sin(A)
    time.sleep(1e-4)

    B = torch.matmul(A, A)
    time.sleep(1e-4)

    torch.matmul(A, A, out=A)
    time.sleep(1e-4)

    B = torch.empty((2,), device=torch.device('cuda'))
    torch.matmul(A, A, out=B)
    time.sleep(1e-4)

    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time

schedule = torch.profiler.schedule(wait=1, warmup=1, active=2)

# Warmup.
do_work()

with profile(use_cuda=True, use_kineto=True, record_shapes=True, 
             with_stack=True, with_flops=True) as p:
    time_taken = do_work()
    print(f"time_taken, flops, mem BW = {time_taken}, "
           "{num_launches_base*N*N/(time_taken*1e9)}, "
           "{num_launches_base*4 * N * N / (time_taken * 1e9)}")


filename = f"./N={N}-flops.json"
p.export_chrome_trace(filename)
print("printed to " + filename)
