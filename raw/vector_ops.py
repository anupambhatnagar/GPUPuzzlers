import time

import torch
from torch.autograd.profiler import profile

'''
Multiple variants that show achieved flops and mem BW for
vector ops (i.e., nongemms).
'''

N = 20000
num_launches_base = 4

cuda = torch.device('cuda')

A = torch.ones((N,N), device=torch.device('cuda'))

def do_work():
    start_time = time.time()
    B = None
    #B = torch.empty((N,N), device=torch.device('cuda'))
    for i in range(num_launches_base):
        # inplace updates of A
        # A.mul_(0.5)

        # difference in mallocs between these:
        #B = A.mul(0.5)
        torch.mul(A,0.5, out=B)

        # difference in mallocs between these:
        #B += A
        #B = B + A
        #print("A = " + str(A))
        #print("B = " + str(B))
        #B = torch.sin(A)
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time

schedule = torch.profiler.schedule(wait=1, warmup=1, active=2)

# Not sure if profile_memory is doing anything?
with profile(use_cuda=True, use_kineto=True, record_shapes=True, 
             with_stack=True, profile_memory=True, with_flops=True) as p:
    for _ in range(2):
        time_taken = do_work()
        print(f"time_taken, flops, mem BW = {time_taken}, "
               "{num_launches_base*N*N/(time_taken*1e9)}, "
               "{num_launches_base*4 * N * N / (time_taken * 1e9)}")

print("with profiler & with_stack & record_shapes = " + str(time_taken))

print("N = " + str(N))

filename = f"./N={N}-vector_perf.json"
p.export_chrome_trace(filename)
print("printed to " + filename)
