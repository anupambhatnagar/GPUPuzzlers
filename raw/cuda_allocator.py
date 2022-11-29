import time
import gc

import torch
from torch.autograd.profiler import profile

'''
Proto-attempt to see what allocations look like with the cuda
caching allocator.
'''

N = 20000
num_launches_base = 100

cuda = torch.device('cuda')

def do_work():
    start_time = time.time()
    for i in range(num_launches_base):
        B = torch.empty((N,N), device=torch.device('cuda'))
        del(B)
        #torch.cuda.empty_cache()
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time

# Warmup.
do_work()

schedule = torch.profiler.schedule(wait=1, warmup=1, active=2)

with profile(use_cuda=True, use_kineto=True, record_shapes=True, 
             with_stack=True, profile_memory=True, with_flops=True) as p:
    for _ in range(2):
        time_taken = do_work()
        print(f"time_taken, mem alloc BW = {time_taken},"
               "{num_launches_base*N*N/(time_taken*1e9)}")

print("with profiler & with_stack & record_shapes = " + str(time_taken))

print("N = " + str(N))

p.export_chrome_trace(f"./N={N}-cuda-allocator.json")
