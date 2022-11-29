import time
import gc

# GPUtil is an OSS library that can be used to 
# get nvidia-smi stats programmatically.
#import GPUtil

import torch
from torch.autograd.profiler import profile

'''
Profile and show stats in the presence of leaks.
'''

N = 50000
num_launches_base = 100

cuda = torch.device('cuda')
s,t,u = [torch.cuda.Stream() for _ in range(3)]

A = []

def do_work():
    start_time = time.time()
    for i in range(num_launches_base):
        x = torch.zeros(N,N)
        x.to(cuda)
        A.append(x)
        #GPUtil.showUtilization()
        print(torch.cuda.memory_allocated(), 
              torch.cuda.max_memory_allocated(), 
              torch.cuda.max_memory_reserved() )
        pass
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time

# Warmup.
do_work()

schedule = torch.profiler.schedule(wait=1, warmup=1, active=2)

with profile(use_cuda=True, use_kineto=True, record_shapes=True, with_stack=True) as p:
    for _ in range(2):
        time_taken = do_work()

print("time taken with profiler & with_stack & record_shapes = " + str(time_taken))

print("N = " + str(N))
p.export_chrome_trace(f"./N={N}-leak.json")
