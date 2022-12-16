import time

from scalene import scalene_profiler

import torch
from torch.autograd.profiler import profile

'''
Tried to use the scalene profiler which is supposed to be good
for host side events, but didn't get meaningul results.
'''

N = 10000
num_launches_base = 6

cuda = torch.device('cuda')
torch.cuda.set_sync_debug_mode(1)


s,t,u = [torch.cuda.Stream() for _ in range(3)]

def do_work():
    start_time = time.time()
    for i in range(num_launches_base):
        with torch.cuda.stream(t):
            torch.zeros(N,N).to(cuda)
            #torch.zeros(N,N).to(cuda)
            #torch.empty(N,N).pin_memory().to(cuda)
            #torch.empty(N,N).pin_memory().to(cuda)
            pass
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time

# Warmup.
do_work()

scalene_profiler.start()
do_work()
scalene_profiler.stop()
