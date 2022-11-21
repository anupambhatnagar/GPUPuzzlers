import time

from scalene import scalene_profiler

import torch
from torch.autograd.profiler import profile

# # Disable tensorcore so that it doesn't cause streams to block.
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

N = 10000
num_launches_base = 6

cuda = torch.device('cuda')
torch.cuda.set_sync_debug_mode(1)

# Create streams that we'll use to execute in parallel. This is not a 
# common use case: usually everything is executed in the default stream, 
# which has id 7.)

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
