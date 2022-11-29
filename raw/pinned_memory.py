import time

import torch
from torch.autograd.profiler import profile

'''
Measure the performance of copying from pinned host-memory
vs non-pinned host-memory. Compare to PCIE bandwidth to see
how close we get to PCIE limit.
'''

N = 1000
num_launches_base = 2

cuda = torch.device('cuda')
torch.cuda.set_sync_debug_mode(1)

# Create multiple streams that we potentially use to see if 
# parallel copies help (they don't, since PCIE is bottleneck).
s,t,u = [torch.cuda.Stream() for _ in range(3)]

def do_work():
    start_time = time.time()
    for i in range(num_launches_base):
        with torch.cuda.stream(t):
            torch.zeros(N,N).to(cuda)
            # Each of these variants shows different performance.
            # It's better to use torch.empty() since torch.zeros()
            # or torch.random() are relatively expensive.
            #torch.zeros(N,N).to(cuda)
            #torch.empty(N,N).pin_memory().to(cuda)
            #torch.empty(N,N).pin_memory().to(cuda)
            pass
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time

# Warmup.
do_work()

with profile(use_cuda=True, use_kineto=True, record_shapes=True, with_stack=True) as p:
    time_taken = do_work()
print("with profiler & with_stack & record_shapes = " + str(time_taken))

print("N = " + str(N))

p.export_chrome_trace(f"./N={N}-bounce.json")
