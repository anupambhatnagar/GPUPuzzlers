import time

import torch
from torch.autograd.profiler import profile

'''
Measure the performance of copying from pinned host-memory
vs non-pinned host-memory. Compare to PCIE bandwidth to see
how close we get to PCIE limit.
'''

N = 1000
num_launches_base = 1

cpu = torch.device('cpu')
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
            torch.empty(N,N).pin_memory().to(cuda)
            #torch.empty(N,N).pin_memory().to(cuda)

            # pinned copies
            g1 = torch.empty((N,N), device=cuda)
            c1 = torch.empty((N,N), device=cpu).pin_memory()
            # semantics - dest.copy(src) 
            # https://pytorch.org/docs/stable/generated/torch.Tensor.copy_.html
            g1.copy_(c1) # g1 to c1
            c1.copy_(g1)
            c2 = torch.empty((N,N), device=cpu)
            # bouncing copies
            g1.copy_(c2)
            c2.copy_(g1)

            # bouncing copies, non_blocking
            torch.add(g1, g1) # random marker in trace
            g1.copy_(c2)
            c2.copy_(g1)
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
