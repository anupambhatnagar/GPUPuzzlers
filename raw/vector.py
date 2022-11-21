import time
import gc

import GPUtil

import torch
from torch.autograd.profiler import profile
#from torch.profiler import profile

# # Disable tensorcore so that it doesn't cause streams to block.
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

N = 20000
num_launches_base = 4

cuda = torch.device('cuda')
#torch.cuda.set_sync_debug_mode(1)

# Create streams that we'll use to execute in parallel. This is not a 
# common use case: usually everything is executed in the default stream, 
# which has id 7.)

s,t,u = [torch.cuda.Stream() for _ in range(3)]

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

with profile(use_cuda=True, use_kineto=True, record_shapes=True, with_stack=True, profile_memory=True, with_flops=True) as p:
    for _ in range(2):
        time_taken = do_work()
        print(f"time_taken, flops, mem BW = {time_taken}, {num_launches_base*N*N/(time_taken*1e9)}, {num_launches_base*4 * N * N / (time_taken * 1e9)}")

print("with profiler & with_stack & record_shapes = " + str(time_taken))

print("N = " + str(N))

filename = f"./N={N}-vector.json"
p.export_chrome_trace(filename)
print("printed to " + filename)
