import time
import random 
import numpy as np

import torch
from torch.autograd.profiler import profile

# Disable tensorcore so that it doesn't cause streams to block.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

N = 1000

# Mimicing host-side IO/preprocessing.
def create_sorted_random_tensor(size):
    L = []
    L = np.random.rand(N,N)
    L.sort(0)
    result = torch.tensor(L)
    return result

# Mimicing host-side simple statistics generation.
def gen_stats(T):
    T_np = T.numpy()
    total = 0.0
    for _ in range(10):
        # Get a random entry per row, and sum them up
        sample_indices = np.random.randint(0, high=N, size=N)
        # https://stackoverflow.com/questions/23435782/
        samples = np.take_along_axis(T_np, sample_indices[:,None], axis=1)
        total = sum(samples)
    return total

def do_work():
    start_time = time.time()
    L = create_sorted_random_tensor(N)
    L = L.to(device='cuda')
    torch.mm(L, L, out=L)
    #Lcpu = L.to(device='cpu')
    #stats = gen_stats(Lcpu)
    stats = gen_stats(L.cpu())
    torch.add(L, L, out=L)
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time, stats

# Warmup.
do_work()

with profile(use_cuda=True, use_kineto=True, record_shapes=True, with_stack=False) as p:
    print("no stack, time = " + str(do_work()))
filename = f"./N={N}-python_function_tracing-nostack.trace.json"
p.export_chrome_trace(filename)

with profile(use_cuda=True, use_kineto=True, record_shapes=True, with_stack=True) as p:
    print("with stack, time = " + str(do_work()))
filename = f"./N={N}-python_function_tracing-stack.trace.json"
p.export_chrome_trace(filename)

print("printed to " + filename)
