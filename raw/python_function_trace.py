import time
import random 
import numpy as np

import torch
from torch.autograd.profiler import profile

'''
Synthetic example created for Aaron Shi's note on PFT - shows
how we need to enable it to see what's happening on host side.
'''

N = 1000

# Mimics host-side I/O and preprocessing.
def create_sorted_random_tensor(size):
    # It's more convenient to use numpy to create and 
    # manipulate random tensors.
    data = np.random.rand(N,N)
    # Sort data, treating it as an array of vectors.
    data.sort(0)
    # Convert to torch tensor (on CPU).
    result = torch.tensor(data)
    return result


# Mimics host-side statistics generation.
def sampled_average(T, num_samples_per_row):
    T_np = T.numpy() # Prefer to work on numpy arrays.
    total = 0.0
    for _ in range(num_samples_per_row):
        # Get a random entry per row by first selecting N
        # random indices, one per row.
        sample_indices = np.random.randint(0, high=N, size=N)
        # https://stackoverflow.com/questions/23435782/
        samples = np.take_along_axis(T_np, sample_indices[:,None], axis=1)
        total = sum(samples)
    return total/(N*num_samples_per_row)


def do_work():
    L = create_sorted_random_tensor(N)
    L = L.to(device='cuda')
    # Do an inplace matmul.
    torch.mm(L, L, out=L)
    stats_before = sampled_average(L.cpu(), 10)
    torch.add(L, L, out=L)
    stats_after = sampled_average(L.cpu(), 10)
    torch.cuda.synchronize()
    return stats_before, stats_after

# Warmup.
do_work()

with profile(use_cuda=True, use_kineto=True, record_shapes=True, with_stack=False) as p:
    print("no stack, stats = " + str(do_work()))
filename = f"./N={N}-python_function_tracing-nostack.trace.json"
p.export_chrome_trace(filename)

with profile(use_cuda=True, use_kineto=True, record_shapes=True, with_stack=True) as p:
    print("with stack, stats = " + str(do_work()))
filename = f"./N={N}-python_function_tracing-stack.trace.json"
p.export_chrome_trace(filename)

print("printed to " + filename)
