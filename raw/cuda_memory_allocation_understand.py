import torch
import gc
import time
import sys

'''
Trying to get better understanding of gc benavior.
'''

gc.disable()

N = 100*1000*1000
M = 100
shape = (N,)
A = []
S = 1
for i in range(M):
    rand_tensor = torch.rand(shape, device='cuda')
    # With the lined below commented out memory does not increase (measured by nvidia-smi)
    # even though gc is explicitly turned off. Why? (Without the comment, each iteration
    # increases memory by 382MB \approx 400556032 Bytes, which makes sense for dfp32.)
    # A.append(rand_tensor)
    time.sleep(S)
    print(i, flush=True)

S = 5
print("Start Sleep")
time.sleep(S)
print("Start GC")
gc.collect()
print("Done GC")
time.sleep(S)
sys.exit(0)
