import os
import time
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.autograd.profiler import profile

'''
Minimal example that shows how send and recv appear on a trace.

Be sure CUDA_VISIBLE_DEVICES is not masking GPUs, e.g., do
% export CUDA_VISIBLE_DEVICES=0,1

Can use this in a shell script to get pairwise bandwidths. Here's 
NVIDIA's code to do the same test:
  - Install from github: https://github.com/NVIDIA/cuda-samples
  - Checkout tag 11.4 for devgpu (test cuda version with nvcc --version)
  - Make and execute p2pBandwidthLatencyTest
  - Example output: P563418103

'''

N = 1000
N = 1000000000

def run(rank, size, sender, receiver):
    print(f"in run, rank = {rank}")
    with profile(use_cuda=True, use_kineto=True, record_shapes=True) as p:
        if rank == 0:
            tensor = torch.zeros(N, device="cuda:" + str(sender))
            tensor += 42
        else:
            tensor = torch.empty(N, device="cuda:" + str(receiver))
        print(f"in run, tensors ready")

        if rank == 0:
            torch.cuda.synchronize()
            start_time = time.time()
            print(f"sending from {rank}")
            dist.send(tensor=tensor, dst=1)
            torch.cuda.synchronize()
            end_time = time.time()
        else:
            torch.cuda.synchronize()
            start_time = time.time()
            print(f"receiving at {rank}")
            dist.recv(tensor=tensor, src=0)
            torch.cuda.synchronize()
            end_time = time.time()
            print(f"Effective receive bandwidth {sender} -> {receiver} = {4 * N/(end_time - start_time)/1e9} GB/s")

        print('Rank ', rank, ' has data ', tensor[0])
    p.export_chrome_trace(f"./nccl-r={rank}-with_stack-with_record_shapes.trace.json")


def init_process(rank, size, fn, sender, receiver, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
        
    #os.environ['MASTER_PORT'] = '29501'
    print(f"in init_process, rank = {rank}")
    dist.init_process_group(backend, rank=rank, world_size=size)
    print(f"in init_process, rank = {rank} - initialized")
    fn(rank, size, sender, receiver)


if __name__ == "__main__":
    size = 2
    # Optionally pass in args for the cuda devices to test
    if len(sys.argv) > 1:
        print(sys.argv)
        R = [int(sys.argv[1]), int(sys.argv[2])]
        sender = R[0]
        receiver = R[1]
        print(f"sender, receiver = {sender}, {receiver}")
    processes = []
    mp.set_start_method("spawn")
    R = range(size)
    for rank in R:
        print(f"launching rank {rank}")
        p = mp.Process(target=init_process, args=(rank, size, run, sender, receiver, 'nccl'))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
