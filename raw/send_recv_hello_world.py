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

N = 1000000000
#N = 1000

def run(rank, size, sender, receiver):
    with profile(use_cuda=True, use_kineto=True, record_shapes=True) as p:
        if rank == sender:
            tensor = torch.zeros(N, device="cuda:" + str(rank))
            tensor += 42
        else:
            tensor = torch.empty(N, device="cuda:" + str(rank))

        if rank == sender:
            torch.cuda.synchronize()
            start_time = time.time()
            dist.send(tensor=tensor, dst=receiver)
            torch.cuda.synchronize()
            end_time = time.time()
        else:
            torch.cuda.synchronize()
            start_time = time.time()
            dist.recv(tensor=tensor, src=sender)
            torch.cuda.synchronize()
            end_time = time.time()
            print(f"Effective receive bandwidth {sender} -> {receiver} = {4 * N/(end_time - start_time)/1e9} GB/s")

        print('Rank ', rank, ' has data ', tensor[0])
    p.export_chrome_trace(f"./nccl-r={rank}-with_stack-with_record_shapes.trace.json")


def init_process(rank, size, fn, sender, receiver, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, sender, receiver)


if __name__ == "__main__":
    size = 2
    R = range(size)
    # Optionally pass in args for the cuda devices to test
    if len(sys.argv) > 1:
        print(sys.argv)
        R = [int(sys.argv[1]), int(sys.argv[2])]
        sender = R[0]
        receiver = R[1]
    processes = []
    mp.set_start_method("spawn")
    for rank in R:
        p = mp.Process(target=init_process, args=(rank, size, run, sender, receiver, 'nccl'))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
