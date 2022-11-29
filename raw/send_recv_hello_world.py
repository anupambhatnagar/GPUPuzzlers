import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.autograd.profiler import profile

'''
Minimal example that shows how send and recv appear on a trace.

TODOs:
 - get effective BW between devices for each pair of devices (generalize to multihost?), 
   tie into the server architecture (nvlink, nvswitch, PCIE, roce)
 - figure out what's happening with the traces, appear very noisy

Warning - be sure CUDA_VISIBLE_DEVICES is not masking GPUs, e.g., do
% export CUDA_VISIBLE_DEVICES=0,1

'''

# Can use this to hard code the two devices we use for distributed.
# deviceA = 6
# deviceB = 7

def run(rank, size):
    with profile(use_cuda=True, use_kineto=True, record_shapes=True) as p:
        tensor = torch.zeros(1).to(rank)
        if rank == 0:
            tensor += 1
            # Send the tensor to process 1
            dist.send(tensor=tensor, dst=1)
        else:
            # Receive tensor from process 0
            dist.recv(tensor=tensor, src=0)
        print('Rank ', rank, ' has data ', tensor[0])
    p.export_chrome_trace(f"./nccl-r={rank}-with_stack-with_record_shapes.trace.json")

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 2
    R = range(2)
    #R = [deviceA, deviceB]  # Use these two GPUs.
    #size = len(R)
    processes = []
    mp.set_start_method("spawn")
    for rank in R:
        p = mp.Process(target=init_process, args=(rank, size, run, 'nccl'))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
