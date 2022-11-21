import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# new(AA):
from torch.autograd.profiler import profile

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

N = 10000

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(N, N)
        self.relu = nn.ReLU()
        #self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        #return self.net2(self.relu(self.net1(x)))
        return self.relu(self.set1(self.net1(self.net1(self.net1(self.net1(x))))))



def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)

    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    with profile(use_cuda=True, use_kineto=True, record_shapes=True) as p:
        #outputs = ddp_model(torch.randn(20, 10))
        outputs = ddp_model(torch.randn(N, N))
        #labels = torch.randn(20, 5).to(rank)
        #loss_fn(outputs, labels).backward()
        #optimizer.step()
        labels = torch.randn(N, N).to(rank)
        loss_fn(outputs, labels).backward()
        optimizer.step()

    cleanup()
    p.export_chrome_trace(f"./ddp-r={rank}-with_stack-with_record_shapes.trace.json")


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print("n_gpus = " + str(n_gpus))
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)
    #run_demo(demo_checkpoint, world_size)
    #run_demo(demo_model_parallel, world_size)
