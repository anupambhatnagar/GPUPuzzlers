import os
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity

def reduce(rank, size):
    """ Distributed function to be implemented later. """
    hostname = socket.gethostname()
    trace_handler = tensorboard_trace_handler(dir_name="./traces/reduce", worker_name=f"{hostname}_{rank}", use_gzip=True)
    with profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        on_trace_ready = trace_handler
    ) as prof:
        group = dist.new_group(list(range(size)))
        tensor = torch.arange(2, dtype=torch.int32, device = torch.device(f"cuda:{rank}")) +1 + 2 * rank
#        print(f"rank is {rank} and tensor is {tensor}")

        dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=group)
#        print(f"rank is {rank} tensor is {tensor}")


def init_process(rank, size, func, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    func(rank, size)

    """
    print(f"distributed is initialized {d.is_initialized()} on rank {rank}")
    print(f"backend is {d.get_backend()}")
    print(f"world size is {d.get_world_size()}")
    """

if __name__ == "__main__":
    size = torch.cuda.device_count()
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, reduce))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
