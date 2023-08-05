import os
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity

def all_gather(rank, size):
    hostname = socket.gethostname()
    trace_handler = tensorboard_trace_handler(dir_name="./traces/all_gather", worker_name=f"{hostname}_{rank}", use_gzip=True)
    with profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        on_trace_ready = trace_handler
    ) as prof:
        device = torch.device(f"cuda:{rank}")
        tensor = torch.tensor(torch.mul(torch.ones(2),36), dtype=torch.float32, device=device)

        tensor_out_list = [torch.zeros(2, dtype=torch.float32, device=device) for _ in range(size)]
        dist.all_gather(tensor_out_list, tensor)
        print(f"rank is {rank} tensor is {tensor_out_list}")


def init_process(rank, size, func, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    func(rank, size)

if __name__ == "__main__":
    size = torch.cuda.device_count()
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, all_gather))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
