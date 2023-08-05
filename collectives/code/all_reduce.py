import os, sys
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity

def all_reduce(rank, size, jobname, param):
    group = dist.new_group(list(range(size)))
    device = torch.device(f"cuda:{rank}")
    tensor = torch.ones(param * size, dtype=torch.float32, device=device)

    hostname = socket.gethostname()
    path = "/home/anupamb/distributed/traces/all_reduce/" + jobname
    trace_handler = tensorboard_trace_handler(dir_name=path, worker_name=f"{hostname}_{rank}", use_gzip=True)

    with profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        on_trace_ready = trace_handler
    ) as prof:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

    #print(f"rank is {rank} tensor is {tensor}")


def init_process(rank, size, func, jobname, param, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    func(rank, size, jobname, param)

if __name__ == "__main__":
    size = torch.cuda.device_count()
    processes = []
    mp.set_start_method("spawn")
    jobname, param = str(sys.argv[1]), int(sys.argv[2])
    assert param is not None
    print(f"param = {param}")

    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, all_reduce, jobname, param))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
