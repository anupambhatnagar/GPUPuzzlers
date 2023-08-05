import os, sys
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity

def reduce_scatter_all_gather(rank, size, jobname, param):
    group = dist.new_group(list(range(size)))
    device = torch.device(f"cuda:{rank}")
    rs_input = torch.ones(param * size, dtype=torch.float32, device=device)
    rs_output = torch.zeros(param, dtype=torch.float32, device=device)
    ag_output = torch.zeros(param * size, dtype=torch.float32, device=device)

    hostname = socket.gethostname()
    path = "/home/anupamb/distributed/traces/reduce_scatter_all_gather/" + jobname
    trace_handler = tensorboard_trace_handler(dir_name=path, worker_name=f"{hostname}_{rank}", use_gzip=True)

    with profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        on_trace_ready = trace_handler
    ) as prof:
        dist.reduce_scatter_tensor(rs_output, rs_input, op=dist.ReduceOp.SUM, group=group)
        dist.all_gather_into_tensor(ag_output, rs_output, group=group)

#    print(f"rank = {rank} tensor = {ag_output}")

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
        p = mp.Process(target=init_process, args=(rank, size, reduce_scatter_all_gather, jobname, param))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
