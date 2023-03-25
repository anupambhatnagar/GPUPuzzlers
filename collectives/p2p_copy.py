import torch
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity


def data_transfer():
    with torch.cuda.stream(first):
        for i in range(len(data)):
            data[i].to(torch.device('cuda:1'), non_blocking=True)

    with torch.cuda.stream(second):
        for i in range(len(data)):
            data[i].to(torch.device('cuda:2'), non_blocking=True)

    with torch.cuda.stream(third):
        for i in range(len(data)):
            data[i].to(torch.device('cuda:4'), non_blocking=True)

first = torch.cuda.Stream()
second = torch.cuda.Stream()
third = torch.cuda.Stream()

data = [torch.rand((10**7), device=torch.device('cuda:0'), dtype=torch.float32) for _ in range(10)]
trace_handler = tensorboard_trace_handler(dir_name="./traces/p2p_copy", use_gzip=True)

data_transfer()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=trace_handler,
    with_stack=True
) as prof:
    data_transfer()
