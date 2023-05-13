import torch
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity


def data_transfer():
    with torch.cuda.stream(first):
        data.to(torch.device('cuda:1'))

    with torch.cuda.stream(second):
        data.to(torch.device('cuda:2'))

    with torch.cuda.stream(third):
        data.to(torch.device('cuda:4'))

first, second, third = [torch.cuda.Stream() for _ in range(3)]
data = torch.rand((10**8), device=torch.device('cuda:0'), dtype=torch.float32)
trace_handler = tensorboard_trace_handler(dir_name="./traces/p2p_copy", use_gzip=True)

data_transfer()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=trace_handler,
    with_stack=True
) as prof:
    data_transfer()
