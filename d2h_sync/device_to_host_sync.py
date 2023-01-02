import time
import torch
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity


def first_sum():
    total = 0.0
    for i in range(tensor.size()[0]):
        total += tensor[i].cpu()
    return total


def second_sum():
    total = torch.zeros(1, device='cuda')
    for i in range(tensor.size()[0]):
        total += tensor[i]
    return total


def third_sum():
    total = 0.0
    tensor_on_cpu = tensor.cpu()
    for i in range(tensor_on_cpu.size()[0]):
        total += tensor_on_cpu[i]
    return total


def sync_and_sleep(delay: float = 0.001):
    torch.cuda.synchronize()
    time.sleep(delay)


torch.manual_seed(145) # Fun fact: 145 = 1! + 4! +5!
tensor = torch.rand(2**12, device='cuda')

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             on_trace_ready=tensorboard_trace_handler(dir_name=f"./d2h_sync_trace", use_gzip=True),
             record_shapes=True,
             with_stack = True) as prof:

    sum1 = first_sum()
    sync_and_sleep()

    sum2 = second_sum()
    sync_and_sleep()

    sum3 = third_sum()

assert sum1 == sum2.item() == sum3
