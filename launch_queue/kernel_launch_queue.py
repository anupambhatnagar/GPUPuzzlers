import time
import torch
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity


def launch_kernels(large_matrix, small_matrix):
    # Block 1: execute small matmuls followed by large matmul
    block1_start = time.time()
    for _ in range(10):
       torch.matmul(small_matrix, small_matrix)
    torch.matmul(large_matrix, large_matrix)
    block1_end = time.time()

    # sleep
    torch.cuda.synchronize()
    time.sleep(delay)

    # Block 2: execute large matmul followed by small matmuls
    block2_start = time.time()
    torch.matmul(large_matrix, large_matrix)
    for _ in range(10):
       torch.matmul(small_matrix, small_matrix)
    block2_end = time.time()

    print(f"Block 1 takes = {block1_end - block1_start}, Block 2 takes = {block2_end - block2_start} ")


if __name__ == "__main__":
    large = 2**12
    small = 2**6
    delay = 1 # in milliseconds
    cuda = torch.device('cuda')

    # Create a big and small matrix
    large_matrix = torch.rand((large, large), device='cuda')
    small_matrix = torch.rand((small, small), device='cuda')

    with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.GPU],
                 trace_handler = tensorboard_trace_handler(dir_name="/tmp", use_gzip=True),
                 with_stack = True) as prof:

        launch_kernels(large_matrix, small_matrix)
