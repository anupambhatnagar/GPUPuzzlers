import time
import torch


def sync_and_pause():
    torch.cuda.synchronize()
    time.sleep(1e-9)

size = 2**10
ones = torch.ones((size, size), device=torch.device('cuda'))

# in place multiplication
ones.mul_(0.5)
sync_and_pause()

result = ones.mul(0.5)
sync_and_pause()

total = ones + result
sync_and_pause()

result = torch.sin(ones)
sync_and_pause()

result = torch.sigmoid(ones)
sync_and_pause()

result = torch.sqrt(ones)
sync_and_pause()

result = torch.log10(ones)
sync_and_pause()

result = torch.pow(ones, 3.14159)
sync_and_pause()

result = torch.matmul(ones, ones)
sync_and_pause()
