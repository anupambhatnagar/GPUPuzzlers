import time
import torch


def sync_and_pause():
torch.cuda.synchronize()
  time.sleep(1e-6)

size = 2^15
ones = torch.ones((size, size), device=torch.device('cuda'))

ones.mul_(0.5)
sync_and_pause()

result = ones.mul(0.5)
sync_and_pause()

result += ones
sync_and_pause()

total = ones + result
sync_and_pause()

result = torch.sin(ones)
sync_and_pause()

result = torch.sigmoid(ones)
sync_and_pause()

torch.sqrt(ones, out=result)
sync_and_pause()

torch.log10(ones, out=result)
sync_and_pause()

torch.pow(ones, 3.14159, out=result)
sync_and_pause()

result = torch.matmul(ones, ones)
sync_and_pause()
