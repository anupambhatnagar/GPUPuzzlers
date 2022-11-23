# https://discuss.pytorch.org/t/did-torch-sort-drastically-improve-in-1-10/135326

import time
import torch

n = 5 * (10 ** 8)

def benchmark_sort(T, device):
  # Start time measurement
  n = len(T)
  if device == 'cuda':
    torch.cuda.synchronize()
  t = time.perf_counter()
  output, _ = torch.sort(x)
  # End time measurement
  if device == 'cuda':
    torch.cuda.synchronize()
  sort_time = time.perf_counter() - t
  
  print(f'{device}, {n/1e6}M Latency: {sort_time}, Throughput: {round((n / sort_time) / 1e6, 3)} MKPS')

for n in [10 ** 7, 5 * (10 **7), 10 ** 8, 2 * 10 ** 8, 5 * (10 ** 8), 10 ** 9, 2 * (10 ** 8)]:
  device = 'cuda'
  x = torch.randn((n, ), device=device)
  benchmark_sort(x, device)

  device = 'cpu'
  xcpu = torch.randn((n, ), device=device)
  benchmark_sort(xcpu, device)
