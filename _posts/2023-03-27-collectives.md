---
layout: post
title: Communication is the Key to Success
permalink: /posts/collectives/
excerpt: Communication matters!
---

<p align = "center">
  <a href="/collectives/topology.png">
    <img src="/collectives/topology.png">
  </a>
  Server topology
</p>

## Puzzler 1: Bandwidth and Topology

The figure above shows that network topology of the GPUs in a server. As observed from the
[trace](/collectives/p2p_bandwidth.json.gz), the `data_transfer` function below does a peer-to-peer
(P2P) copy of a list of tensors from GPU 0 to GPU 1, GPU 2 and GPU 4 in 4.5 ms, 53.2 ms and 8.8 ms
respectively.

Why do the P2P copies to different GPUs vary so much?

<p align = "center">
  <a href="/collectives/p2p_trace.png">
    <img src="/collectives/p2p_trace.png">
  </a>
  Trace for data_transfer function
</p>

``` python
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

cuda = torch.device('cuda:0')
fp32 = torch.float32
data = [torch.rand((10**7), device = cuda, dtype = fp32) for _ in range(10)]
```

## Puzzler 2: Collective Performance

Message Passing Interface (MPI) is a communication protocol used in parallel computing. It supports
both point-to-point (Send and Receive) and collective (Reduce, Broadcast, Scatter, Gather,
All_Reduce etc.) communication. PyTorch DDP and FSDP use MPI collectives under the hood to do
distributed training.

It is [well
known](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html#reduce) that
All_Reduce is mathematically equivalent to Reduce followed by Broadcast. With some thought one can
prove that All_Reduce is equivalent to Reduce_Scatter followed by All_Gather.

The functions below use All_Reduce, Reduce + Broadcast, and Reduce_Scatter + All_Gather to sum
tensors. What are the factors influence the performance of the implementations given?

``` python
import torch
import torch.distributed as dist

# approach 1 - call All_Reduce directly
def all_reduce_demo(size, local_rank, tensor_size=2**20):
  group = dist.new_group(list(range(size)))
  device = torch.device(f"cuda:{local_rank}")
  tensor = torch.rand(tensor_size, device=device)
  dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

# approach 2 - implement All_Reduce using Reduce and Broadcast
def reduce_broadcast_demo(size, local_rank, tensor_size=2**20):
  group = dist.new_group(list(range(size)))
  device = torch.device(f"cuda:{local_rank}")
  tensor = torch.rand(tensor_size, device=device)

  dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=group)
  dist.broadcast(tensor, src=0, group=group)

# approach 3 - implement All_Reduce using Reduce_Scatter and All_Gather
def reduce_scatter_all_gather_demo(size, local_rank, tensor_size=2**20):
  group = dist.new_group(list(range(size)))
  device = torch.device(f"cuda:{local_rank}")

  reduce_scatter_input = torch.rand(tensor_size, device=device)
  reduce_scatter_output = torch.zeros(tensor_size, device=device)
  all_gather_output = torch.zeros(tensor_size, device=device)

  dist.reduce_scatter_tensor(
    reduce_scatter_output,
    reduce_scatter_input,
    op=dist.ReduceOp.SUM,
    group=group
  )
  dist.all_gather_into_tensor(
    all_gather_output,
    reduce_scatter_output,
    group=group
  )
```
