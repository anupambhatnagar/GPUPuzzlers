---
layout: post
title: Communication is the key to success
permalink: /posts/collectives/
excerpt: Communication matters!
---

<p align = "center">
  <a href="/collectives/topology.png">
    <img src="/collectives/topology.png">
  </a>
</p>

## Puzzler 1

The figure above shows that network topology of the GPUs in a server. The `data_transfer` function
below does a peer-to-peer (P2P) copy of a list of tensors from GPU0 to GPU1, GPU2 and GPU4 in 4.5
ms, 53.2 ms and 8.8 ms respectively. What is the reason for the variance in duration of the P2P
copies?

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
