---
layout: page
title: Answer
permalink: /collectives-answer/
---

### Puzzler 1: Peer to Peer Bandwidth

The network topology within a server can be obtained executing `nvidia-smi topo -m` on the command
line. The variance in the duration is due to the fact that the GPUs are connected to each other
using different interconnects. The most common interconnects are NVLink (a direct GPU-to-GPU
interconnect that scales multi-GPU input/output within the server) and PCIe.

GPU0 and GPU1 are connected via 4 NVLinks, GPU0 and GPU4 are connected via 2 NVLinks and GPU0 and
GPU2 are connected via PCIe. In essence, the underlying topology is the main cause for the variation
in the data transfer durations.

### Puzzler 2: Collective Performance

While the All_Reduce, Reduce + Broadcast and Reduce_Scatter + All_Gather are mathematically equivalent
their performance depends on:

1. NCCL algorithm and protocol
1. Network topology
1. Number of GPUs
1. Message size

Empirically, we tested the Ring and Tree Algorithms on 8 and 16 GPUs with a 4GB message size and
observed that



## Discussion

__What are the different algorithms and protocols available in NCCL?__

The available algorithms in NCCL are:
1. Ring
1. Tree
1. CollNet

In the Ring and Tree algorithms the GPUs form a ring and tree respectively. CollNet can be used,
only when Infiniband is available. At a high level CollNet creates hierarchical rings and the Ring
All Reduce can be broken down as: Inter-node Reduce Scatter followed by Intra-node All Reduce and
then an Inter-node All Gather.

The protocols available in NCCL are:
1. Simple
1. Low Latency (LL)
1. Low Latency 128 (LL128)

The protocol can be specified using the NCCL_PROTO environment variable although the environment
variable is primarily for debugging purposes only. For more details on the protocols, please see
[here](https://github.com/NVIDIA/nccl/issues/281).

__How does NCCL determine which algorithm and protocol to use for any given collective operation?__

NCCL implements an [Auto Tuner](https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc)
which computes heuristics for all possible algorithm and protocol combinations for all the
collectives. The Auto Tuner executes once when NCCL is initialized. It performs an analysis for
different combinations of algorithms and protocols and caches the results for subsequent use. The
Auto Tuner uses the number of nodes, number of GPUs per node and the message size as parameters.
Implicitly, it also uses the underlying toplogy and fabric to estimate the performance of the
communication collectives along with the algorithms and protocols.

__Does NCCL take the topology and interconnect into account?__

Yes, the NCCL Auto Tuner does that implicitly under the hood.

__What is the difference between NCCL Tree and Ring algorithms?__

The Tree algorithm is available only for the All Reduce collective when the number of nodes is
greater than 1.

__Can you force NCCL to use a particular algorithm?__

Users can specify the algorithm using the NCCL_ALGO environment variable.

__Can you bypass NCCL, e.g., by using .to('cuda:3') operations? Is there any performance benefit/loss?__

Yes, users can bypass NCCL but it strongly advised not to do so. NCCL implements sophisticated
algorithms and protocols along with efficient pipelining techniques and makes use of underlying
topology and fabric which cannot be mimicked with a naive approach such as using `.to('cuda:3')`.

## What should you remember in years to come?

Communication plays a critical role in any distributed system. In the context of distributed
training on several thousand GPUs communication libraries like NCCL utilize sophisticated
algorithms and protocols which are opaque to a casual user. It is probably best to leave the
communication optimizations to the experts in the domain.

## Explore More

- [NCCL Tests](https://github.com/NVIDIA/nccl-tests) a library to benchmark performance and
  correctness of NCCL operations.

- Blogs on NCCL [1](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/), [2](https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/).

- NCCL on the Summit Supercomputer. [Slides by Sylvain Jeaugey](https://www.olcf.ornl.gov/wp-content/uploads/2019/12/Summit-NCCL.pdf).

- Learn about [SHARP](https://docs.nvidia.com/networking/display/sharpv214/Introduction).
