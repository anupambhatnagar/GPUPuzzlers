import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

N = 1000000000

'''
Illustrating all_reduce and broadcast, draws from
https://pytorch.org/tutorials/intermediate/dist_tuto.html

TODO:
  - get custom allreduce working
  - figure out why broadcase async vs sync has such different perf
'''

""" Implementation of a ring-reduce with addition. """
def allreduce(send, recv):
   rank = dist.get_rank()
   size = dist.get_world_size()
   send_buff = send.clone()
   recv_buff = send.clone()
   accum = send.clone()

   left = ((rank - 1) + size) % size
   right = (rank + 1) % size

   for i in range(size - 1):
       if i % 2 == 0:
           # Send send_buff
           send_req = dist.isend(send_buff, right)
           dist.recv(recv_buff, left)
           accum[:] += recv_buff[:]
       else:
           # Send recv_buff
           send_req = dist.isend(recv_buff, right)
           dist.recv(send_buff, left)
           accum[:] += send_buff[:]
       send_req.wait()
   recv[:] = accum[:]


""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    #for param in model.parameters():
    #    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
    #    param.grad.data /= size
    for param in params:
        dist.all_reduce(params, op=dist.ReduceOp.SUM)
        param /= size
        print(param)

def run(rank, size):
    torch.manual_seed(1234)
    group = dist.new_group([0, 1])

    params = torch.ones(N) * float(rank)
    tensor = params.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank SUM ', rank, ' has data ', tensor)

    tensor = params.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.PRODUCT, group=group)
    print('Rank PRODUCT', rank, ' has data ', tensor)

    tensor = params.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    print('Rank MAX', rank, ' has data ', tensor)

    tensor = params.clone()
    start_time = time.time()
    print(f"starting broadcast from rank {rank}, tensor = {tensor}")
    dist.broadcast(tensor, src=0, group=group, async_op=True)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"rank {rank} has new tensor {tensor},  "
            f"time = {end_time - start_time}, " 
            f"BW = {4*N/(end_time - start_time)/1e9} GB/s")


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
