import random
import time

import numpy as np

import torch
from torch import nn
from torch.autograd.profiler import profile

embedding = nn.Embedding(10, 3)
#print(embedding)
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])

e = embedding(input)
#print(e)


weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
embedding = nn.Embedding.from_pretrained(weight)
input = torch.LongTensor([1])
#print(embedding(input))

cuda = torch.device('cuda')
def bwcheck(num_lookups, pooling_factor, M, N):
  #weight = torch.FloatTensor([ [float(i) for _ in range(N)] for i in range(M) ]).pin_memory().to(cuda)
  data =  np.random.rand(M,N)
  weight = torch.tensor(data).to(cuda)
  embedding = nn.Embedding.from_pretrained(weight)
  #tmp = [ [random.randrange(M) for _ in range(pooling_factor) for _ in range(num_lookups)] ]
  batchsize = 100
  tmp = np.random.randint(N, size=(batchsize, num_lookups, pooling_factor))
  lookup = torch.LongTensor(tmp).pin_memory().to(cuda)

  start_time = time.time()
  for _ in range(100):
    result = embedding(lookup)
  torch.cuda.synchronize()
  end_time = time.time()
  total_time = end_time - start_time
  print(f"{total_time}: Memory BW ({M, N, pooling_factor, num_lookups} = {(4 * N * pooling_factor * num_lookups)/total_time/1e9} GB/s")

with profile(use_cuda=True, use_kineto=True, record_shapes=True, with_stack=True) as p:
    #bwcheck(1024, 128, 100000, 128)
    #bwcheck(256, 1, 1000000, 128)
    for dim in [1,2,4,8,16,32,64,1128,256,512]:
        bwcheck(128, 16, 100000, dim)


filename = f"./embedding_mwe.json"
p.export_chrome_trace(filename)
print("printed to " + filename)
