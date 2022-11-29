import torch
from torch import nn

embedding = nn.Embedding(10, 3)
print(embedding)
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])

e = embedding(input)
print(e)


weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
embedding = nn.Embedding.from_pretrained(weight)
input = torch.LongTensor([1])
print(embedding(input))
