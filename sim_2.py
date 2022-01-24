import torch
import numpy as np
import time

torch.manual_seed(0)

t = torch.Tensor([[1, 2, 3], [4,5,2]])
print((t < 2.0).nonzero(as_tuple=True))


a = torch.rand(1000000,1000)
print(a)