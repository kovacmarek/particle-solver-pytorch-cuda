import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as f

torch.manual_seed(0)

x = torch.zeros(5, 3)
# t = torch.tensor([3,4,6,6,5,4,6,5], dtype=torch.float)
t = torch.rand(5,3)*10
index = torch.tensor([0,0,0,1,2,3,3,4])
x.index_add_(0, index, t)
print(x)




