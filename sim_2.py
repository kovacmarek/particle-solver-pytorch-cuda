import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as f

torch.manual_seed(0)

a = torch.tensor([
        [0, 3],
        [0, 4],
        [0, 6],
        [1, 6],
        [2, 5],
        [3, 4],
        [3, 6],
        [4, 5]], device='cuda:0')

b = torch.select(0, a[:,0])
print(b)




