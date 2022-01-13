import torch
import numpy as np
import time

torch.manual_seed(0)

x = torch.tensor([[ 13.6019, -11.2565,   0.0566],
        [  7.7608,  -0.9985,  -9.0562],
        [ -2.5147,  18.5134,  -8.0817],
        [  1.1400,   0.2767,   2.8587],
        [ -7.9098,   1.7254,   6.1462],
        [-12.0256,  -1.2363,  -4.8732],
        [  0.5822,   8.1032,  13.4576]], device='cuda:0')

t = torch.tensor([[  8.1881,  -5.0513,  -8.6289],
        [-18.6380,  -4.6253, -24.2050],
        [  2.5620,  11.5330,  15.4375]], device='cuda:0')

index = torch.tensor([1, 2, 6], device='cuda:0')
x.index_copy_(0, index, t)
print(x)