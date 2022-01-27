import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as f

torch.manual_seed(0)

BB_resolution = 2
chunks = BB_resolution**3
BB_chunks = []
xyz = torch.zeros(chunks,3)

# 0,1,2,3 0,1,2,3 0,1,2,3 0,1,2,3
iter = 0
while iter < (chunks):     
        index = iter % BB_resolution 
        xyz[iter,0] = index
        iter += 1

# # 0,0,0,0 1,1,1,1 2,2,2,2 3,3,3,3
iter = 0
while iter <= chunks:
        counter = iter * BB_resolution # 0,4,8,16
        xyz[counter:counter + BB_resolution,1] = iter % BB_resolution
        iter += 1 

# # 0,0,0,0 0,0,0,0 0,0,0,0 0,0,0,0     1,1,1,1 1,1,1,1 1,1,1,1 1,1,1,1     2,2,2,2 2,2,2,2 2,2,2,2 2,2,2,2    3,3,3,3 3,3,3,3 3,3,3,3 3,3,3,3
iter = 0
while iter <= chunks:
        counter = iter * (BB_resolution**2) # 0,4,8,16
        xyz[counter:counter + BB_resolution**2,2] = iter % BB_resolution
        iter += 1

print("xyz: ")
print(xyz)
