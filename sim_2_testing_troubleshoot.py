import torch
import numpy as np
import time

torch.manual_seed(0)
ptnums = 10

# pos = torch.zeros(ptnums,3, device='cuda')
# vel = torch.rand(ptnums,3, device='cuda')
# mass = torch.ones(ptnums,1, device='cuda')

# total = torch.zeros(ptnums,7)
# total[:,0:3] = pos[:,:]
# total[:,3:6] = vel[:,:]
# total[:,-1] = mass[0,:]
# print(total)

# Globals
negative_vector = torch.tensor([-1.0, -1.0, -1.0], device='cuda')

Total = torch.rand(10,7, device='cuda')
Total[5,1] = 0
Loss = torch.tensor([1.0, 1.0, 1.0], device='cuda')

# if( Total[:,1] < 0):
#                 Total[:,0:4] *= negative_vector
#                 Total[:,3:6] *= negative_vector
#                 Total[:,3:6] = torch.mul(Total[:,3:6], Loss)

collision_mask = torch.where(Total.double()[:,1] < 0.01, True, False) * -1
collision_mask = torch.where(collision_mask == 0, 1, collision_mask)

print(collision_mask)
Total[:,0:6] = torch.transpose(torch.mul(torch.transpose(Total[:,0:6], dim0=0,dim1=1), collision_mask),dim0=0,dim1=1)

print(Total)