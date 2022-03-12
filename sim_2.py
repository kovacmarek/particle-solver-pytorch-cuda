import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as f

torch.manual_seed(0)


def cross_product_rows(a, b):
    c = torch.zeros(4,3, device='cuda:1')
    c[:,0] = a[:,1] * b[:,2] - a[:,2] * b[:,1]
    c[:,1] = a[:,2] * b[:,0] - a[:,0] * b[:,2]
    c[:,2] = a[:,0] * b[:,1] - a[:,1] * b[:,0]

    return c

# init values
upvector = torch.zeros(4,3, device='cuda:1').float()
upvector[:,1] = 1
normal = torch.rand(4,3, device='cuda:1').float()

# normalize vector
N_upvector_mult = (1 / torch.sum(abs(upvector + 0.00001), dim=-1)) 
N_upvector = upvector * N_upvector_mult.view(4,1)
print("N_upvector: ", N_upvector)

# normalize vector
N_normal_mult = (1 / torch.sum(abs(normal + 0.00001), dim=-1)) 
N_normal = normal * N_normal_mult.view(4,1)

# cross products
ppr = cross_product_rows(N_normal, N_upvector)
direction = -cross_product_rows(ppr, N_normal)

# compute angle in radians
vector_dot = torch.sum(upvector * N_normal, dim=1)
len_a = torch.norm(abs(upvector), p=2, dim=-1)
len_b = torch.norm(abs(N_normal), p=2, dim=-1)
angle = torch.acos(vector_dot / (len_a * len_b)  )

# radians to deg
angle = angle * (180/3.14)

# mult direction vector with scalar
slide = direction * angle.view(4,1)

