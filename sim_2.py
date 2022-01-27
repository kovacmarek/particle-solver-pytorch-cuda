import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as f

torch.manual_seed(0)

a = torch.rand(7,3)
b = torch.zeros(7,3)

final = torch.cat((a, b),1)

print(final)
final[:,0] = torch.where(a[:,0] > 0, 5,-1)
print(final)


