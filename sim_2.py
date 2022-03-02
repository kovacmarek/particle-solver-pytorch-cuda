import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as f

torch.manual_seed(0)

selected = torch.rand(7,3, device='cuda:1')*10

if ((selected + 2) < selected) & ((selected - 2) > selected):
    cdist = torch.cdist(selected, selected, p=2)
print(cdist)




