import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as f

# planeNormal = torch.tensor([0.0, 1.0])
# planeOrigin = torch.tensor([2.0, 2.0])
# rayOrigin = torch.tensor([4.0, 4.0])
# rayDirection = torch.tensor([1.0,-1.0])


# print("planeNormal: ")
# print(planeNormal)
# print("planeOrigin: ")
# print(planeOrigin)
# print("rayOrigin: ")
# print(rayOrigin)
# print("rayDirection: ")
# print(rayDirection)


# first = torch.dot(planeNormal, rayOrigin - planeOrigin)
# print("first: ")
# print(first)

# second = torch.dot(planeNormal, -rayDirection)
# print("second: ")
# print(second)

# third = first/second
# print("third: ")
# print(third)

# length = rayOrigin + third * rayDirection
# print("length: ")
# print(length)

# If DOT is less than 0.0001, it's a correct primitive = smallest DOT
# If first DOT is negative, shoot ray backwards to get position on the plane
# then with position on plane check for DOT product over all primitives, smallest is the correct primitive it collided with



# ----- REFLECTION OF VECTOR ----
torch.manual_seed(0)

normal = torch.tensor([-1.0, 1.0, 0.0])
planeOr = torch.tensor([40.0, 40.0, 0.0])
planeNo = torch.tensor([0.0, 1.0, 0.0])

N_normal = f.normalize(normal, p=2, dim=0)
Va = torch.tensor([1.0, -0.5, 0.0])
N_Va = f.normalize(Va, p=2, dim=0)


# R = 2(N . L) * N - L (all normalized)
Vb = 2*(torch.dot(N_normal, N_Va)) * N_normal - N_Va
print("N_normal: ")
print(N_normal)
print("N_Va: ")
print(N_Va)
print("Vb: ")
print(Vb)

Vb = 2*(torch.matmul(torch.t(N_normal, dim0=0, dim1=1), N_Va)) * N_normal - N_Va


# INSIDE HOUDINI:

inputs = node.inputs()
geo1 = inputs[1].geometry()

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as f

# Add code to modify contents of geo.
# Use drop down menu to select examples.

init_collision_norm = geo1.pointFloatAttribValues("N") 
t_collision_norm = torch.tensor(init_collision_norm, device='cuda')
normal = t_collision_norm.reshape(3,1)

init_collision_norm = geo.pointFloatAttribValues("N") 
t_collision_norm = torch.tensor(init_collision_norm, device='cuda')
Va = t_collision_norm.reshape(3,1)
#print(Va)


torch.manual_seed(0)

N_normal = f.normalize(normal, p=2, dim=0)
N_Va = f.normalize(Va, p=2, dim=0)



Vb = 2*(torch.matmul(torch.transpose(N_normal, dim0=0, dim1=1), N_Va)) * N_normal - N_Va
print(Vb)

final_pos = torch.flatten(Vb).cpu().numpy()
print(final_pos)
geo.setPointFloatAttribValuesFromString("N", final_pos)