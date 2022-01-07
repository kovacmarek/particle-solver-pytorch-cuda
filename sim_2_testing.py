import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# particles = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
# primitives = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
# print(torch.cdist(primitives, particles, p=2)) 

# Each collumn is one particle and distance to each primitive is in rows
# Rows = primitives, collumns = particles
# min function over each collumn outputs a row number (Closest primitive ID)

# In this example first particle (first collumn) has closest 3rd primitive (row=2.2830), so we do DOT product
# of first particle and 3rd primitive's normal. If DOT product is negative value, we take position and normal from this primitive.
# with a position & normal given by a primitive, we can flip the first particle over this primitive's normal plane.

# Biggest negative DOT value

planeNormal = torch.tensor([0.0, 1.0])
planeOrigin = torch.tensor([2.0, 2.0])
rayOrigin = torch.tensor([4.0, 4.0])
rayDirection = torch.tensor([1.0,-1.0])


print("planeNormal: ")
print(planeNormal)
print("planeOrigin: ")
print(planeOrigin)
print("rayOrigin: ")
print(rayOrigin)
print("rayDirection: ")
print(rayDirection)


first = torch.dot(planeNormal, rayOrigin - planeOrigin)
print("first: ")
print(first)

second = torch.dot(planeNormal, -rayDirection)
print("second: ")
print(second)

third = first/second
print("third: ")
print(third)

length = rayOrigin + third * rayDirection
print("length: ")
print(length)