import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as f

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

# If DOT is less than 0.0001, it's a correct primitive = smallest DOT
# If first DOT is negative, shoot ray backwards to get position on the plane
# then with position on plane check for DOT product over all primitives, smallest is the correct primitive it collided with


# Compute distance

ptnums = len(geo.points())
collisionPtnums = len(geo1.points())

collisionTotal = torch.zeros(collisionPtnums,6, device='cuda')
particlesTotal = torch.zeros(ptnums,12, device='cuda')

# collision append
init_collision_pos = geo1.pointFloatAttribValues("P") 
t_collision_pos = torch.tensor(init_collision_pos, device='cuda')
collisionTotal[:,0:3] = t_collision_pos.reshape(collisionPtnums,3)

init_collision_norm = geo1.pointFloatAttribValues("N") 
t_collision_norm = torch.tensor(init_collision_norm, device='cuda')
collisionTotal[:,3:6] = t_collision_norm.reshape(collisionPtnums,3)

# particles append
init_particles_pos = geo.pointFloatAttribValues("P") 
t_particles_pos = torch.tensor(init_particles_pos, device='cuda')
particlesTotal[:,0:3] = t_particles_pos.reshape(ptnums,3)

init_particles_norm = geo.pointFloatAttribValues("N") 
t_particles_norm = torch.tensor(init_particles_norm, device='cuda')
particlesTotal[:,3:6] = t_particles_norm.reshape(ptnums,3)

torch.manual_seed(0)

# compute distance
dist = torch.cdist(collisionTotal[:,0:3], particlesTotal[:,0:3], p=2.0)

print("dist: ")
print(dist)

# find minarg for each collumn (particle)
mina = torch.argmin(dist, dim=0)
print("mina: ")
print(mina)

print("particle_pos: ")
print(particlesTotal[:,0:3])

print("collisionTotal[:,0:3]: ")
print(collisionTotal[:,0:3])

# append each particle it's closest primitive's position
particlesTotal[:,6:9] = collisionTotal[:,0:3].index_select(0, mina)
particlesTotal[:,9:12] = collisionTotal[:,3:6].index_select(0, mina)

final_dir = particlesTotal[:,6:9] - particlesTotal[:,0:3]

print(particlesTotal)
final_pos = torch.flatten(particlesTotal[:,6:9]).cpu().numpy()
geo.setPointFloatAttribValuesFromString("P", final_pos)



# ----- REFLECTION OF VECTOR ----

##########################
# INSIDE HOUDINI:

node = hou.pwd()
geo = node.geometry()
inputs = node.inputs()
geo1 = inputs[1].geometry()

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as f

# Add code to modify contents of geo.
# Use drop down menu to select examples.

ptnums = len(geo.points())

init_collision_norm = geo1.pointFloatAttribValues("N") 
t_collision_norm = torch.tensor(init_collision_norm, device='cuda')
normal = t_collision_norm.reshape(1,3)
print("normal :")
print(normal)

init_collision_norm = geo.pointFloatAttribValues("N") 
t_collision_norm = torch.tensor(init_collision_norm, device='cuda')
Va = t_collision_norm.reshape(ptnums,3)

torch.manual_seed(0)

N_normal = f.normalize(normal, p=2, dim=0)
N_Va = f.normalize(Va, p=2, dim=0)
print(N_normal)
print(N_Va)
print("-----")


Vb = 2*(torch.matmul(normal, torch.transpose(N_Va, dim0=0, dim1=1)))
print("Vb1: ")
print(Vb)

Vb = (torch.transpose(Vb, dim0=0, dim1=1) * normal)
print("Vb2: ")
print(Vb)

Vb -= N_Va
Vb *= -1
print("Vb: ")
print(Vb)

final_pos = torch.flatten(Vb).cpu().numpy()
print(final_pos)
geo.setPointFloatAttribValuesFromString("N", final_pos)