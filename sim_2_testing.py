import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as f

node = hou.pwd()
geo = node.geometry()
inputs = node.inputs()
geo1 = inputs[1].geometry()


# ----- FIND CLOSEST PRIM AND IT'S POS & NORMAL ----
##########################

ptnums = len(geo.points())
collisionPtnums = len(geo1.points())

collisionTotal = torch.zeros(collisionPtnums,7, device='cuda') #7th value is distance
particlesTotal = torch.zeros(ptnums,7, device='cuda') # 13th value is boolean if it's intersecting

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

# Compute distance
dist_A = torch.cdist(collisionTotal[:,0:3], particlesTotal[:,0:3], p=2.0)
dist_B = torch.cdist(collisionTotal[:,0:3], particlesTotal[:,3:6] + particlesTotal[:,0:3], p=2.0)
dist_both = torch.add(dist_A, dist_B)

# Find minarg for each collumn (particle)
mina = torch.argmin(dist_both, dim=0)

# Check if DOT is negative with primitive it intersects == inside the geometry
normalOfChosen = collisionTotal[:,3:6].index_select(0, mina)
dotprod = torch.sum(particlesTotal[:,0:3] * normalOfChosen, dim=-1).double()

# Initialize intersect tensor, if particles is facing back-face, it's value stays, otherwise it's set to -1
intersection = torch.zeros(1,ptnums)
intersection = torch.where(dotprod < 0.0, mina, -1)

# Append intersection as 13th value for each particle
particlesTotal[:,-1] = intersection
print("intersection: ")
print(intersection)

mina_export = torch.flatten(mina).double().cpu().numpy()
geo.setPointFloatAttribValues("mina", mina_export)
intersectedPrims = intersection[intersection!=-1].int()
print(intersectedPrims)

# indices of particles that intersected
intersectedPtnums = (intersection != -1).nonzero(as_tuple=True)[0]


# ----- PROJECT RAY ONTO PRIMITIVE ----
##########################

init = particlesTotal[:,0:3].index_select(0, intersectedPtnums) - collisionTotal[:,0:3].index_select(0, intersectedPrims)
print("init: ")
print(init)

first = torch.sum(collisionTotal[:,3:6].index_select(0, intersectedPtnums) * init, dim=1)
print("first: ")
print(first)

second = torch.sum(collisionTotal[:,3:6].index_select(0, intersectedPrims) * -particlesTotal[:,3:6].index_select(0, intersectedPtnums), dim=1)
print("second: ")
print(second)

third = first/second
print("third: ")
print(third)

print("particlesTotal[:,3:6]: ")
print(torch.transpose(particlesTotal[:,3:6].index_select(0, intersectedPtnums), dim0=0, dim1=1))

length = third * torch.transpose(particlesTotal[:,3:6].index_select(0, intersectedPtnums), dim0=0, dim1=1)
length = torch.transpose(length, dim0=0, dim1=1)
length += particlesTotal[:,0:3].index_select(0, intersectedPtnums)
print("length: ")
print(length)

print("original pos: ")
print(particlesTotal[:,0:3])
print("projected pos: ")
particlesTotal[:,0:3].index_copy_(0, intersectedPtnums, length)
print(particlesTotal[:,0:3])

final_vel = torch.flatten(length).cpu().numpy()
geo.setPointFloatAttribValuesFromString("P", final_vel)

mina_export = torch.flatten(mina).double().cpu().numpy()
print(mina_export)
geo.setPointFloatAttribValues("mina", mina_export)




# ----- REFLECTION OF VECTOR ----
##########################


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