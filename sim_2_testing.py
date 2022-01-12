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

collisionTotal = torch.zeros(collisionPtnums,7, device='cuda')
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
dist_A = torch.cdist(collisionTotal[:,0:3], particlesTotal[:,0:3], p=2.0)
dist_B = torch.cdist(collisionTotal[:,0:3], particlesTotal[:,3:6] + particlesTotal[:,0:3], p=2.0)
dist_both = torch.add(dist_A, dist_B)

print("dist_A: ")
print(dist_A)
print("dist_B: ")
print(dist_B)
print("dist_both: ")
print(dist_both)

# find minarg for each collumn (particle)
mina = torch.argmin(dist_both, dim=0)
print("mina: ")
print(mina)

print("particle_pos: ")
print(particlesTotal[:,0:3])

print("collisionTotal[:,0:3]: ")
print(collisionTotal[:,0:3])

mina_export = torch.flatten(mina).double().cpu().numpy()
print(mina_export)
geo.setPointFloatAttribValues("mina", mina_export)

# write to detail
geo.addArrayAttrib(hou.attribType.Global, "data", hou.attribData.Float, tuple_size=1)
geo.setGlobalAttribValue("data", mina_export)

# ----- PROJECT RAY ONTO PRIMITIVE ----
##########################

init = particlesTotal[:,0:3] - particlesTotal[:,6:9]
first = torch.sum(particlesTotal[:,9:12] * init, dim=1)
print("first: ")
print(first)

second = torch.sum(particlesTotal[:,9:12] * -particlesTotal[:,3:6], dim=1)
print("second: ")
print(second)

third = first/second
print("third: ")
print(third)

print("particlesTotal[:,3:6]: ")
print(torch.transpose(particlesTotal[:,3:6], dim0=0, dim1=1))

length = third * torch.transpose(particlesTotal[:,3:6], dim0=0, dim1=1)
length = torch.transpose(length, dim0=0, dim1=1)
length += particlesTotal[:,0:3]
print("length: ")
print(length)


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