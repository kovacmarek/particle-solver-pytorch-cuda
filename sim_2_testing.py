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
posOfChosen = collisionTotal[:,0:3].index_select(0, mina)
dotprod = torch.sum(normalOfChosen * (particlesTotal[:,0:3] - posOfChosen), dim=-1).double() # corrected dot


# Initialize intersect tensor, if particles is facing back-face, it's value stays, otherwise it's set to -1
intersection = torch.zeros(1,ptnums)
intersection = torch.where(dotprod < 0.0, mina, -1)

# Append intersection as 13th value for each particle
particlesTotal[:,-1] = intersection
print("intersection: ")
print(intersection)
print("\n")

mina_export = torch.flatten(mina).double().cpu().numpy()
geo.setPointFloatAttribValues("mina", mina_export)
intersectedPrims = intersection[intersection!=-1].int()
print("intersectedPrims: ")
print(intersectedPrims)
print("\n")


# indices of particles that intersected
intersectedPtnums = (intersection != -1).nonzero(as_tuple=True)[0]

print("intersectedPtnums: ")
print(intersectedPtnums)
print("\n")

#########################################
# ----- PROJECT RAY ONTO PRIMITIVE ----
#########################################

init = particlesTotal[:,0:3].index_select(0, intersectedPtnums) - collisionTotal[:,0:3].index_select(0, intersectedPrims)

first = torch.sum(collisionTotal[:,3:6].index_select(0, intersectedPrims) * init, dim=1)
second = torch.sum(collisionTotal[:,3:6].index_select(0, intersectedPrims) * -particlesTotal[:,3:6].index_select(0, intersectedPtnums), dim=1)
third = first/second

projectedPos = third * torch.transpose(particlesTotal[:,3:6].index_select(0, intersectedPtnums), dim0=0, dim1=1)
projectedPos = torch.transpose(projectedPos, dim0=0, dim1=1)
projectedPos += particlesTotal[:,0:3].index_select(0, intersectedPtnums)
print("projectedPos: ")
print(projectedPos)
print("\n")

#########################################
# ----- REFLECTION OF VECTOR ----
#########################################

# Compute normal from current position of the particle to projected position on the prim
correct_ParticleNormal = particlesTotal[:,0:3].index_select(0, intersectedPtnums) - projectedPos 

# Initialize / Normalize
normal = collisionTotal[:,3:6].index_select(0, intersectedPrims)
N_normal = f.normalize(normal, p=2, dim=0)
N_ParticleNormal = f.normalize(correct_ParticleNormal, p=2, dim=0)

# Reflection vector
Vb = 2*(torch.sum(normal * N_ParticleNormal , dim=-1))
Vb = (Vb.reshape(intersectedPtnums.size(0),1) * normal) # TODO NUMBER 6 TO BE NUMBER OF ITEMS IN THE ARRAY: intersectedPtnums
Vb -= N_ParticleNormal
Vb *= -1

# Correcting normal vector
normalScale = N_ParticleNormal / correct_ParticleNormal
Vb = Vb / normalScale

# Setting variables
Vb_final = projectedPos + Vb # Set new position
final_pos = particlesTotal[:,0:3].index_copy_(0, intersectedPtnums, Vb_final) # INSERT POSITION AT GIVEN INDICES
final_pos_f = torch.flatten(final_pos).cpu().numpy()
geo.setPointFloatAttribValuesFromString("P", final_pos_f)

yo = projectedPos - Vb_final
final_vel = particlesTotal[:,3:6].index_copy_(0, intersectedPtnums, yo)
final_vel = torch.flatten(final_vel).cpu().numpy()
geo.setPointFloatAttribValuesFromString("N", final_vel)

