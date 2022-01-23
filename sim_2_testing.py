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
particlesTotal = torch.zeros(ptnums,7, device='cuda') # 7th value is boolean if it's intersecting

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

start_time = time.time()

class FindIntersection():
    def __init__(self, particles, collision) -> None:
        self.particlesTotal = particles
        self.collisionTotal = collision 

    def compute(self):
        # Compute distance
        dist_A = torch.cdist(self.collisionTotal[:,0:3], self.particlesTotal[:,0:3], p=2.0)
        dist_B = torch.cdist(self.collisionTotal[:,0:3], self.particlesTotal[:,3:6] + self.particlesTotal[:,0:3], p=2.0)
        dist_both = torch.add(dist_A, dist_B)

        # Find minarg for each collumn (particle)
        mina = torch.argmin(dist_both, dim=0)

        # Check if DOT is negative with primitive it intersects == inside the geometry
        normalOfChosen = self.collisionTotal[:,3:6].index_select(0, mina)
        posOfChosen = self.collisionTotal[:,0:3].index_select(0, mina)
        dotprod = torch.sum(normalOfChosen * (self.particlesTotal[:,0:3] - posOfChosen), dim=-1).double() # corrected dot

        # Initialize intersect tensor, if particles is facing back-face, it's value stays, otherwise it's set to -1
        self.intersection = torch.zeros(1,ptnums)
        self.intersection = torch.where(dotprod < 0.0, mina, -1)

        # Append self.intersection as 13th value for each particle
        self.particlesTotal[:,-1] = self.intersection

        mina_export = torch.flatten(mina).double().cpu().numpy()
        geo.setPointFloatAttribValues("mina", mina_export)
        self.intersectedPrims = self.intersection[self.intersection!=-1].int()

        # indices of particles that intersected
        self.intersectedPtnums = (self.intersection != -1).nonzero(as_tuple=True)[0]

#########################################
# ----- PROJECT RAY ONTO PRIMITIVE ----
#########################################

class ProjectOntoPrim():
    def __init__(self, particles, collision) -> None:
        self.particlesTotal = particles
        self.collisionTotal = collision   

    def compute(self):
        init = self.particlesTotal[:,0:3].index_select(0, self.intersectedPtnums) - self.collisionTotal[:,0:3].index_select(0, self.intersectedPrims)

        first = torch.sum(self.collisionTotal[:,3:6].index_select(0, self.intersectedPrims) * init, dim=1)
        second = torch.sum(self.collisionTotal[:,3:6].index_select(0, self.intersectedPrims) * -self.particlesTotal[:,3:6].index_select(0, self.intersectedPtnums), dim=1)
        third = first/second

        self.projectedPos = third * torch.transpose(self.particlesTotal[:,3:6].index_select(0, self.intersectedPtnums), dim0=0, dim1=1)
        self.projectedPos = torch.transpose(self.projectedPos, dim0=0, dim1=1)
        self.projectedPos += self.particlesTotal[:,0:3].index_select(0, self.intersectedPtnums)

#########################################
# ----- REFLECTION OF VECTOR ----
#########################################

class ReflectVector():
    def __init__(self, particles, collision) -> None:
        self.particlesTotal = particles
        self.collisionTotal = collision  

    def compute(self):
        # Compute normal from current position of the particle to projected position on the prim
        correct_ParticleNormal = particlesTotal[:,0:3].index_select(0, self.intersectedPtnums) - self.projectedPos 

        # Initialize / Normalize
        normal = collisionTotal[:,3:6].index_select(0, self.intersectedPrims)
        N_normal = f.normalize(normal, p=2, dim=0)
        N_ParticleNormal = f.normalize(correct_ParticleNormal, p=2, dim=0)

        # Reflection vector
        Vb = 2*(torch.sum(normal * N_ParticleNormal , dim=-1))
        Vb = (Vb.reshape(self.intersectedPtnums.size(0),1) * normal)
        Vb -= N_ParticleNormal
        Vb *= -1

        # Correcting normal vector
        normalScale = N_ParticleNormal / correct_ParticleNormal
        Vb = Vb / normalScale

        # Setting variables
        Vb_final = self.projectedPos + Vb # Set new position
        final_pos = particlesTotal[:,0:3].index_copy_(0, self.intersectedPtnums, Vb_final) # INSERT POSITION AT GIVEN INDICES
        final_v = self.projectedPos - Vb_final
        final_vel = particlesTotal[:,3:6].index_copy_(0, self.intersectedPtnums, final_v) # INSERT VELOCITY AT GIVEN INDICES

        return final_pos, final_vel

results = ReflectVector()
final_pos = results.compute()[0]
final_vel = results.compute()[1]

final_pos = torch.flatten(final_pos).cpu().numpy() # Flatten & copy to CPU
geo.setPointFloatAttribValuesFromString("P", final_pos) # Houdini set

final_vel = torch.flatten(final_vel).cpu().numpy() # Flatten & copy to CPU
geo.setPointFloatAttribValuesFromString("N", final_vel) # Houdini set

