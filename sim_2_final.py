import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as f

node = hou.pwd()
geo = node.geometry()
inputs = node.inputs()
geo1 = inputs[1].geometry()

ptnums = len(geo.points())
collisionPtnums = len(geo1.points())

torch.manual_seed(0)
simFrame = int(hou.frame()) - 1000

# Load to RAM at the begining 
ptnums = None
if simFrame == 1:
    hou.session.staticSimulation = None
    hou.session.ptnums = None
    hou.session.collisionPtnums = None
    ptnums = len(geo.points())
    collisionPtnums = len(geo1.points())
    hou.session.ptnums = ptnums
else:
    ptnums = hou.session.ptnums
    collisionPtnums = hou.session.collisionPtnums

# Globals
negative_vector = torch.tensor([-1.0, -1.0, -1.0], device='cuda')
TIME  = 0.2
start_time = time.time()

class Gravity:
    def __init__(self, total) -> None:
        self.particlesTotal = total
        self.Acc = torch.zeros(ptnums,3, device='cuda')
        self.Acc[:,1] = -9.8 # Y-axis
        # self.Acc[:,0] = torch.add(self.Acc[:,0], torch.randn(ptnums, device='cuda'))
        # self.Acc[:,2] = torch.add(self.Acc[:,0], torch.randn(ptnums, device='cuda'))

    def Apply(self):
        mass = self.particlesTotal[:,6]
        acc = torch.transpose(self.Acc, 0, 1)
        return torch.transpose(mass * acc, dim0=0,dim1=1) # ptnums x 3

class Noise:
    def __init__(self, total) -> None:
        self.particlesTotal = total
        self.Acc = torch.zeros(ptnums,3, device='cuda')
        self.Acc[:,0] = torch.add(self.Acc[:,0], torch.randn(ptnums, device='cuda')) # X
        torch.manual_seed(1)
        self.Acc[:,1] = torch.add(self.Acc[:,0], torch.randn(ptnums, device='cuda')) # Y
        torch.manual_seed(2)
        self.Acc[:,2] = torch.add(self.Acc[:,0], torch.randn(ptnums, device='cuda')) # Z

    def Apply(self):
        mass = self.particlesTotal[:,-1]
        acc = torch.transpose(self.Acc, 0, 1)
        torch.manual_seed(0) # reset seed
        return torch.transpose(mass * acc, dim0=0,dim1=1) # ptnums x 3

class Damping:
    def __init__( self, total, scaling = -1.0 ):
        self.particlesTotal = total
        self.Scaling = torch.tensor([scaling, scaling, scaling], device='cuda')
    def Apply( self ):
        return torch.mul(self.particlesTotal[:,3:6], self.Scaling )

class CollisionDetection():
    def __init__(self, particles, collision) -> None:
        self.particlesTotal = particles
        self.collisionTotal = collision

    def selfCollision(self):
        particleDistance = f.pdist(self.particlesTotal[:,0:3], p=2.0)
        print(particleDistance)

        minaParticle = torch.argmin(particleDistance, dim=0)
        print(minaParticle)
        print(torch.subtract(self.particlesTotal[:,0:3].unsqueeze(0), self.particlesTotal[:,0:3].unsqueeze(1)))

        # self.particlesTotal[:,0:3] += firstCompute_N.index_select(0, closeParticlesIndicies[:,0])

    def findIntersection(self):
        # Compute distance
        dist_A = torch.cdist(self.collisionTotal[:,0:3], self.particlesTotal[:,0:3], p=2.0)
        # dist_B = torch.cdist(self.collisionTotal[:,0:3], self.particlesTotal[:,3:6] + self.particlesTotal[:,0:3], p=2.0)
        # dist_both = torch.add(dist_A, dist_B)

        # Find minarg for each collumn (particle)
        mina = torch.argmin(dist_A, dim=0)

        # Clean unoccupied GPU memory cache
        torch.cuda.empty_cache()

        # Check if DOT is negative with primitive it intersects == inside the geometry
        normalOfChosen = self.collisionTotal[:,3:6].index_select(0, mina)
        posOfChosen = self.collisionTotal[:,0:3].index_select(0, mina)
        dotprod = torch.sum(normalOfChosen * (self.particlesTotal[:,0:3] - posOfChosen), dim=-1).double() # corrected dot

        # Initialize intersect tensor, if particles is facing back-face, it's value stays, otherwise it's set to -1
        self.intersection = torch.zeros(1,ptnums)
        self.intersection = torch.where(dotprod < 0.0, mina, -1)

        mina_export = torch.flatten(mina).double().cpu().numpy()
        geo.setPointFloatAttribValues("mina", mina_export)

        # Append self.intersection as 13th value for each particle
        self.intersectedPrims = self.intersection[self.intersection!=-1]

        # indices of particles that intersected
        self.intersectedPtnums = (self.intersection != -1).nonzero(as_tuple=True)[0]
        # print("self.intersectedPtnums: ")
        # print(self.intersectedPtnums)

        # print("self.intersectedPrims: ")
        # print(self.intersectedPrims)

    #########################################
    # ----- PROJECT RAY ONTO PRIMITIVE ----
    #########################################

    def projectOntoPrim(self):
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

    def reflectVector(self):
        # Compute normal from current position of the particle to projected position on the prim
        correct_ParticleNormal = self.projectedPos - self.particlesTotal[:,0:3].index_select(0, self.intersectedPtnums)
              

        # Initialize / Normalize
        normal = self.collisionTotal[:,3:6].index_select(0, self.intersectedPrims)
        N_normal = f.normalize(normal, p=2, dim=0)
        N_ParticleNormal = f.normalize(correct_ParticleNormal, p=2, dim=0)

        # Reflection vector
        Vb = 2*(torch.sum(normal *correct_ParticleNormal , dim=-1))
        Vb = (Vb.reshape(self.intersectedPtnums.size(0),1) * normal)
        Vb -= N_ParticleNormal

        # ##### Export to Houdini  --------- TEMPORARY
        # correct_dir = torch.rand_like(self.particlesTotal[:,0:3]).index_copy_(0, self.intersectedPtnums, Vb)
        # correct_dir = torch.flatten(correct_dir).cpu().numpy()
        # geo.setPointFloatAttribValuesFromString("N", correct_dir)
        
        # Correcting normal vector
        normalScale = N_ParticleNormal / correct_ParticleNormal

        # Setting variables
        loss = 2
        Vb_final = (self.particlesTotal[:,3:6].index_select(0, self.intersectedPtnums) * (1.0 / loss)) + (self.projectedPos + 0.001)  # Set new position
        final_v = (self.projectedPos - Vb_final)

        self.particlesTotal[:,0:3].index_copy_(0, self.intersectedPtnums, Vb_final + final_v) # INSERT POSITION AT GIVEN INDICES
        self.particlesTotal[:,3:6].index_copy_(0, self.intersectedPtnums, Vb) # INSERT VELOCITY AT GIVEN INDICES

        self.projectedPos = torch.zeros_like(self.projectedPos) # reset zeros
        
    def Apply(self):
        self.findIntersection()
        self.projectOntoPrim()
        self.reflectVector()
        # self.selfCollision()


class Simulation:
    def __init__(self) -> None:
        self.Forces = []
        self.Constraints = []
        pass

    def InitialState(self):
        self.collisionTotal = torch.zeros(collisionPtnums,7, device='cuda') # 7th value is distance
        self.particlesTotal = torch.zeros(ptnums,8, device='cuda') # 7th value is boolean if it's intersecting

        # collision append
        init_collision_pos = geo1.pointFloatAttribValues("P") 
        t_collision_pos = torch.tensor(init_collision_pos, device='cuda')
        self.collisionTotal[:,0:3] = t_collision_pos.reshape(collisionPtnums,3)

        init_collision_norm = geo1.pointFloatAttribValues("N") 
        t_collision_norm = torch.tensor(init_collision_norm, device='cuda')
        self.collisionTotal[:,3:6] = t_collision_norm.reshape(collisionPtnums,3)

        # particles append
        init_particles_pos = geo.pointFloatAttribValues("P") 
        t_particles_pos = torch.tensor(init_particles_pos, device='cuda')
        self.particlesTotal[:,0:3] = t_particles_pos.reshape(ptnums,3)

        init_particles_norm = geo.pointFloatAttribValues("v") 
        t_particles_norm = torch.tensor(init_particles_norm, device='cuda')
        self.particlesTotal[:,3:6] = t_particles_norm.reshape(ptnums,3)
        
        # --- SET MASS ---
        mass = torch.rand(ptnums,1, device='cuda') * 10
        self.particlesTotal[:,6] = mass[0,:] # 7th value is mass, 8th is intersection boolean

        self.Forces.append(Gravity(self.particlesTotal))
        # self.Forces.append(Damping(self.particlesTotal))
        # self.Forces.append(Noise(self.particlesTotal))

        #self.Constraints.append(Ground(self.particlesTotal))
        self.Constraints.append(CollisionDetection(self.particlesTotal, self.collisionTotal))

    def update(self):

        # Clean unoccupied GPU memory cache
        torch.cuda.empty_cache()

        sumForce = torch.zeros(ptnums,3, device='cuda') # reset all forces

        # Accumulate Forces
        for force in self.Forces:
            a = force.Apply()
            sumForce += torch.add(sumForce, a) * 0.1
        
        # Symplectic Euler Integration
        acc = torch.zeros(ptnums,3, device='cuda')        
        normalized_mass = torch.div(1.0, self.particlesTotal[:,6])
        acc = torch.transpose(torch.mul(torch.transpose(sumForce, dim0=0, dim1=1), normalized_mass), dim0=0, dim1=1)
        self.particlesTotal[:,3:6] += acc * TIME 
        self.particlesTotal[:,0:3] += self.particlesTotal[:,3:6] * TIME 
        

        # Apply constraints
        for constraint in self.Constraints:
            constraint.Apply()
        
        return self.particlesTotal # RETURN RESULT


staticSimulation = hou.session.staticSimulation

if simFrame == 1:
    print("new sim")
    staticSimulation = Simulation()
    hou.session.staticSimulation = staticSimulation
    staticSimulation.InitialState()
else:
    final = staticSimulation.update()
    final_pos = torch.flatten(final[:,0:3]).cpu().numpy()
    final_vel = torch.flatten(final[:,3:6]).cpu().numpy()
    geo.setPointFloatAttribValuesFromString("P", final_pos)
    geo.setPointFloatAttribValuesFromString("v", final_vel)
    
end_time = time.time()  
# print("memory: ")
# print(torch.cuda.memory_summary(device='cuda', abbreviated=True))  

print("Compute time for " + str(ptnums) + " particles: " + str(end_time - start_time))
print("-----------------")
