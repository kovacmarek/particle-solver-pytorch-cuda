node = hou.pwd()
geo = node.geometry()

# Add code to modify contents of geo.
# Use drop down menu to select examples.

import torch
import numpy as np
import time
import matplotlib.pyplot as plt

torch.manual_seed(0)

simFrame = int(hou.frame()) - 1000

staticPtnums = None
if simFrame == 1:
    hou.session.staticSimulation = None
    hou.session.staticPtnums = None
    staticPtnums = len(geo.points())
    hou.session.staticPtnums = staticPtnums
else:
    staticPtnums = hou.session.staticPtnums

# pos = torch.zeros(staticPtnums,3, device='cuda')
# vel = torch.rand(staticPtnums,3, device='cuda')
# mass = torch.ones(staticPtnums,1, device='cuda')

# total = torch.zeros(staticPtnums,7)
# total[:,0:3] = pos[:,:]
# total[:,3:6] = vel[:,:]
# total[:,-1] = mass[0,:]
# print(total)

# Globals
negative_vector = torch.tensor([-1.0, -1.0, -1.0], device='cuda')
TIME  = 0.2
start_time = time.time()
class Gravity:
    def __init__(self, total) -> None:
        self.Total = total
        self.Acc = torch.zeros(staticPtnums,3, device='cuda')
        self.Acc[:,1] = -9.8 
        self.Acc[:,0] = torch.add(self.Acc[:,0], torch.randn(staticPtnums, device='cuda'))
        self.Acc[:,2] = torch.add(self.Acc[:,0], torch.randn(staticPtnums, device='cuda'))

    def Apply(self):
        mass = self.Total[:,-1]
        acc = torch.transpose(self.Acc, 0, 1)
        return torch.transpose(mass * acc, dim0=0,dim1=1) # staticPtnums x 3

class Damping:
    def __init__( self, total, scaling = -1.0 ):
        self.Total = total
        self.Scaling = torch.tensor([scaling, scaling, scaling], device='cuda')
    def Apply( self ):
        return torch.mul(self.Total[:,3:6], self.Scaling )

class Ground:
    def __init__( self, total, loss = 0.9 ):
        self.Total = total
        self.Loss = torch.ones(1,staticPtnums, device='cuda')
        self.Loss = loss

        # print("loss: ")
        # print(self.Loss)
        
    def Apply( self ):              
        # Create Boolean collision mask
        collision_mask = torch.where(self.Total.double()[:,1] <= 0.0, True, False) * -1
        collision_mask = collision_mask.double()
        collision_mask = torch.where(collision_mask == 0.0, 1.0, collision_mask * self.Loss)

        # Apply Pos
        self.Total[:,1] = torch.t(self.Total[:,1]) * collision_mask

        # Apply Vel
        self.Total[:,4] = torch.t(self.Total[:,4]) * collision_mask

 

class Simulation:
    def __init__(self) -> None:
        # self.Total = torch.zeros(staticPtnums,7, device='cuda')
        self.Forces = []
        self.Constraints = []
        pass

    def update(self):
        sumForce = torch.zeros(staticPtnums,3, device='cuda') # reset all forces

        # Accumulate Forces
        for force in self.Forces:
            a = force.Apply() # staticPtnums x 3
            sumForce += torch.add(sumForce, a)
        
        # Symplectic Euler Integration
        acc = torch.zeros(staticPtnums,3, device='cuda')        
        normalized_mass = torch.div(1.0, self.Total[:,-1])
        acc = torch.transpose(torch.mul(torch.transpose(sumForce, dim0=0, dim1=1), normalized_mass), dim0=0, dim1=1)
        self.Total[:,3:6] += acc * TIME 
        self.Total[:,0:3] += self.Total[:,3:6] * TIME 

        for constraint in self.Constraints: # Apply constraints
            Total = constraint.Apply()
        
        return self.Total # RETURN RESULT

    def BouncingParticles(self):
        init_tensor = geo.pointFloatAttribValues("P")
        t_init_tensor = torch.tensor(init_tensor, device='cuda')
        pos = t_init_tensor.reshape(staticPtnums,3)
        
        pos[:,1] = 150
        #pos[:,0] = 0
        #pos[:,2] = 10
        vel = torch.rand(staticPtnums,3, device='cuda')
        mass = torch.ones(staticPtnums,1, device='cuda')
        mass[:,0] = 10

        total = torch.zeros(staticPtnums,7, device='cuda')
        total[:,0:3] = pos[:,:]
        total[:,3:6] = vel[:,:]
        total[:,-1] = mass[0,:]
        self.Total = total

        self.Forces.append(Gravity(total))
        self.Forces.append(Damping(total))

        self.Constraints.append(Ground(total))

staticSimulation = hou.session.staticSimulation
print(staticSimulation)
print(simFrame)

if simFrame == 1:
    print("new sim")
    staticSimulation = Simulation()
    hou.session.staticSimulation = staticSimulation
    staticSimulation.BouncingParticles()
else:
    final = staticSimulation.update().cpu()
    final_array = torch.flatten(final[:,0:3]).tolist()
    geo.setPointFloatAttribValues("P", final_array)
end_time = time.time()    
#iter = 0

#while iter < simFrame:
#    iter += 1
    # print("---------------------------ITERATION " + str(iter) + " ---------------------------")


print("Compute time for " + str(staticPtnums) + " particles: " + str(end_time - start_time))
# print(staticSimulation.total)