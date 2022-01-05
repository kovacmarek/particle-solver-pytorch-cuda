import torch
import numpy as np
import time

torch.manual_seed(0)
ptnums = 10

# pos = torch.zeros(ptnums,3, device='cuda')
# vel = torch.rand(ptnums,3, device='cuda')
# mass = torch.ones(ptnums,1, device='cuda')

# total = torch.zeros(ptnums,7)
# total[:,0:3] = pos[:,:]
# total[:,3:6] = vel[:,:]
# total[:,-1] = mass[0,:]
# print(total)

# Globals
negative_vector = torch.tensor([-1.0, -1.0, -1.0], device='cuda')
TIME  = 1

class Gravity:
    def __init__(self, total) -> None:
        self.Total = total
        self.Acc = torch.zeros(ptnums,3, device='cuda')
        self.Acc[:,:] = -9.8 

    def Apply(self):
        mass = self.Total[:,-1]
        acc = torch.transpose(self.Acc, 0, 1)
        return torch.transpose(mass * acc, dim0=0,dim1=1) # ptnums x 3

class Damping:
    def __init__( self, total, scaling = -0.25 ):
        self.Total = total
        self.Scaling = torch.tensor([scaling, scaling, scaling], device='cuda')
    def Apply( self ):
        return torch.mul(self.Total[:,3:6], self.Scaling )

class Ground:
    def __init__( self, total, loss = 1.0 ):
        self.Total = total
        self.Loss = torch.tensor([loss, loss, loss], device='cuda')
        
    def Apply( self ):              
        # Create Boolean collision mask
        collision_mask = torch.where(self.Total.double()[:,1] <= 0.0, True, False) * -1
        collision_mask = torch.where(collision_mask == 0, 1, collision_mask)

        # Apply Pos & Vel reverse
        self.Total[:,0:6] = torch.transpose(torch.mul(torch.transpose(self.Total[:,0:6], dim0=0,dim1=1), collision_mask),dim0=0,dim1=1)
        return self.Total   

class Simulation:
    def __init__(self) -> None:
        # self.Total = torch.zeros(ptnums,7, device='cuda')
        self.Forces = []
        self.Constraints = []
        pass

    def update(self):
        sumForce = torch.zeros(ptnums,3, device='cuda') # reset all forces

        # Accumulate Forces
        for force in self.Forces:
            a = force.Apply() # ptnums x 3
            sumForce += torch.add(sumForce, a)

        # print("sumForce: ")
        # print(sumForce[5:6,:])
        # print("\n")
        
        # print("before velocity: ")
        # print(self.Total[5:6,1])
        # print("\n")
        
        # Symplectic Euler Integration
        acc = torch.zeros(ptnums,3, device='cuda')        
        normalized_mass = torch.div(1.0, self.Total[:,-1])
        acc = torch.transpose(torch.mul(torch.transpose(sumForce, dim0=0, dim1=1), normalized_mass), dim0=0, dim1=1)
        self.Total[:,3:6] += acc * TIME 
        self.Total[:,0:3] += self.Total[:,3:6] * TIME 

        # print("before constraint: ")
        # print(self.Total[5:6,1])
        # print("\n")

        for constraint in self.Constraints: # Apply constraints
            Total = constraint.Apply()
        
        print("after constraint: ")
        print(self.Total[5:6,1])
        print("\n")

    def BouncingParticles(self):
        pos = torch.zeros(ptnums,3, device='cuda')
        pos[:,:] = 150
        vel = torch.rand(ptnums,3, device='cuda')
        mass = torch.ones(ptnums,1, device='cuda')
        mass[:,0] = 2

        total = torch.zeros(ptnums,7, device='cuda')
        total[:,0:3] = pos[:,:]
        total[:,3:6] = vel[:,:]
        total[:,-1] = mass[0,:]
        self.Total = total
        # self.Total[5,1] = -1.5 # TEMPORARY

        self.Forces.append(Gravity(total))
        self.Forces.append(Damping(total))

        self.Constraints.append(Ground(total))
someSim = Simulation()
someSim.BouncingParticles()

iter = 0
while iter < 10:
    iter += 1
    print("---------------------------ITERATION " + str(iter) + " ---------------------------")
    someSim.update()
# print(someSim.total)