node = hou.pwd()
geo = node.geometry()
inputs = node.inputs()
geo1 = inputs[1].geometry()

# Add code to modify contents of geo.
# Use drop down menu to select examples.

import torch
import numpy as np
import time
import matplotlib.pyplot as plt

torch.manual_seed(0)
simFrame = int(hou.frame()) - 1000

# Load to RAM at the begining 
staticPtnums = None
if simFrame == 1:
    hou.session.staticSimulation = None
    hou.session.staticPtnums = None
    hou.session.staticCollisionPtnums = None
    staticPtnums = len(geo.points())
    staticCollisionPtnums = len(geo1.points())
    hou.session.staticPtnums = staticPtnums
else:
    staticPtnums = hou.session.staticPtnums
    staticCollisionPtnums = hou.session.staticCollisionPtnums

# Globals
negative_vector = torch.tensor([-1.0, -1.0, -1.0], device='cuda')
TIME  = 0.2
start_time = time.time()

class Gravity:
    def __init__(self, total) -> None:
        self.Total = total
        self.Acc = torch.zeros(staticPtnums,3, device='cuda')
        self.Acc[:,1] = -9.8 # Y-axis
        # self.Acc[:,0] = torch.add(self.Acc[:,0], torch.randn(staticPtnums, device='cuda'))
        # self.Acc[:,2] = torch.add(self.Acc[:,0], torch.randn(staticPtnums, device='cuda'))

    def Apply(self):
        mass = self.Total[:,-1]
        acc = torch.transpose(self.Acc, 0, 1)
        return torch.transpose(mass * acc, dim0=0,dim1=1) # staticPtnums x 3

class Noise:
    def __init__(self, total) -> None:
        self.Total = total
        self.Acc = torch.zeros(staticPtnums,3, device='cuda')
        self.Acc[:,0] = torch.add(self.Acc[:,0], torch.randn(staticPtnums, device='cuda')) # X
        torch.manual_seed(1)
        self.Acc[:,1] = torch.add(self.Acc[:,0], torch.randn(staticPtnums, device='cuda')) # Y
        torch.manual_seed(2)
        self.Acc[:,2] = torch.add(self.Acc[:,0], torch.randn(staticPtnums, device='cuda')) # Z

    def Apply(self):
        mass = self.Total[:,-1]
        acc = torch.transpose(self.Acc, 0, 1)
        torch.manual_seed(0) # reset seed
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
        
    def Apply( self ):              
        # Create Boolean collision mask
        collision_mask = torch.where(self.Total.double()[:,1] <= 0.0, True, False) * -1
        collision_mask = collision_mask.double()
        collision_mask = torch.where(collision_mask == 0.0, 1.0, collision_mask * self.Loss)

        # Apply Pos
        self.Total[:,1] = torch.t(self.Total[:,1]) * collision_mask

        # Apply Vel
        self.Total[:,4] = torch.t(self.Total[:,4]) * collision_mask

class Collision:
    def __init__( self, total, collision, loss = 0.9 ):
        self.Total = total
        self.Loss = torch.ones(1,staticPtnums, device='cuda')
        self.Loss = loss
        self.CollisionGeo = collision
        
    def Apply( self ):              

        # input primitive's position and normal
        planeNormal = self.CollisionGeo[:,3:6]
        planeOrigin = self.CollisionGeo[:,0:3]


        # find intersect point position on all primitive
        first = torch.matmul(planeNormal, self.Total[:,0:3] - planeOrigin)
        print("first: ")
        print(first)
        print(-self.Total[:,3:6])
        second = torch.matmul(planeNormal, torch.transpose(-self.Total[:,3:6], dim0=0, dim1=1))
        print("second: ")
        print(second)

        third = first/second
        print("third: ")
        print(third)

        positionOnPlane = self.Total[:,0:3] + third * self.Total[:,3:6]
        print("PositionOnPlane: ")
        
        print(positionOnPlane)

 

class Simulation:
    def __init__(self) -> None:
        self.Forces = []
        self.Constraints = []
        pass

    def update(self):
        sumForce = torch.zeros(staticPtnums,3, device='cuda') # reset all forces

        # Accumulate Forces
        for force in self.Forces:
            a = force.Apply()
            sumForce += torch.add(sumForce, a)
        
        # Symplectic Euler Integration
        acc = torch.zeros(staticPtnums,3, device='cuda')        
        normalized_mass = torch.div(1.0, self.Total[:,-1])
        acc = torch.transpose(torch.mul(torch.transpose(sumForce, dim0=0, dim1=1), normalized_mass), dim0=0, dim1=1)
        self.Total[:,3:6] += acc * TIME 
        self.Total[:,0:3] += self.Total[:,3:6] * TIME 
        
        
        # Apply constraints
        for constraint in self.Constraints:
            constraint.Apply()
        
        return self.Total # RETURN RESULT

    def InitialState(self):
        # --- GET POINTS ---
        init_tensor = geo.pointFloatAttribValues("P")
        t_init_tensor = torch.tensor(init_tensor, device='cuda')
        pos = t_init_tensor.reshape(staticPtnums,3)

        # --- GET COLLISION ---
        # Fetch position
        init_collision_pos = geo1.pointFloatAttribValues("P")
        t_collision_pos = torch.tensor(init_collision_pos, device='cuda')
        t_collision_pos = t_collision_pos.reshape(staticCollisionPtnums,3)

        # Fetch normal
        init_collision_norm = geo1.pointFloatAttribValues("N") 
        t_collision_norm = torch.tensor(init_collision_norm, device='cuda')
        t_collision_norm = t_collision_norm.reshape(staticCollisionPtnums,3)
        
        
        # Append pos & norm to self.CollisionGeo tensor
        self.CollisionGeo = torch.zeros(staticCollisionPtnums,6, device='cuda')
        self.CollisionGeo[:,0:3] = t_collision_pos # append position
        self.CollisionGeo[:,3:6] = t_collision_norm # append normal
        # print(self.CollisionGeo)
        
        # --- INITIALIZE ATTRIBUTES ---
        pos[:,1] = 150
        vel = torch.rand(staticPtnums,3, device='cuda')
        mass = torch.ones(staticPtnums,1, device='cuda')
        mass[:,0] = 10

        # --- APPEND POINT ATTRIBS TO TENSOR ---
        total = torch.zeros(staticPtnums,7, device='cuda')
        total[:,0:3] = pos[:,:]
        total[:,3:6] = vel[:,:]
        total[:,-1] = mass[0,:]
        self.Total = total

        # self.Forces.append(Gravity(total))
        self.Forces.append(Damping(total))
        self.Forces.append(Noise(total))

        self.Constraints.append(Ground(total))
        self.Constraints.append(Collision(total,self.CollisionGeo))


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

print("Compute time for " + str(staticPtnums) + " particles: " + str(end_time - start_time))
