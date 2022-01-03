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

class Gravity:
    def __init__(self, total) -> None:
        self.Total = total
        self.Acc = torch.zeros(ptnums,3, device='cuda')
        self.Acc[:,:] = -9.8 

    def Apply(self):
        a = self.Total[:,-1]
        b = torch.transpose(self.Acc, 0, 1)
        return torch.transpose(b * a, 1, 0)


class Simulation:
    def __init__(self) -> None:
        self.total = None
        self.Forces = []
        pass


    def update(self):
        sumForce = torch.zeros(ptnums,3, device='cuda') # reset all forces
        for force in self.Forces:
            a = force.Apply()
            sumForce = torch.add(sumForce, a)

    def BouncingParticles(self):
        pos = torch.zeros(ptnums,3, device='cuda')
        vel = torch.rand(ptnums,3, device='cuda')
        mass = torch.ones(ptnums,1, device='cuda')
        mass[:,0] = 3

        total = torch.zeros(ptnums,7, device='cuda')
        total[:,0:3] = pos[:,:]
        total[:,3:6] = vel[:,:]
        total[:,-1] = mass[0,:]
        self.total = total

        self.Forces.append(Gravity(total))
        self.Forces.append(Gravity(total))

someSim = Simulation()
someSim.BouncingParticles()
someSim.update()
print(someSim.total)