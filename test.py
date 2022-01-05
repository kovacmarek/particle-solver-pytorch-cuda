iteration = 2
input1 = [1, 2, 3]

class Simulation:
    def __init__(self, values) -> None:
        self.Total = values
        pass

    def BouncingParticles(self):
        self.Total = [1, 2, 3]

    def update(self):
        if iteration == 2:
            self.Total = [4, 5, 6]
        if iteration == 3:
            self.Total = [7, 8, 9]

someSim = Simulation(input1)
if iteration == 1:
    someSim.BouncingParticles()
else:
    someSim.update()
output1 = someSim.values
print(output1)