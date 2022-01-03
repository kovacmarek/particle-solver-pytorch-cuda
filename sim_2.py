import torch
import numpy as np
import time

torch.manual_seed(0)
ptnums = 10

init_tensor = torch.rand(ptnums,3, device='cuda')
negative_vector = torch.tensor([-1.0, -1.0, -1.0], device='cuda')

class Particle:
    def __init__( self, position, velocity, mass ):
        self.Position = position
        self.Velocity = torch.zeros(ptnums,3, device='cuda')
        self.SumForce = torch.zeros(ptnums,3, device='cuda')
        self.Mass     = torch.zeros(ptnums,1, device='cuda')       
    def Display( self ):
        return [self.Position]
 
class Gravity:
    def __init__( self, particles, acceleration = 9.8 ):
        self.Particles    = particles
        self.Acceleration = torch.mult(torch.tensor([acceleration, acceleration, acceleration], device='cuda'), negative_vector)
    def Apply( self ):
        self.Particles.SumForce += torch.mul(self.Particles.Mass, self.Acceleration )

class Damping:
    def __init__( self, particles, scaling = 1.0 ):
        self.Particles = particles
        self.Scaling   = torch.tensor([scaling, scaling, scaling], device='cuda')
    def Apply( self ):
        self.Particles.SumForce += torch.mul(self.Particles.Velocity, -self.Scaling )

class Ground:
    def __init__( self, particles, loss = 1.0 ):
        self.Particles = particles
        self.Loss = torch.tensor([loss, loss, loss])
    def Apply( self ):   
        for particle in self.Particles:            
            if( self.Particles[0][:,1] < 0):
                particle.Position *= negative_vector
                particle.Velocity *= negative_vector
                particle.Velocity = torch.mul(particle.Velocity, self.Loss)


class Simulation:
    def __init__( self ):
        self.Particles   = [] # particle
        self.Forces      = [] # drag, gravity
        self.Constraints = [] # ground
        
    def Update( self, dt ):        
        for particle in self.Particles:       #-- Zero All Sums of Forces
            particle.SumForce = Vector3d.Zero
            
        for force in self.Forces:             #-- Accumulate Forces
            force.Apply( )
            
        for particle in self.Particles:       #-- Symplectic Euler Integration
            if( particle.Mass == 0 ): continue
            acceleration = particle.SumForce * ( 1.0 / particle.Mass )
            particle.Velocity += acceleration * dt
            particle.Position += particle.Velocity * dt
            
        for constraint in self.Constraints:   #-- Apply Penalty Constraints
            constraint.Apply( )
        
    def Display( self ):
        #-- Geometry
        #--
        geometry = []        
        for particle in self.Particles:
            geometry += particle.Display( )
        
        #-- Messages
        #--
        ke = self.KineticEnergy( )
        pe = self.PotentialEnergy( )
        print( "Kinetic   {0}".format( ke      ) )
        print( "Potential {0}".format( pe      ) )
        print( "Total     {0}".format( ke + pe ) )
        
        return geometry
        
    def BouncingParticles( self ):
        #-- A number of particles along X-Axis with increasing mass
        particle = Particle( Point3d( index, 0, 100 ), Vector3d.Zero, index + 1 )
        self.Pos.append( particle )
        
        #-- Setup forces
        gravity = Gravity( self.Particles )
        self.Forces.append( gravity )
        
        drag = Damping( self.Particles, 0.1 )
        self.Forces.append( drag )
        
        #-- Ground constraint
        ground = Ground( self.Particles, 0.5 )
        self.Constraints.append( ground )

if "simulation" in vars( ) and not reset:
    for iterations in range( 10 ): #-- Non-Visual Updates
        simulation.Update( dt )
    geometry = simulation.Display( )
else:                              #-- Reset & Construction
    simulation = Simulation( )
    simulation.BouncingParticles( )
    geometry = simulation.Display( )




start_time = time.time()
end_time = time.time()

# print("---------")
# print("Initial position: " + str(init_tensor[9]))
# print("Vector to add: " + str(final_tensor_i[closest_point_num]))
# print("Final position: " + str(final_pos[9]))
print("--------")
print("Get distances time: " + str(end_time - start_time))

