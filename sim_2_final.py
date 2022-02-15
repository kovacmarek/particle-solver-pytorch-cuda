import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as f

node = hou.pwd()
geo = node.geometry()
inputs = node.inputs()
geo1 = inputs[1].geometry()
geo2 = inputs[2].geometry()

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
negative_vector = torch.tensor([-1.0, -1.0, -1.0], device='cuda:1')
TIME  = 0.2
start_time = time.time()

class Gravity:
    def __init__(self, total) -> None:
        self.particlesTotal = total
        self.Acc = torch.zeros(ptnums,3, device='cuda:1')
        self.Acc[:,1] = -9.8 # Y-axis
        # self.Acc[:,0] = torch.add(self.Acc[:,0], torch.randn(ptnums, device='cuda:1'))
        # self.Acc[:,2] = torch.add(self.Acc[:,0], torch.randn(ptnums, device='cuda:1'))

    def Apply(self):
        mass = self.particlesTotal[:,6]
        acc = torch.transpose(self.Acc, 0, 1)
        return torch.transpose(mass * acc, dim0=0,dim1=1) # ptnums x 3

class Noise:
    def __init__(self, total) -> None:
        self.particlesTotal = total
        self.Acc = torch.zeros(ptnums,3, device='cuda:1')
        self.Acc[:,0] = torch.add(self.Acc[:,0], torch.randn(ptnums, device='cuda:1')*5) # X
        torch.manual_seed(1)
        self.Acc[:,1] = torch.add(self.Acc[:,0], torch.randn(ptnums, device='cuda:1')*5) # Y
        torch.manual_seed(2)
        self.Acc[:,2] = torch.add(self.Acc[:,0], torch.randn(ptnums, device='cuda:1')*5) # Z

    def Apply(self):
        mass = self.particlesTotal[:,-1]
        acc = torch.transpose(self.Acc, 0, 1)
        torch.manual_seed(0) # reset seed
        return torch.transpose(mass * acc, dim0=0,dim1=1) # ptnums x 3

class Damping:
    def __init__( self, total, scaling = -1.0 ):
        self.particlesTotal = total
        self.Scaling = torch.tensor([scaling, scaling, scaling], device='cuda:1')
    def Apply( self ):
        return torch.mul(self.particlesTotal[:,3:6], self.Scaling )

class CollisionDetection():
    def __init__(self, particles, collision) -> None:
        self.particlesTotal = particles
        self.collisionTotal = collision

    def createBoundingBoxes(self):

        # APPEND BOUNDARIES
        self.boundaries = torch.zeros(8,3) # initialize tensor
        init_collision_pos = geo2.pointFloatAttribValues("P") 
        t_collision_pos = torch.tensor(init_collision_pos, device='cuda:1')
        self.boundaries = t_collision_pos.reshape(8,3)

        Boundaries_x_max = torch.max(self.boundaries[:,0]) 
        Boundaries_x_min = torch.min(self.boundaries[:,0])     
        Boundaries_y_max = torch.max(self.boundaries[:,1]) 
        Boundaries_y_min = torch.min(self.boundaries[:,1])
        Boundaries_z_max = torch.max(self.boundaries[:,2])
        Boundaries_z_min = torch.min(self.boundaries[:,2]) 

        boundaries_max_corner = torch.tensor([Boundaries_x_max, Boundaries_y_max, Boundaries_z_max])
        boundaries_min_corner = torch.tensor([Boundaries_x_min, Boundaries_y_min, Boundaries_z_min])
        
        # INSIDE BOUNDING BOXES
        padding = 0.15
        BB_x_max = torch.max(self.particlesTotal[:,0]) 
        BB_x_max += abs(BB_x_max * padding) + 1                        # Padding +15%
        BB_x_min = torch.min(self.particlesTotal[:,0])
        BB_x_min -= abs(BB_x_min * padding)                       # Padding +15%        

        BB_y_max = torch.max(self.particlesTotal[:,1]) 
        BB_y_max += abs(BB_y_max * padding) + 1                     # Padding +15%
        BB_y_min = torch.min(self.particlesTotal[:,1])
        BB_y_min -= abs(BB_y_min * padding)                      # Padding +15%

        BB_z_max = torch.max(self.particlesTotal[:,2])
        BB_z_max += abs(BB_z_max * padding) + 1                        # Padding +15%
        BB_z_min = torch.min(self.particlesTotal[:,2]) 
        BB_z_min -= abs(BB_z_min * padding)                       # Padding +15%

        BB_A = torch.tensor([BB_x_max, BB_y_min, BB_z_min])
        BB_B = torch.tensor([BB_x_min, BB_y_min, BB_z_min])
        BB_C = torch.tensor([BB_x_max, BB_y_min, BB_z_max])
        BB_D = torch.tensor([BB_x_min, BB_y_min, BB_z_max])
        BB_E = torch.tensor([BB_x_max, BB_y_max, BB_z_min])
        BB_F = torch.tensor([BB_x_min, BB_y_max, BB_z_min])
        BB_G = torch.tensor([BB_x_max, BB_y_max, BB_z_max])
        BB_H = torch.tensor([BB_x_min, BB_y_max, BB_z_max])

        BB_centroid = (BB_A + BB_B + BB_C + BB_D + BB_E + BB_F + BB_G + BB_H) / 8

        # MAX CAP
        if BB_x_max > Boundaries_x_max: BB_x_max = Boundaries_x_max 
        if BB_y_max > Boundaries_y_max: BB_y_max = Boundaries_y_max
        if BB_z_max > Boundaries_z_max: BB_z_max = Boundaries_z_max

        # MIN CAP
        if BB_x_min < Boundaries_x_min: BB_x_min = Boundaries_x_min
        if BB_y_min < Boundaries_y_min: BB_y_min = Boundaries_y_min
        if BB_z_min < Boundaries_z_min: BB_z_min = Boundaries_z_min

    #########################################
    # ----- INDEXING BOUNDING BOXES ----
    #########################################    
        
        BB_resolution = int(hou.parm('./bb_resolution').rawValue())
        self.chunks = BB_resolution**3
        xyz = torch.zeros(self.chunks,3)

        # 0,1,2,3 0,1,2,3 0,1,2,3 0,1,2,3
        iter = 0
        while iter < (self.chunks):     
                index = iter % BB_resolution 
                xyz[iter,0] = index
                iter += 1

        # # 0,0,0,0 1,1,1,1 2,2,2,2 3,3,3,3
        iter = 0
        while iter <= self.chunks:
                counter = iter * BB_resolution # 0,4,8,16
                xyz[counter:counter + BB_resolution,1] = iter % BB_resolution
                iter += 1 

        # # 0,0,0,0 0,0,0,0 0,0,0,0 0,0,0,0     1,1,1,1 1,1,1,1 1,1,1,1 1,1,1,1     2,2,2,2 2,2,2,2 2,2,2,2 2,2,2,2    3,3,3,3 3,3,3,3 3,3,3,3 3,3,3,3
        iter = 0
        while iter <= self.chunks:
                counter = iter * (BB_resolution**2) # 0,4,8,16
                xyz[counter:counter + BB_resolution**2,2] = iter % BB_resolution
                iter += 1      

    #########################################
    # ----- ASIGNING BOUNDING BOXES ----
    ######################################### 
      
        max_corner = torch.tensor([BB_x_max, BB_y_max, BB_z_max])
        min_corner = torch.tensor([BB_x_min, BB_y_min, BB_z_min])
        scene_center = torch.tensor([0,0,0])
        # print("max_min_corner: ")
        # print(max_corner)
        # print(min_corner)

        # EXPORT MIN_MAX_CORNERS
        export_min_corner = torch.flatten(min_corner).double().cpu().numpy()
        geo.setGlobalAttribValue("min_corner", export_min_corner)

        export_max_corner = torch.flatten(max_corner).double().cpu().numpy()
        geo.setGlobalAttribValue("max_corner", export_max_corner)

        # GET UNIT LENGTH FOR EACH AXIS
        x_unit = (abs(max_corner[0].item() - min_corner[0].item())) / self.chunks
        y_unit = (abs(max_corner[1].item() - min_corner[1].item())) / self.chunks
        z_unit = (abs(max_corner[2].item() - min_corner[2].item())) / self.chunks

        # MULTIPLY EACH UNIT AXIS WITH THEIR CORRESPONDING INDEX
        # PLACE BOUNDING BOXES TO THEIR CORRECT WORLD POSITION
        unit_xyz = torch.tensor([x_unit, y_unit, z_unit]) * BB_resolution
        unit_xyz *= BB_resolution

        xyz_finalmin = (xyz * unit_xyz) - (scene_center - min_corner)
        xyz_finalmax = ((xyz+1) * unit_xyz) - (scene_center - min_corner)
        
        # FINAL BOUNDING BOXES AND THEIR MIN & MAX
        self.BB_min_max = torch.cat((xyz_finalmin, xyz_finalmax), 1)

    def findWhichBoundingBox(self):
        x_axis = self.particlesTotal[:,0]
        y_axis = self.particlesTotal[:,1]
        z_axis = self.particlesTotal[:,2]

        x_min = self.BB_min_max[:,0]
        y_min = self.BB_min_max[:,1]
        z_min = self.BB_min_max[:,2]

        BB = self.BB_min_max[:,0:6]

        BB_id = torch.arange(0,len(self.BB_min_max[:,0])).reshape(len(self.BB_min_max[:,0]),1)
        BB = torch.cat((BB,BB_id),1).cuda(device='cuda:1')


        # ASIGN BLOCK ID TO EACH POINT
        for i in range(0, len(self.BB_min_max[:,0])):
            self.particlesTotal[:,7] = torch.where((x_axis > BB[i,0]) & (x_axis < BB[i,3]) & (y_axis > BB[i,1]) & (y_axis < BB[i,4]) & (z_axis > BB[i,2]) & (z_axis < BB[i,5]) , BB[i,6], self.particlesTotal[:,7])
            # TODO OPTIMIZE THIS STEP, MAYBE MULTITHREADING WITH TENSOR INDEX ASSIGNEMENT
            
        self.block_ID = torch.flatten(self.particlesTotal[:,7]).cpu().numpy()
        geo.setPointFloatAttribValuesFromString("block_id", self.block_ID) 
        # print("block_ID")
        # print(block_ID) 
        # print("block_ID: \n", self.block_ID)
        


    def selfCollision(self):
        for i in range(0, self.chunks):            
            selected = (self.particlesTotal[:,7] == i).nonzero(as_tuple=False).flatten() # Filter out only points that match current Block ID
            if len(selected) == 0: # check if there are any points in the block
                pass
            else:          
                selected_pos = self.particlesTotal[:,0:3].index_select(0, selected) # Fetch positions of the points in current block only
                selected_ptnums = len(selected)
                
                # Clean unoccupied GPU memory cache
                torch.cuda.empty_cache()

                # Start of self collision
                iterations = 2
                iter = 0
                while iter < iterations:
                    iter += 1
                    diameter = float(hou.parm('./pscale').rawValue())
                    
                    time_particle_dist = time.time()
                    # Compute distances between points
                    particle_dist = torch.cdist(selected_pos, selected_pos, p=2.0).double()

                    time_particle_dist_end = time.time()
                    
                    # Diagonal zeros set to double the diameter (to ignore these values for argmin)
                    eye = torch.zeros(selected_ptnums,selected_ptnums, device="cuda:1") + (diameter * 2)
                    eye = torch.tril(eye, diagonal=0)
                    # print("eye: \n",eye)
                    particle_dist = particle_dist + eye

                    time_find_closest_particle = time.time()
                    # print(particle_dist)
                    closest_particle = torch.argmin(particle_dist, dim=1)
                    closest_particle_value = torch.min(particle_dist, dim=1)
                    closest_particle = torch.where(closest_particle_value.values > diameter, -1, closest_particle) # filter out non penetrating (more than radius * 2)
                    closest_particle_index = closest_particle[closest_particle != -1]
                    time_find_closest_particle_end = time.time()

                    # print("closest_particle: \n", closest_particle)
                    # print("closest_particle_index: \n", closest_particle_index)
                    

                    # Correcting collision position
                    iterated_index = (closest_particle != -1).nonzero(as_tuple=False).flatten()
                    # print("iterated_index: \n", iterated_index)

                    iteratedPos = selected_pos.index_select(0, iterated_index)
                    iteratedVel = self.particlesTotal[:,3:6].index_select(0, iterated_index)
                    collisionPos = selected_pos.index_select(0, closest_particle_index)
                    collisionVel = self.particlesTotal[:,3:6].index_select(0, closest_particle_index)
                    

                    # print("iteratedPos: \n", iteratedPos)
                    # print("iteratedVel: \n", iteratedVel)
                    # print("collisionPos: \n", collisionPos)
                    # print("collisionVel: \n", collisionVel)

                    final_pos_dir = f.normalize(iteratedPos - collisionPos, p=1, dim=1)
                    # print("particle_dist: \n", particle_dist)
                    final_pos_magnitude = particle_dist[iterated_index, closest_particle_index]
                    final_pos_amount = (diameter - final_pos_magnitude) * 1.05
                    # print("final_pos_magnitude: \n", final_pos_magnitude)
                    # print("final_pos_amount: \n", final_pos_amount)
                    # print("final_pos_dir: \n", final_pos_dir)

                    
                    # print("transposed_final_pos_dir: \n", torch.transpose(final_pos_dir,dim0=0,dim1=1))
                    required_move = torch.transpose(final_pos_dir,dim0=0,dim1=1) * final_pos_amount
                    required_move = torch.transpose(required_move,dim0=0,dim1=1)
                    # print("required_move: \n", required_move)
                    final_pos = (iteratedPos + required_move/iterations).float() 
                    final_vel = (iteratedVel * 0.1 ).float() 

                    
                    # print("final_pos: \n", final_pos)
                    # print("positions: \n", selected_pos)

                    # print("---------------------------------------------------------")
                    
                    selected_pos.index_copy_(0, iterated_index, final_pos) # INPUT PENETRATED POSITIONS INTO ALL IN THIS BLOCK   
                    self.particlesTotal[:,0:3].index_copy_(0, selected, selected_pos) # INPUT ALL(penetrated & non penetrated) FROM CURRENT BLOCK INTO MAIN POSITION
                    
                    # Clean unoccupied GPU memory cache
                    torch.cuda.empty_cache()
                    

                    # print("Particle Distance: ", time_particle_dist_end - time_particle_dist)
                    # print("Find Closest Particle: ", time_find_closest_particle_end - time_find_closest_particle)
                    # print("VRAM:" + str((selected_pos.element_size() * selected_pos.nelement()) / 1000000) + " MB") 
                    # print("---------")

    def findIntersection(self):
        # TODO: MAKE THIS WORK INSIDE BOUNDING BOXES

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
        Vb = 2*(torch.sum(normal * correct_ParticleNormal , dim=-1))
        Vb = (Vb.reshape(self.intersectedPtnums.size(0),1) * normal)
        Vb -= N_ParticleNormal
        
        # Friction
        friction = 1
        final_friction = N_normal * (1.0 / (friction + 0.5))

        diameter = N_ParticleNormal * 4

        # Setting variables
        bounce = 0.05
        Vb_final = self.particlesTotal[:,3:6].index_select(0, self.intersectedPtnums) + (self.projectedPos - 0.0001 )  # Set new position
        final_v = (self.projectedPos - Vb_final)

        self.particlesTotal[:,0:3].index_copy_(0, self.intersectedPtnums, Vb_final + final_v + final_friction) # INSERT POSITION AT GIVEN INDICES
        self.particlesTotal[:,3:6].index_copy_(0, self.intersectedPtnums, Vb + final_v * bounce + final_friction) # INSERT VELOCITY AT GIVEN INDICES

        self.projectedPos = torch.zeros_like(self.projectedPos) # reset to zeros
        
    def Apply(self):
        self.findIntersection()
        self.projectOntoPrim()
        create_time = time.time() # create
        self.createBoundingBoxes()
        find_time = time.time() # find
        self.findWhichBoundingBox()
        end2_time = time.time() # end
        time_self_collision= time.time()
        self.selfCollision()
        time_self_collision_end = time.time()
        self.reflectVector()

        print("create time: " + str(ptnums) + " particles: " + str(find_time - create_time))
        print("BBox find time: " + str(ptnums) + " particles: " + str(end2_time - find_time))
        print("Self Collision: ", time_self_collision_end - time_self_collision)
        print("----------------------------------------")

class Simulation:
    def __init__(self) -> None:
        self.Forces = []
        self.Constraints = []
        pass

    def InitialState(self):
        self.collisionTotal = torch.zeros(collisionPtnums,7, device='cuda:1') # 7th value is distance
        self.particlesTotal = torch.zeros(ptnums,9, device='cuda:1') # 7th value is boolean if it's intersecting #8th value is bounding box ID

        # collision append
        init_collision_pos = geo1.pointFloatAttribValues("P") 
        t_collision_pos = torch.tensor(init_collision_pos, device='cuda:1')
        self.collisionTotal[:,0:3] = t_collision_pos.reshape(collisionPtnums,3)

        init_collision_norm = geo1.pointFloatAttribValues("N") 
        t_collision_norm = torch.tensor(init_collision_norm, device='cuda:1')
        self.collisionTotal[:,3:6] = t_collision_norm.reshape(collisionPtnums,3)

        # particles append
        init_particles_pos = geo.pointFloatAttribValues("P") 
        t_particles_pos = torch.tensor(init_particles_pos, device='cuda:1')
        self.particlesTotal[:,0:3] = t_particles_pos.reshape(ptnums,3)

        init_particles_norm = geo.pointFloatAttribValues("v") 
        t_particles_norm = torch.tensor(init_particles_norm, device='cuda:1')
        self.particlesTotal[:,3:6] = t_particles_norm.reshape(ptnums,3)
        
        # --- SET MASS ---
        mass = torch.rand(ptnums,1, device='cuda:1') * 10
        self.particlesTotal[:,6] = mass[0,:] # 7th value is mass, 8th is intersection boolean

        self.Forces.append(Gravity(self.particlesTotal))
        # self.Forces.append(Damping(self.particlesTotal))
        # self.Forces.append(Noise(self.particlesTotal))

        #self.Constraints.append(Ground(self.particlesTotal))
        self.Constraints.append(CollisionDetection(self.particlesTotal, self.collisionTotal))

    def update(self):

        # Clean unoccupied GPU memory cache
        torch.cuda.empty_cache()

        sumForce = torch.zeros(ptnums,3, device='cuda:1') # reset all forces

        # Accumulate Forces
        for force in self.Forces:
            a = force.Apply()
            sumForce += torch.add(sumForce, a) *0.1
        
        # Symplectic Euler Integration
        acc = torch.zeros(ptnums,3, device='cuda:1')        
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
# print(torch.cuda.memory_summary(device='cuda:1', abbreviated=True))  

print("Compute time for " + str(ptnums) + " particles: " + str(end_time - start_time))
print("-----------------")
