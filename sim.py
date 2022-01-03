import torch
import numpy as np
import time

torch.manual_seed(0)
ptnums = 1000

init_tensor = torch.rand(ptnums,3, device='cuda')

nearestpoint = torch.zeros(ptnums,1, device='cuda')
final_tensor_i = torch.zeros(1,3, device='cuda')
distance_i = torch.zeros(1,ptnums, device='cuda')
closest_point_num = torch.zeros(1, device='cuda')
final_pos = torch.zeros_like(init_tensor)


# final_tensor_i = torch.sub(init_tensor,mult_tensor[2,:])
# distance_i = torch.sum(final_tensor_i,1)
# print(final_tensor_i)
# print(distance_i)



start_time = time.time()
# Get distance for each point
for i in range(0,ptnums):
    final_tensor_i = torch.sub(init_tensor,init_tensor[i]) # Vector subtract for each tensor
    distance_i = abs(torch.sum(final_tensor_i,1)) # List of distances for each point
    distance_i[distance_i==0] = torch.max(distance_i) # Replace Zero with max
    closest_point_num = torch.argmin(distance_i) # Find smallest distance
    # closest_point_num = closest_point_num.type(torch.int64)
    nearestpoint[i] = closest_point_num
    final_pos[i] = torch.add(init_tensor[i], final_tensor_i[closest_point_num]) # Add direction from closest point to initial position 


    # print("INITIAL TENSOR: " + str(init_tensor))
    # print("SUBTRACTED:" + str(final_tensor_i))
    # print(distance_i)
    # print("in: " + str(distance))



end_time = time.time()

print("---------")
print("Initial position: " + str(init_tensor[9]))
print("Vector to add: " + str(final_tensor_i[closest_point_num]))
print("Final position: " + str(final_pos[9]))
print("--------")
print("Get distances time: " + str(end_time - start_time))

