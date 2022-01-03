node = hou.pwd()
geo = node.geometry()
hou.updateProgressAndCheckForInterrupt()
import numpy as np
import torch
import time

# Add code to modify contents of geo.
# Use drop down menu to select examples.

start_time = time.time()
init_array = geo.pointFloatAttribValues("P")
print("get_points: " + str(time.time() - start_time))

torch_init_array = torch.tensor(init_array, device='cuda')
print("convert init to tensor: " + str(time.time() - start_time))
torch_extra_array = torch.rand((len(geo.points()),3), device='cuda')
print("after_rand: " + str(time.time() - start_time))



#torch_extra_array[:,0] = 0
#torch_extra_array[:,2] = 0


torch_init_array = torch_init_array.reshape(len(geo.points()),3)
print("after_reshape: " + str(time.time() - start_time))

      
#add
iter = 1
while iter < 2:
    if iter == 1:
        torch_final_array = torch_init_array
        iter += 1
    else:
        torch_final_array = torch.add(torch_final_array, torch_extra_array)
        iter += 1
print("after_compute: " + str(time.time() - start_time))

final_array = torch.flatten(torch_final_array).tolist()
print(final_array[1])

geo.setPointFloatAttribValues("P", tuple(final_array))

geo.addArrayAttrib(hou.attribType.Global, "data", hou.attribData.Float,  tuple_size=3)
geo.setGlobalAttribValue("data", final_array)



end_time = time.time()
print("Total Time: " + str(end_time - start_time))
print(torch_final_array.size())
print("------")