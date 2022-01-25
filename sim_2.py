import torch
import numpy as np
import time

torch.manual_seed(0)

a = torch.rand(50000000,3).chunk(4)
chunks = len(a)

start_time = time.time()
for i in range(0, len(a)):
        a[i][:,1] = 12.0 + i
        

mid_time = time.time()
print(len(a))  
a = torch.cat(a,0)
end_time = time.time()

print("--------")
print("Compute time for " + str(50000000) + " particles with " + str(chunks) + " chunks: "  + str(mid_time - start_time))
print("Concat time for " + str(50000000) + " particles with " + str(chunks) + " chunks: "  + str(end_time - mid_time))
print("Total time: " + str(end_time - start_time))
print("--------")
print("concat:" + str((a.element_size() * a.nelement()) / 1000000) + " MB")
print(a)
