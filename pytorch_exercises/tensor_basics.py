import torch
import numpy as np

# create a 0D tensor (scalar) from a Python integer
tensor0d = torch.tensor(1)

# create a 1D tensor (vector) from a Python list
tensor1d = torch.tensor([1, 2, 3])

# create a 2D tensor from a nested Python list
tensor2d = torch.tensor([[1, 2], 
                         [3, 4]])

# create a 3D tensor from a nested Python list
tensor3d_1 = torch.tensor([[[1, 2], [3, 4]], 
                           [[5, 6], [7, 8]]])



# create a 3D tensor from NumPy array
ary3d = np.array([[[1, 2], [3, 4]], 
                  [[5, 6], [7, 8]]])

tensor3d_2 = torch.tensor(ary3d)  # Copies NumPy array
tensor3d_3 = torch.from_numpy(ary3d)  # Shares memory with NumPy array

ary3d[0, 0, 0] = 999
print(tensor3d_2) # remains unchanged

print(tensor3d_3) # changes because of memory sharing
