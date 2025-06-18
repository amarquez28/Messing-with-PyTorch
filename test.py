import torch
import numpy as np

#Tensors can be created directly from data. The data type is automatically inferred.
# data = [[1,2],[3,4]]
# x_data = torch.tensor(data)
#Tensors can be created from NumPy arrays and vice versa
# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)


shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeroes_tensor = torch.zeros(shape)

print(f"Random Tensor: \n{rand_tensor}")
print(f"Ones Tensor: \n{ones_tensor}")
print(f"Zeros Tensor: \n{zeroes_tensor}")

#tensors also have attributes that can be accessed 
tensor = torch.rand(3,4)
print(f"Shape: {tensor.shape}")
print(f"Datatype: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#By default, tensors are created on the CPU. We need to explicitly move tensors to the accelerator using .to method 
# (after checking for accelerator availability). Keep in mind that copying large tensors across devices can be expensive in terms of time and memory!

if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
print(f"Device tensor is stored on: {tensor.device}")

