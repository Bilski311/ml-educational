import torch
import numpy as np

data = [[1,2], [3,4]]
nparray = np.array(data)
tensor_data = torch.tensor(data)
tensor_data_from_np = torch.from_numpy(nparray)
tensor_data_from_tensor = torch.ones_like(tensor_data)
random_data_from_tensor = torch.rand_like(tensor_data, dtype=torch.float)
print(tensor_data)
print(tensor_data_from_np)
print(tensor_data_from_tensor)
print(random_data_from_tensor)

shape = (4, 2,)
rand_tensor_from_shape = torch.rand(shape)
ones_tensor_from_shape = torch.ones(shape)
zeroes_tensor_from_shape = torch.zeros(shape)
print(rand_tensor_from_shape)
print(rand_tensor_from_shape.shape)
print(rand_tensor_from_shape.dtype)
print(rand_tensor_from_shape.device)
print(ones_tensor_from_shape)
print(zeroes_tensor_from_shape)

# For Mac GPU acceleration
if torch.backends.mps.is_available():
    rand_tensor_from_shape = rand_tensor_from_shape.to("mps")
    print(rand_tensor_from_shape)
    print(rand_tensor_from_shape.device)

tensor = torch.rand(4, 2)
print(f"Whole tensor: {tensor}")
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[:, -1]}")

tensor[:, 1] = 0
print(tensor)
concatenated_tensor = torch.cat([tensor, tensor, tensor], dim=1)
concatenated_tensor_by_rows = torch.cat([tensor, tensor, tensor], dim=0)
print(concatenated_tensor)
print(concatenated_tensor_by_rows)

y1 = tensor @ tensor.T
print(y1)
y2 = tensor.matmul(tensor.T)
print(y2)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(y3)