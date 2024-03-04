import torch

# Creating a PyTorch tensor from a Python list
my_tensor = torch.tensor([[1, 2, 3],
                          [4, 5, 6]])

print(my_tensor)
print(my_tensor.dtype)
print(type(my_tensor))
print(my_tensor.ndim)