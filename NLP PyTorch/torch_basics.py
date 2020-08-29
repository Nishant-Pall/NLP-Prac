import numpy as np
import torch

# x = torch.empty(size=(3, 3))
# print(x)
# x = torch.zeros((3, 3))
# print(x)
# x = torch.rand((3, 3))
# print(x)
# x = torch.eye(5, 5)
# print(x)
# x = torch.arange(start=0, end=5, step=1)
# print(x)
# x = torch.linspace(start=0.1, end=1, steps=50)
# print(x)
# x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
# print(x)
# x = torch.empty(size=(1, 5)).uniform_(0, 1)
# print(x)
# x = torch.diag(torch.ones(3))
# print(x)

# How to initialize and convert tensors to other types
tensor = torch.arange(4)
# print(tensor.type)
# print(tensor.bool())

print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double())

np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()
print(tensor)
print(np_array_back)
