import torch

batch_size = 10
features = 25
x = torch.rand((batch_size, features))
print(x[0].shape)  # x[0,:]

print(x[:, 0].shape)

print(x[2, 0:10])


# Fancy Indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])

# More Advanced Indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0])

# Usefule operations
print(torch.where(x > 5, x, x*2))
# 0,2,4,6,8,10,6,7,8,9
print(torch.tensor([0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 5]).unique())
print(x.ndimension())  # number of dimensions

print(x.numel())  # number of elements
