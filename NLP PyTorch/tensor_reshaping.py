import torch

x = torch.arange(9)

x_3X3 = x.view(3, 3)
print(x_3X3)
x_3X3 = x.reshape(3, 3)
print(x_3X3)

y = x_3X3.t()
print(y)
# print(y.view(9))  # WONT WORK
print(y.contiguous().view(9))
print(y.reshape(9))

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=1))

z = x1.view(-1)  # flatten x1
print(z.shape)
batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

z = x.permute(0, 2, 1)  # dim 0 at 0, dim 2 at 1 and dim 1 at 2
print(z.shape)

x = torch.arange(10)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1)  # 1x1x10
