import torch

x = torch.Tensor([1, 2, 3])
y = torch.Tensor([9, 8, 7])

# Addition
z = torch.empty(3)
torch.add(x, y, out=z)
print(z)

z1 = torch.add(x, y)
print(z1)

z3 = x + y

# Subtraction
z = x - y

# Division
z = torch.true_divide(x, y)
print(z)

# inplace operations
t = torch.zeros(3)
# '_' after method means inplace
t.add_(x)
t += x  # t = t + x will create a copy but t += x won't
print(t)

# exponentiation
z = x.pow(2)
print(z)
z = x**2

# simple comparison

z = x > 0
print(z)

# Matrix multiplication

x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)
print(x3)

# matrix exponentiation
matrix_exp = torch.rand(5, 5)
matrix_exp.matrix_power(3)
print(matrix_exp)


# Element wise multiplication

z = x*y
print(z)

# dot product
z = torch.dot(x, y)
print(x)
print(y)
print(z)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)  # (batch, n, p)
print(out_bmm)

# Example of broadcasting
