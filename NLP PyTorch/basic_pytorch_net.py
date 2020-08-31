# IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # activation
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create fully connected network


class NN(nn.Module):
    def __init__(self, input_size, num_classes):  # (28x28)
        super(NN, self).__init__()  # calls initialization of parent class
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = NN(784, 10)
x = torch.rand(64, 784)
print(model(x))
