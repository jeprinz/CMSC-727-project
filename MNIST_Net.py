import torch
import torch.nn as nn
import torch.nn.functional as F

# from:
# https://github.com/pytorch/examples/blob/master/mnist/main.py

class Net(nn.Module):
    def __init__(self, num_filters, fc1_size, fc2_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, 3, 1)
        self.conv2 = nn.Conv2d(num_filters, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
