import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, num_filters, fc1_size, fc2_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x