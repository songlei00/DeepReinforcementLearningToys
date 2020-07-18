from torch import nn
import torch.nn.functional as F

class SimpleNet(nn.Module):

    def __init__(self, input_sz=4, output_sz=2):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_sz, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, output_sz)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x