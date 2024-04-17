import torch
import torch.nn as nn

class SimpleMlp(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleMlp, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
