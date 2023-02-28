# Imports
import torch.nn as nn
import torch.nn.functional as F

# Create Fully Connected Layer
class NN(nn.Module):
    def __init__(self, input_shape, num_classes):

        super(NN, self).__init__()
        input_size = input_shape[0] * input_shape[1]
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):

        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

