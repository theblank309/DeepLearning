# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create Fully Connected Layer
class CNN(nn.Module):
    def __init__(self, input_shape, num_classes):

        super(CNN, self).__init__()

        # Input data 
        in_channels = input_shape[-1]
        self.num_classes = num_classes

        # layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )

        in_features = self.feature_extractor(torch.rand(1, *input_shape[::-1])).view(1, -1).shape[-1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):

        x = self.feature_extractor(x)
        x = self.classifier(x)

        return x

