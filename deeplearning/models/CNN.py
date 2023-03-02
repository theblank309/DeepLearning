# Imports
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

# Supported models
ResNet = ['ResNet18', 'ResNet34']

# Created Identity class to bypass selected layers
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class CNN(nn.Module):
    def __init__(self, input_shape, num_classes, feature_extractor):

        super(CNN, self).__init__()

        # Input data 
        self.num_classes = num_classes

        # Get feature extractor
        if feature_extractor in ResNet:
            print(f"Feature Extractor: {feature_extractor}")
            self.feature_extractor, in_features = self.get_ResNet(feature_extractor)

        else:
            print(f"Feature Extractor: {feature_extractor}")
            self.feature_extractor, in_features = self.get_default(input_shape)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):

        x = self.feature_extractor(x)
        x = self.classifier(x)

        return x
    
    def get_ResNet(self, feature_extractor_model):

        if feature_extractor_model == "ResNet18":
            feature_extractor = torchvision.models.resnet18(pretrained=True)
            for param in feature_extractor.parameters():
                param.requires_grad = False

        if feature_extractor_model == "ResNet34":
            feature_extractor = torchvision.models.resnet18(pretrained=True)
            for param in feature_extractor.parameters():
                param.requires_grad = False

        in_features = feature_extractor.fc.in_features
        feature_extractor.fc = Identity()
        
        return feature_extractor, in_features
    
    def get_default(self, input_shape):

        in_channels = input_shape[-1]
        feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        in_features = feature_extractor(torch.rand(1, *input_shape[::-1])).view(1, -1).shape[-1]

        return feature_extractor, in_features



