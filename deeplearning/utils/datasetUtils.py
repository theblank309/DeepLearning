import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Split train & test dataset
# --------------------------------------------------------------------------------------------------
def get_split_data(dataset, split_per):
    train_length = int(len(dataset) * split_per)
    test_length = len(dataset) - train_length
    train_set, test_set = torch.utils.data.random_split(dataset, [train_length, test_length])
    return train_set, test_set

# Load default data 
# --------------------------------------------------------------------------------------------------
def load_data(batch_size, dataset_path=None):
    train_data = datasets.MNIST(root='dataset/dataset/', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data = datasets.MNIST(root='dataset/dataset/', train=True, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    return train_loader,test_loader

# Load custom data 
# --------------------------------------------------------------------------------------------------
def load_custom_data(batch_size, dataset, split_per):
    train_set, test_set = get_split_data(dataset, split_per)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    return train_loader,test_loader