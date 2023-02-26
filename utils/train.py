import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from configs.configs_reader.configReader import config_map
from models.NN import NN
from models.CNN import CNN
from utils.progressbar import ProgressBar
from dataset.dataset_reader.customDatasetImages import CustomDatasetImages

import traceback
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_map = {
    "NN":NN,
    "CNN":CNN
}

# seperator for print statement
sepr = [80,"-"]

# Exception handling
# --------------------------------------------------------------------------------------------------
def Error_Handler(func):
    def Inner_Function(*args, **kwargs):
        try:
            progressbar = ProgressBar()
            args = tuple(list(args) + [progressbar])
            model = func(*args, **kwargs)
            return model
        except Exception as error:
            progressbar.failed()
            traceback.print_exc()
    return Inner_Function

# Hyper parameter 
# --------------------------------------------------------------------------------------------------
def load_hyperparameter(path):
    with open(path,"r+") as file:
        data = json.load(file)
    config = config_map[data['type']](data)

    print(f"\n{sepr[1]*sepr[0]}\nHyperparameters: \n{sepr[1]*sepr[0]}")
    for key,value in config.__dict__.items():
        print(f"{key}: {value}")
    print(f"{sepr[1]*sepr[0]}\n")
    return config

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

# Save model checkpoint
# --------------------------------------------------------------------------------------------------
def save_checkpoint(model, param):
    checkpoint = {"state_dict":model.state_dict()}
    save_path = f"{param.save_path}\\{param.model_type}_{param.save_mode}.pth"
    torch.save(checkpoint, save_path)

# Select model using save mode
# --------------------------------------------------------------------------------------------------
def select_checkpoint(selected_model, current_model, selected_loss, current_loss, save_mode, epoch):
    save_mode = save_mode.lower()

    if save_mode == "best_model":
        if current_loss < selected_loss or epoch == 0:
            return current_model, current_loss
        return selected_model, selected_loss

    if save_mode == "last_model":
        return current_model, current_loss

# Load model checkpoint
# --------------------------------------------------------------------------------------------------
def load_checkpoint(path, param):
    checkpoint = torch.load(path)
    print(f"Load Model Type: {param.model_type}")
    model_class = model_map[param.model_type]
    model = model_class(input_shape=param.input_shape, num_classes=param.num_classes).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model

# Train model (Intialized model, loss and optimizer)
# --------------------------------------------------------------------------------------------------
@Error_Handler
def train_model(model, criterion, optimizer, train_loader, param, progressbar):

    selected_model = None
    selected_loss = 0
    
    for epoch in range(param.epochs):
        batches_acc = []
        batches_loss = []
        progressbar.pbar(epoch+1, param.epochs, len(train_loader))

        for batch_idx, (x,y) in enumerate(train_loader):

            # Get data to cuda if possible
            data = x.to(device=device)
            targets = y.to(device=device)

            # Forward
            scores = model(data)
            loss = criterion(scores, targets)
            batches_loss.append(loss.item())

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient descent or adam step
            optimizer.step()

            batch_acc = check_accuracy(model, [(x,y)])
            batches_acc.append(batch_acc)
            progressbar.update(loss.item(), batch_acc)
        
        final_acc = sum(batches_acc)/len(batches_acc)
        final_loss = sum(batches_loss)/len(batches_loss)
        progressbar.update(final_loss, final_acc, update_value=0)
        progressbar.close()
        selected_model, selected_loss = select_checkpoint(selected_model, model, selected_loss, final_loss, param.save_mode, epoch)
    save_checkpoint(selected_model, param)

    return model

# Check accuracy of model
# --------------------------------------------------------------------------------------------------
def check_accuracy(model, loader):

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        # Loop through the data
        for x, y in loader:

            # Move data to device
            x = x.to(device=device)
            y = y.to(device=device)

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)

            # Calculate
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return float(num_correct) / num_samples * 100

# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    path = r"C:\Siddhesh\Programming\Machine Learning\DeepLearning\configs\hyperparameter.json"
    param = load_hyperparameter(path)

    # Load train & test data
    image_size = tuple(param.input_shape[:-1])
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    dataset = CustomDatasetImages(root_dir=param.dataset_path, transform=transform)
    train_loader,test_loader = load_custom_data(param.batch_size, dataset, 0.85)

    # Define Model
    model_class = model_map[param.model_type]
    model = model_class(param.input_shape, param.num_classes, param.feature_extractor).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=param.learning_rate)
    model = train_model(model, criterion, optimizer, train_loader, param)

    print(f"Accuracy on training set: {check_accuracy(model, train_loader):.2f}")
    print(f"Accuracy on testing set: {check_accuracy(model, test_loader):.2f}")